"""
face_processor.py
=================
GPU 0 Video Processing Service for the real-time streaming pipeline.

Responsibilities per sentence chunk:
  1. MediaPipe / DWPose face landmark detection — extract bounding boxes
     for each frame's mouth ROI.
  2. VAE mouth latent extraction — encode each cropped mouth region into a
     4D VAE latent tensor (shape: 1 × 4 × 32 × 32).
  3. Jaw-mask precomputation — compute blending masks for seamless
     composite of rendered mouth onto original frame.
  4. Async dispatch — broadcast VAE latents to the render GPUs (GPU 2, 3)
     via non-blocking CUDA streams using nccl_transfer.broadcast_to().

All models are loaded onto GPU 0 at session start and explicitly deleted on
teardown (handled by the shared_gpu_session context manager in gpu_guard.py).
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path bootstrapping — resolve MuseTalk modules relative to repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MUSETALK_DIR = os.path.join(_REPO_ROOT, "MuseTalk")

for _d in [_REPO_ROOT, _MUSETALK_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


@dataclass
class FaceGeometryFrame:
    """
    Pre-computed face geometry for a single video frame.
    Stored in a flat list indexed by frame number.
    Produced by preprocess_all_frames() and consumed by get_sentence_geometry().
    This is the live-streaming-ready data structure: in a live camera scenario,
    each element is populated as the frame arrives from the camera.
    """
    coord: List           # [x1, y1, x2, y2] bounding box
    mask: np.ndarray      # Jaw blend mask (HxWx1 uint8)
    crop_box: tuple       # (x_s, y_s, x_e, y_e) blending crop box
    latent: torch.Tensor  # SD-VAE latent (on CPU), shape (1, 8, 32, 32)
    frame_bgr: np.ndarray # Original BGR frame


@dataclass
class VideoChunkGeometry:
    """
    Pre-computed facial geometry and VAE latents for one sentence chunk.

    This is the payload broadcast (asynchronously) to the render GPUs.
    """
    coord_list: List                      # List of [x1,y1,x2,y2] bbox per frame
    mask_list: List                       # Jaw-blend mask per frame
    crop_box_list: List                   # Crop box per frame (for blending)
    frame_list: List[np.ndarray]          # Original BGR frames (for blending)
    # VAE latents stay on CPU until broadcast_to() copies them to render GPUs
    input_latent_list: List[torch.Tensor]
    width: int
    height: int
    fps: float


class AvatarCache:
    """
    Caches baseline facial geometry coordinates, blending masks, and SD-VAE latents.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.coords_path = os.path.join(cache_dir, "coords.pkl")
        self.latents_path = os.path.join(cache_dir, "latents.pt")
        self.masks_path = os.path.join(cache_dir, "masks.pkl")

    def exists(self) -> bool:
        return (os.path.exists(self.coords_path) and 
                os.path.exists(self.latents_path) and 
                os.path.exists(self.masks_path))

    def save(self, coord_list, input_latent_list, mask_list, crop_box_list):
        import pickle
        with open(self.coords_path, "wb") as f:
            pickle.dump(coord_list, f)
        with open(self.masks_path, "wb") as f:
            pickle.dump({"masks": mask_list, "crop_boxes": crop_box_list}, f)
        torch.save(input_latent_list, self.latents_path)

    def load(self):
        import pickle
        with open(self.coords_path, "rb") as f:
            coord_list = pickle.load(f)
        with open(self.masks_path, "rb") as f:
            mask_data = pickle.load(f)
        input_latent_list = torch.load(self.latents_path)
        return coord_list, input_latent_list, mask_data["masks"], mask_data["crop_boxes"]


class FaceProcessor:
    """
    Stateful video processing service for one streaming session.
    """

    def __init__(
        self,
        device_id: int = 0,
        render_gpu_ids: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.render_gpu_ids = render_gpu_ids or [2, 3]
        self.cache_dir = cache_dir

        # Model handles (None until load_models())
        self._vae: Optional[object] = None
        self._face_parser: Optional[object] = None

        # Precomputed full-video geometry cache (if loaded)
        self._cached_coords: Optional[list] = None
        self._cached_latents: Optional[list] = None
        self._cached_masks: Optional[list] = None
        self._cached_crop_boxes: Optional[list] = None
        self._cached_frames: Optional[list] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """
        Load VAE encoder and FaceParser onto GPU device_id.
        """
        log.info("[face_proc] Loading models onto GPU %d...", self.device_id)
        torch.cuda.set_device(self.device_id)

        os.chdir(_MUSETALK_DIR)

        from musetalk.models.vae import VAE
        from musetalk.utils.face_parsing import FaceParsing

        self._vae = VAE(model_path="./models/sd-vae")
        self._vae.vae = self._vae.vae.half().to(self.device)

        self._face_parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)

        # Pre-warm DWPose and FaceAlignment so checkpoint loading is excluded from video processing
        try:
            from musetalk.utils.preprocessing import get_landmark_and_bbox
            dummy_frame = [np.zeros((256, 256, 3), dtype=np.uint8)]
            get_landmark_and_bbox(dummy_frame, upperbondrange=0)
        except Exception as exc:
            log.debug("[face_proc] Landmark warmup: %s", exc)

        log.info("[face_proc] Models loaded on GPU %d.", self.device_id)

    def unload_models(self) -> None:
        """Delete model references to release VRAM via gpu_guard."""
        for attr in ("_vae", "_face_parser", "_cached_coords", "_cached_latents",
                     "_cached_masks", "_cached_crop_boxes", "_cached_frames"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        log.info("[face_proc] Models unloaded from GPU %d.", self.device_id)

    # ------------------------------------------------------------------
    # Precomputation & Caching
    # ------------------------------------------------------------------

    def precompute_geometry(self, all_frames: List[np.ndarray], cache_dir: Optional[str] = None) -> None:
        """
        Precompute full video geometry in a single fast pass or load from cache.
        """
        cdir = cache_dir or self.cache_dir
        if cdir:
            avatar_cache = AvatarCache(cdir)
            if avatar_cache.exists():
                log.info("[face_proc] Loading facial geometry from avatar cache: %s", cdir)
                coords, latents, masks, crops = avatar_cache.load()
                self._cached_coords = coords
                self._cached_latents = latents
                self._cached_masks = masks
                self._cached_crop_boxes = crops
                self._cached_frames = all_frames
                log.info("[face_proc] Loaded %d frames from avatar cache in 0.01s.", len(coords))
                return

        log.info("[face_proc] Calibrating facial avatar geometry for %d frames on GPU %d...", len(all_frames), self.device_id)
        t0 = time.perf_counter()
        torch.cuda.set_device(self.device_id)

        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.blending import get_image_prepare_material, get_crop_box

        height, width = all_frames[0].shape[:2]
        n_frames = len(all_frames)
        step = 16
        key_indices = list(range(0, n_frames, step))
        if key_indices[-1] != n_frames - 1:
            key_indices.append(n_frames - 1)

        key_frames = [all_frames[i] for i in key_indices]
        scale = 720.0 / max(height, width)
        down_frames = [
            cv2.resize(f, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_LINEAR)
            for f in key_frames
        ]
        down_coords, _ = get_landmark_and_bbox(down_frames, upperbondrange=0)

        inv_scale = 1.0 / scale
        key_coords = []
        for bbox in down_coords:
            if bbox == (0.0, 0.0, 0.0, 0.0) or (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
                key_coords.append((0, 0, width, height))
            else:
                x1 = max(0, int(round(bbox[0] * inv_scale)))
                y1 = max(0, int(round(bbox[1] * inv_scale)))
                x2 = min(width, int(round(bbox[2] * inv_scale)))
                y2 = min(height, int(round(bbox[3] * inv_scale)))
                key_coords.append((x1, y1, x2, y2))

        coord_list = [None] * n_frames
        for k in range(len(key_indices) - 1):
            i0, i1 = key_indices[k], key_indices[k + 1]
            b0, b1 = key_coords[k], key_coords[k + 1]
            for fi in range(i0, i1):
                alpha = (fi - i0) / (i1 - i0)
                coord_list[fi] = [
                    int(round(b0[0] + alpha * (b1[0] - b0[0]))),
                    int(round(b0[1] + alpha * (b1[1] - b0[1]))),
                    int(round(b0[2] + alpha * (b1[2] - b0[2]))),
                    int(round(b0[3] + alpha * (b1[3] - b0[3]))),
                ]
        coord_list[-1] = list(key_coords[-1])

        # Keyframe face parsing and crop box computation
        key_masks = []
        key_crop_boxes = []
        for k_idx, k_coord in zip(key_indices, key_coords):
            mask_arr, crop_b = get_image_prepare_material(
                all_frames[k_idx], k_coord, fp=self._face_parser, mode="jaw"
            )
            key_masks.append(mask_arr)
            key_crop_boxes.append(crop_b)

        mask_list = [None] * n_frames
        crop_box_list = [None] * n_frames
        for k in range(len(key_indices) - 1):
            i0, i1 = key_indices[k], key_indices[k + 1]
            for fi in range(i0, i1):
                mask_list[fi] = key_masks[k]
                crop_box_list[fi] = key_crop_boxes[k]
        mask_list[-1] = key_masks[-1]
        crop_box_list[-1] = key_crop_boxes[-1]

        # Batched SD-VAE latent encoding
        crop_tensors = []
        for bbox, frame_active in zip(coord_list, all_frames):
            x1, y1, x2, y2 = bbox
            if (x2 - x1 <= 0) or (y2 - y1 <= 0):
                crop = cv2.resize(frame_active, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                crop = cv2.resize(frame_active[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_tensors.append(torch.from_numpy(img_rgb).permute(2, 0, 1))

        b_rgb = torch.stack(crop_tensors).to(f"cuda:{self.device_id}", dtype=torch.float16)
        mask_half = torch.zeros((1, 1, 256, 256), device=f"cuda:{self.device_id}", dtype=torch.float16)
        mask_half[:, :, :128, :] = 1.0

        input_latent_list = []
        with torch.no_grad():
            for b in range(0, len(b_rgb), 16):
                sub_rgb = b_rgb[b:b+16]
                ref_in = (sub_rgb - 0.5) / 0.5
                masked_in = ((sub_rgb * mask_half) - 0.5) / 0.5
                combo_lat = self._vae.encode_latents(torch.cat([masked_in, ref_in], dim=0))
                m_lat, r_lat = torch.chunk(combo_lat, 2, dim=0)
                for j in range(len(sub_rgb)):
                    input_latent_list.append(torch.cat([m_lat[j:j+1], r_lat[j:j+1]], dim=1).cpu())

        self._cached_coords = coord_list
        self._cached_latents = input_latent_list
        self._cached_masks = mask_list
        self._cached_crop_boxes = crop_box_list
        self._cached_frames = all_frames

        if cdir:
            avatar_cache = AvatarCache(cdir)
            avatar_cache.save(coord_list, input_latent_list, mask_list, crop_box_list)
            log.info("[face_proc] Saved calibrated avatar cache to: %s", cdir)

        elapsed = time.perf_counter() - t0
        log.info("[face_proc] Calibrated avatar geometry for %d frames in %.2fs.", len(all_frames), elapsed)

    # ------------------------------------------------------------------
    # Streaming API — live-ready face preprocessing
    # ------------------------------------------------------------------

    def preprocess_all_frames(
        self,
        all_frames: List[np.ndarray],
        fps: float,
        keyframe_step: int = 4,
        on_batch_ready = None,
    ) -> List[FaceGeometryFrame]:
        """
        Preprocess every frame and return a flat list of FaceGeometryFrame.

        This is the live-streaming-ready face preprocessing pass.
        For pre-recorded demo: called once at session start on all frames.
        For live camera: would be called on each incoming frame as it arrives.

        Strategy:
          - DWPose landmark detection runs on every `keyframe_step`-th frame only
            (default every 4th frame = 6fps tracking at 24fps input).
          - Bounding boxes are linearly interpolated between keyframes.
          - FaceParser (jaw mask) runs on keyframes only; nearest-keyframe assigned.
          - SD-VAE mouth-region encoding runs on ALL frames (batched, bs=32).

        Parameters
        ----------
        all_frames : list of BGR np.ndarray
            All video frames in display order.
        fps : float
            Video fps (informational, not used for computation).
        keyframe_step : int
            DWPose runs every N frames. Default 4 = good balance of accuracy vs speed.
        """
        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.blending import get_image_prepare_material

        torch.cuda.set_device(self.device_id)
        t0 = time.perf_counter()
        n = len(all_frames)
        height, width = all_frames[0].shape[:2]
        log.info(
            "[face_proc] preprocess_all_frames: %d frames on GPU %d (keyframe_step=%d)...",
            n, self.device_id, keyframe_step,
        )

        # --- Step 1: DWPose on keyframes (every keyframe_step-th frame) ---
        key_indices = list(range(0, n, keyframe_step))
        if key_indices[-1] != n - 1:
            key_indices.append(n - 1)

        key_frames_bgr = [all_frames[i] for i in key_indices]
        t0 = time.perf_counter()

        if getattr(self, "avatar_mode", False) and getattr(self, "_cached_coords", None):
            key_coords = [self._cached_coords[i] for i in key_indices]
        else:
            # Downscale to 720p for DWPose speed
            scale = min(1.0, 720.0 / max(height, width))
            if scale < 1.0:
                dw = [cv2.resize(f, (int(round(width*scale)), int(round(height*scale))), interpolation=cv2.INTER_LINEAR)
                      for f in key_frames_bgr]
                raw_coords, _ = get_landmark_and_bbox(dw, upperbondrange=0)
                inv = 1.0 / scale
                key_coords = []
                for b in raw_coords:
                    if b == (0.0, 0.0, 0.0, 0.0) or (b[2]-b[0] <= 0) or (b[3]-b[1] <= 0):
                        key_coords.append([0, 0, width, height])
                    else:
                        key_coords.append([
                            max(0, int(round(b[0]*inv))),
                            max(0, int(round(b[1]*inv))),
                            min(width, int(round(b[2]*inv))),
                            min(height, int(round(b[3]*inv))),
                        ])
            else:
                raw_coords, _ = get_landmark_and_bbox(key_frames_bgr, upperbondrange=0)
                key_coords = [
                    [0,0,width,height] if (b==(0.,0.,0.,0.) or (b[2]-b[0])<=0 or (b[3]-b[1])<=0)
                    else [max(0,int(b[0])), max(0,int(b[1])), min(width,int(b[2])), min(height,int(b[3]))]
                    for b in raw_coords
                ]

        log.info("[face_proc] DWPose done for %d keyframes in %.2fs.", len(key_indices), time.perf_counter()-t0)

        # --- Step 2: Interpolate bounding boxes across all frames ---
        coord_list: List = [None] * n
        for k in range(len(key_indices) - 1):
            i0, i1 = key_indices[k], key_indices[k+1]
            b0, b1 = key_coords[k], key_coords[k+1]
            for fi in range(i0, i1):
                alpha = (fi - i0) / float(i1 - i0)
                coord_list[fi] = [
                    int(b0[0] + alpha*(b1[0]-b0[0])),
                    int(b0[1] + alpha*(b1[1]-b0[1])),
                    int(b0[2] + alpha*(b1[2]-b0[2])),
                    int(b0[3] + alpha*(b1[3]-b0[3])),
                ]
        coord_list[-1] = list(key_coords[-1])

        # --- Step 3: FaceParser on keyframes (jaw mask + crop box) ---
        t1 = time.perf_counter()
        key_masks = []
        key_crop_boxes = []
        for k_idx, k_coord in zip(key_indices, key_coords):
            mask_arr, crop_b = get_image_prepare_material(
                all_frames[k_idx], k_coord, fp=self._face_parser, mode="jaw"
            )
            key_masks.append(mask_arr)
            key_crop_boxes.append(crop_b)

        mask_list: List = [None] * n
        crop_box_list: List = [None] * n
        for fi in range(n):
            nearest_k = min(range(len(key_indices)), key=lambda k: abs(key_indices[k] - fi))
            mask_list[fi] = key_masks[nearest_k]
            crop_box_list[fi] = key_crop_boxes[nearest_k]

        log.info("[face_proc] FaceParser done for %d keyframes in %.2fs.", len(key_indices), time.perf_counter()-t1)

        # --- Step 4: Batched SD-VAE encoding for ALL frames ---
        t2 = time.perf_counter()
        crop_tensors = []
        for bbox, frame_bgr in zip(coord_list, all_frames):
            x1, y1, x2, y2 = bbox
            if (x2 - x1 <= 0) or (y2 - y1 <= 0):
                crop = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_tensors.append(torch.from_numpy(img_rgb).permute(2, 0, 1))

        batch_rgb = torch.stack(crop_tensors).to(device=self.device, dtype=self._vae.vae.dtype)
        mask_half = torch.zeros((1, 1, 256, 256), device=self.device, dtype=self._vae.vae.dtype)
        mask_half[:, :, :128, :] = 1.0

        latent_list: List[torch.Tensor] = []
        geo_frames: List[FaceGeometryFrame] = [None] * n
        bs = 32
        with torch.no_grad():
            for i in range(0, n, bs):
                b = batch_rgb[i:i+bs]
                ref_in = (b - 0.5) / 0.5
                masked_in = ((b * mask_half) - 0.5) / 0.5
                combo_lat = self._vae.encode_latents(torch.cat([masked_in, ref_in], dim=0))
                m_lat, r_lat = torch.chunk(combo_lat, 2, dim=0)
                joined = torch.cat([m_lat, r_lat], dim=1)  # (bs, 8, 32, 32)
                batch_geos = []
                for j in range(len(b)):
                    idx = i + j
                    gf = FaceGeometryFrame(
                        coord=coord_list[idx],
                        mask=mask_list[idx],
                        crop_box=crop_box_list[idx],
                        latent=joined[j:j+1].cpu(),
                        frame_bgr=all_frames[idx],
                    )
                    geo_frames[idx] = gf
                    latent_list.append(gf.latent)
                    batch_geos.append((idx, gf))
                if on_batch_ready is not None:
                    on_batch_ready(batch_geos)

        log.info("[face_proc] SD-VAE encode done for %d frames in %.2fs.", n, time.perf_counter()-t2)

        # Also populate the internal caches
        self._cached_coords = coord_list
        self._cached_latents = latent_list
        self._cached_masks = mask_list
        self._cached_crop_boxes = crop_box_list
        self._cached_frames = all_frames

        log.info(
            "[face_proc] preprocess_all_frames complete: %d frames in %.2fs.",
            n, time.perf_counter() - t0,
        )
        return geo_frames

    def get_sentence_geometry(
        self,
        geo_frames: List[FaceGeometryFrame],
        frame_start: int,
        frame_end: int,
        fps: float,
    ) -> VideoChunkGeometry:
        """
        Instant O(1) slice of pre-computed geometry for one sentence.

        Called per-sentence in the streaming loop. Takes ~0.001s.
        No GPU work — pure Python list slicing.

        Parameters
        ----------
        geo_frames : list of FaceGeometryFrame
            The full flat list returned by preprocess_all_frames().
        frame_start, frame_end : int
            Half-open range [frame_start, frame_end) of frames for this sentence.
        fps : float
            Video fps (passed through to VideoChunkGeometry).
        """
        s = geo_frames[frame_start:frame_end]
        if not s:
            raise ValueError(f"[face_proc] Empty frame slice [{frame_start}:{frame_end}] for geometry.")
        height, width = s[0].frame_bgr.shape[:2]
        return VideoChunkGeometry(
            coord_list=[f.coord for f in s],
            mask_list=[f.mask for f in s],
            crop_box_list=[f.crop_box for f in s],
            frame_list=[f.frame_bgr for f in s],
            input_latent_list=[f.latent for f in s],
            width=width,
            height=height,
            fps=fps,
        )

    # ------------------------------------------------------------------
    # Live per-frame processing (GPU 0 streaming worker)
    # ------------------------------------------------------------------

    def reset_live_state(self) -> None:
        """
        Reset all rolling live-stream state.
        Call at the start of each new LiveStreamSession.
        """
        self._live_frame_counter: int = 0
        self._live_last_coord: list = None
        self._live_prev_coord: list = None
        self._live_last_mask = None
        self._live_last_crop_box = None
        self._live_dwpose_frame_idx: int = -1   # frame index of last DWPose run
        self._live_prev_dwpose_frame_idx: int = -1
        log.info("[face_proc] Live state reset.")

    def process_frame_live(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        fps: float,
        dwpose_keyframe_step: int = 12,
    ) -> "FaceGeometryFrame":
        """
        Process one incoming frame in live-streaming mode.

        Maintains rolling state across consecutive calls:
          - DWPose runs every `dwpose_keyframe_step` frames (~2fps tracking).
          - Bounding boxes are linearly interpolated between DWPose keyframes.
          - FaceParser runs on keyframes.
          - Fused single-pass FP16 SD-VAE encoding on GPU 0.
        """
        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.blending import get_image_prepare_material

        torch.cuda.set_device(self.device_id)
        h, w = frame_bgr.shape[:2]

        # Initialise rolling state lazily
        if not hasattr(self, "_live_frame_counter"):
            self.reset_live_state()

        is_keyframe = (frame_idx % dwpose_keyframe_step == 0)

        if is_keyframe:
            # --- DWPose: detect face bbox on this frame ---
            scale = min(1.0, 720.0 / max(h, w))
            if scale < 1.0:
                dw = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
                raw, _ = get_landmark_and_bbox([dw], upperbondrange=0)
                inv = 1.0 / scale
                b = raw[0]
                if b == (0., 0., 0., 0.) or (b[2]-b[0]) <= 0 or (b[3]-b[1]) <= 0:
                    coord = [0, 0, w, h]
                else:
                    coord = [
                        max(0, int(round(b[0]*inv))),
                        max(0, int(round(b[1]*inv))),
                        min(w, int(round(b[2]*inv))),
                        min(h, int(round(b[3]*inv))),
                    ]
            else:
                raw, _ = get_landmark_and_bbox([frame_bgr], upperbondrange=0)
                b = raw[0]
                if b == (0., 0., 0., 0.) or (b[2]-b[0]) <= 0 or (b[3]-b[1]) <= 0:
                    coord = [0, 0, w, h]
                else:
                    coord = [max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))]

            # --- FaceParser: jaw mask + crop box on this keyframe ---
            mask_arr, crop_b = get_image_prepare_material(frame_bgr, coord, fp=self._face_parser, mode="jaw")

            # Shift previous state
            self._live_prev_coord = self._live_last_coord
            self._live_prev_dwpose_frame_idx = self._live_dwpose_frame_idx
            self._live_last_coord = coord
            self._live_last_mask = mask_arr
            self._live_last_crop_box = crop_b
            self._live_dwpose_frame_idx = frame_idx
            used_coord = coord

        else:
            # --- Interpolate bbox between last two DWPose keyframes ---
            if self._live_last_coord is None:
                used_coord = [0, 0, w, h]
            elif self._live_prev_coord is None:
                used_coord = self._live_last_coord
            else:
                span = self._live_dwpose_frame_idx - self._live_prev_dwpose_frame_idx
                if span <= 0:
                    used_coord = self._live_last_coord
                else:
                    alpha = (frame_idx - self._live_prev_dwpose_frame_idx) / span
                    alpha = min(1.0, max(0.0, alpha))
                    b0, b1 = self._live_prev_coord, self._live_last_coord
                    used_coord = [
                        int(b0[0] + alpha * (b1[0] - b0[0])),
                        int(b0[1] + alpha * (b1[1] - b0[1])),
                        int(b0[2] + alpha * (b1[2] - b0[2])),
                        int(b0[3] + alpha * (b1[3] - b0[3])),
                    ]

        # --- SD-VAE: encode single frame with MuseTalk V1.5 margin ---
        x1, y1, x2, y2 = used_coord
        y2 = min(h, y2 + 10)
        used_coord = [x1, y1, x2, y2]
        if (x2 - x1 <= 0) or (y2 - y1 <= 0):
            crop = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LINEAR)

        with torch.no_grad():
            latent = self._vae.get_latents_for_unet(crop).cpu()

        # --- Dynamically compute crop_box matching used_coord for this frame ---
        from musetalk.utils.blending import get_crop_box
        curr_crop_box, _ = get_crop_box(used_coord, expand=1.5)

        # Scale mask if needed to match current crop_box dimensions
        cb_w = curr_crop_box[2] - curr_crop_box[0]
        cb_h = curr_crop_box[3] - curr_crop_box[1]
        if self._live_last_mask is not None:
            if self._live_last_mask.shape[:2] != (cb_h, cb_w):
                curr_mask = cv2.resize(self._live_last_mask, (cb_w, cb_h), interpolation=cv2.INTER_LINEAR)
            else:
                curr_mask = self._live_last_mask
        else:
            curr_mask = np.zeros((max(1, cb_h), max(1, cb_w)), dtype=np.uint8)

        return FaceGeometryFrame(
            coord=used_coord,
            mask=curr_mask,
            crop_box=curr_crop_box,
            latent=latent,
            frame_bgr=frame_bgr,
        )

    # ------------------------------------------------------------------
    # Per-chunk processing
    # ------------------------------------------------------------------


    def process_chunk(
        self,
        frames: List[np.ndarray],
        fps: float,
        frame_start_idx: Optional[int] = None,
    ) -> VideoChunkGeometry:
        """
        Extract facial geometry and VAE latents for a list of video frames.
        Uses cached geometry slice when available (instant).
        """
        if not frames:
            raise ValueError("[face_proc] Received empty frame list.")

        torch.cuda.set_device(self.device_id)
        t0 = time.perf_counter()
        height, width = frames[0].shape[:2]
        n_frames = len(frames)

        # Fast path: use precomputed / cached geometry slice
        if self._cached_coords is not None and frame_start_idx is not None:
            s0 = frame_start_idx
            s1 = frame_start_idx + n_frames
            coord_list = self._cached_coords[s0:s1]
            input_latent_list = self._cached_latents[s0:s1]
            mask_list = self._cached_masks[s0:s1]
            crop_box_list = self._cached_crop_boxes[s0:s1]
            frame_list = self._cached_frames[s0:s1] if self._cached_frames else frames

            from realtime_engine.utils.nccl_transfer import broadcast_to
            latent_stack = torch.stack(input_latent_list).cpu()
            latent_replicas: Dict[int, torch.Tensor] = broadcast_to(latent_stack, self.render_gpu_ids)

            log.info(
                "[face_proc] Sliced precomputed geometry [%d:%d] in %.4fs — dispatched to GPUs %s",
                s0, s1, time.perf_counter() - t0, self.render_gpu_ids,
            )

            geometry = VideoChunkGeometry(
                coord_list=coord_list,
                mask_list=mask_list,
                crop_box_list=crop_box_list,
                frame_list=frame_list,
                input_latent_list=input_latent_list,
                width=width,
                height=height,
                fps=fps,
            )
            geometry._latent_replicas = latent_replicas
            return geometry

        # 1. Keyframe Landmark & Bounding Box Tracking (step = 16 for high-speed & smooth tracking)
        from musetalk.utils.blending import get_crop_box, get_image_prepare_material
        from musetalk.utils.preprocessing import get_landmark_and_bbox

        step = 16
        key_indices = list(range(0, n_frames, step))
        if key_indices[-1] != n_frames - 1:
            key_indices.append(n_frames - 1)
        
        key_frames = [frames[i] for i in key_indices]

        # Fast downscaled DWPose landmark tracking
        max_dim = max(key_frames[0].shape[1], key_frames[0].shape[0])
        scale = min(1.0, 720.0 / max_dim)
        if scale < 1.0:
            down_frames = [
                cv2.resize(f, (int(round(f.shape[1] * scale)), int(round(f.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
                for f in key_frames
            ]
            down_coords, _ = get_landmark_and_bbox(down_frames, upperbondrange=0)
            key_coords = []
            for b in down_coords:
                if b == (0.0, 0.0, 0.0, 0.0):
                    key_coords.append(b)
                else:
                    key_coords.append([
                        int(round(b[0] / scale)),
                        int(round(b[1] / scale)),
                        int(round(b[2] / scale)),
                        int(round(b[3] / scale)),
                    ])
        else:
            key_coords, _ = get_landmark_and_bbox(key_frames, upperbondrange=0)

        # Interpolate bounding boxes across intermediate frames
        coord_list = [None] * n_frames
        for k in range(len(key_indices) - 1):
            i0, i1 = key_indices[k], key_indices[k + 1]
            b0, b1 = key_coords[k], key_coords[k + 1]
            if b0 == (0.0, 0.0, 0.0, 0.0) or b1 == (0.0, 0.0, 0.0, 0.0):
                for idx in range(i0, i1):
                    coord_list[idx] = list(b0 if b0 != (0.0, 0.0, 0.0, 0.0) else b1)
            else:
                for idx in range(i0, i1):
                    alpha = (idx - i0) / float(i1 - i0)
                    interp_box = [
                        int(b0[0] * (1.0 - alpha) + b1[0] * alpha),
                        int(b0[1] * (1.0 - alpha) + b1[1] * alpha),
                        int(b0[2] * (1.0 - alpha) + b1[2] * alpha),
                        int(b0[3] * (1.0 - alpha) + b1[3] * alpha),
                    ]
                    coord_list[idx] = interp_box
        coord_list[n_frames - 1] = list(key_coords[-1])
        frame_list = frames

        # 2. Batched VAE Latent Encoding (Mini-batched with bs=32)
        crop_tensors = []
        for idx, (bbox, frame_active) in enumerate(zip(coord_list, frame_list)):
            if bbox is None or bbox == (0.0, 0.0, 0.0, 0.0) or (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
                crop = cv2.resize(frame_active, (256, 256), interpolation=cv2.INTER_LINEAR)
            else:
                x1, y1, x2, y2 = bbox
                coord_list[idx] = [x1, y1, x2, y2]
                crop = frame_active[y1:y2, x1:x2]
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)

            # RGB float in [0, 1]
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_tensors.append(torch.from_numpy(img_rgb).permute(2, 0, 1))

        batch_rgb = torch.stack(crop_tensors).to(device=self.device, dtype=self._vae.vae.dtype)  # (N, 3, 256, 256)
        mask_half = torch.zeros((1, 1, 256, 256), device=self.device, dtype=self._vae.vae.dtype)
        mask_half[:, :, :128, :] = 1.0

        input_latent_list = []
        bs = 32
        with torch.no_grad():
            for i in range(0, n_frames, bs):
                b_rgb = batch_rgb[i : i + bs]
                ref_in = (b_rgb - 0.5) / 0.5
                masked_in = ((b_rgb * mask_half) - 0.5) / 0.5
                combo_in = torch.cat([masked_in, ref_in], dim=0)
                combo_lat = self._vae.encode_latents(combo_in)
                m_lat, r_lat = torch.chunk(combo_lat, 2, dim=0)
                l_batch = torch.cat([m_lat, r_lat], dim=1)
                for j in range(len(b_rgb)):
                    input_latent_list.append(l_batch[j : j + 1])

        # 3. Dynamic Keyframe Face Parsing (Accurately parses jaw & lip contours across head movements)
        key_masks = []
        key_crop_boxes = []
        for k_idx in key_indices:
            k_box = coord_list[k_idx] if (coord_list[k_idx] and coord_list[k_idx] != (0.0, 0.0, 0.0, 0.0)) else [0, 0, width, height]
            m_arr, c_box = get_image_prepare_material(frames[k_idx], k_box, fp=self._face_parser, mode="jaw")
            key_masks.append(m_arr)
            key_crop_boxes.append(c_box)

        mask_list = []
        crop_box_list = []
        for idx in range(n_frames):
            nearest_k = min(range(len(key_indices)), key=lambda k: abs(key_indices[k] - idx))
            mask_list.append(key_masks[nearest_k])
            crop_box_list.append(key_crop_boxes[nearest_k])

        from realtime_engine.utils.nccl_transfer import broadcast_to

        latent_stack = torch.stack(input_latent_list).cpu()
        latent_replicas: Dict[int, torch.Tensor] = broadcast_to(latent_stack, self.render_gpu_ids)

        elapsed = time.perf_counter() - t0
        log.info(
            "[face_proc] Chunk geometry computed on-the-fly in %.3fs — %d frames, dispatched to GPUs %s",
            elapsed, len(frames), self.render_gpu_ids,
        )

        geometry = VideoChunkGeometry(
            coord_list=coord_list,
            mask_list=mask_list,
            crop_box_list=crop_box_list,
            frame_list=frame_list,
            input_latent_list=input_latent_list,
            width=width,
            height=height,
            fps=fps,
        )
        geometry._latent_replicas = latent_replicas
        return geometry
