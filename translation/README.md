# Translation — Lip-sync Pipeline

Translates English transcripts into Spanish while matching syllable counts to the original so the dubbed audio fits the speaker's mouth movements.

## What it does

Reads `.txt` files from `input/` in either of two formats:

- **Word-level Whisper:** `[MM:SS.ss --> MM:SS.ss] word`
- **Phrase-level with emotion:** `[MM:SS.ss --> MM:SS.ss] emotion (conf) | phrase text`

Auto-detects per file. For each phrase, it:
1. Generates 10 Spanish translation candidates via beam search.
2. Counts syllables in English and each Spanish candidate.
3. Picks the candidate with the smallest syllable-count difference.
4. If still off, tries one-word synonym swaps from Spanish WordNet to nudge it closer.

## Output

Writes per-clip CSVs plus `all_lipsync_summary.csv` in `output/`. Columns:

`phrase_id, clip_id, phrase_start, phrase_end, phrase_duration, english, best_spanish, english_syllables, spanish_syllables, syllable_diff, alternatives, emotion, emotion_confidence, source_file`

The `phrase_id` (e.g. `english6_phrases_001`) is the join key for merging with word-level timestamp data from the alignment component.

## Stack

- Python 3.12
- [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) — English→Spanish translation (~300 MB, CPU)
- `pyphen` — syllable counting (Spanish + English)
- `nltk` + Open Multilingual WordNet — Spanish synonym lookup

## Setup

```bash
source venv/bin/activate
pip install pyphen nltk sentencepiece sacremoses
```

Other deps (`transformers`, `pandas`, `numpy<2`, `torch`) come from the tone analysis setup.

## Usage

1. Drop transcript `.txt` files into `translation/input/`.
2. Run from the project root:

```bash
   python translation/lipsync_translate.py
```

3. Results land in `translation/output/`.