"""
lipsync_translate.py
Reads two formats of input .txt files in ../translation/input/:
  1. Word-level Whisper:        [MM:SS.ss --> MM:SS.ss] word
  2. Phrase-level w/ emotion:   [MM:SS.ss --> MM:SS.ss] emotion (conf) | phrase text

Auto-detects per file. Translates each phrase to Spanish with syllable-count
matching (falls back to synonym swap). Writes a unified CSV with timing,
translation, and (when available) emotion tag per phrase.
"""

import re
from pathlib import Path

import nltk
import pandas as pd
import pyphen
from transformers import pipeline


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input"
OUTPUT_DIR = HERE / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CANDIDATES = 10
SYLLABLE_TOLERANCE = 1

for resource in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

from nltk.corpus import wordnet as wn

print("Loading translation model: Helsinki-NLP/opus-mt-en-es")
print("(First run downloads ~300 MB.)")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
en_dic = pyphen.Pyphen(lang="en_US")
es_dic = pyphen.Pyphen(lang="es_ES")


# --- Two input formats ---------------------------------------------------
PHRASE_EMOTION_LINE = re.compile(
    r'\[(\d+):(\d+(?:\.\d+)?)\s*-->\s*(\d+):(\d+(?:\.\d+)?)\]\s*([a-zA-Z_]+)\s*\(([\d.]+)\)\s*\|\s*(.+)'
)
WORD_LINE = re.compile(
    r'\[(\d+):(\d+(?:\.\d+)?)\s*-->\s*(\d+):(\d+(?:\.\d+)?)\]\s*(.+)'
)
SENTENCE_END = re.compile(r'[.!?]$')


def parse_time(m, s):
    return int(m) * 60 + float(s)


def parse_input_file(path):
    """Return list of phrases: {english, start, end, emotion, confidence}."""
    phrases = []
    word_tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = PHRASE_EMOTION_LINE.match(line)
        if m:
            sm, ss, em, es, emo, conf, text = m.groups()
            phrases.append({
                "english": text.strip(),
                "start": parse_time(sm, ss),
                "end": parse_time(em, es),
                "emotion": emo,
                "confidence": float(conf),
            })
            continue
        m = WORD_LINE.match(line)
        if m:
            sm, ss, em, es, word = m.groups()
            word_tokens.append((word.strip(), parse_time(sm, ss), parse_time(em, es)))
    if word_tokens and not phrases:
        current = []
        for w, st, en in word_tokens:
            current.append((w, st, en))
            if SENTENCE_END.search(w):
                phrases.append({
                    "english": " ".join(x[0] for x in current),
                    "start": current[0][1],
                    "end": current[-1][2],
                    "emotion": None,
                    "confidence": None,
                })
                current = []
        if current:
            phrases.append({
                "english": " ".join(x[0] for x in current),
                "start": current[0][1],
                "end": current[-1][2],
                "emotion": None,
                "confidence": None,
            })
    return phrases


# --- Syllable + synonym helpers ------------------------------------------
def count_syllables(text, dic):
    cleaned = re.sub(r"[^\w\s'\-]", "", text)
    return sum(max(1, len(dic.inserted(w).split("-"))) for w in cleaned.split() if w.strip())


def get_spanish_synonyms(word):
    out = set()
    for syn in wn.synsets(word, lang="spa"):
        for lemma in syn.lemmas("spa"):
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                out.add(name)
    return list(out)


def synonym_swap(spanish_text, target_syllables):
    words = spanish_text.split()
    current_diff = abs(count_syllables(spanish_text, es_dic) - target_syllables)
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w]", "", word).lower()
        if len(clean) < 3:
            continue
        for syn in get_spanish_synonyms(clean):
            candidate_words = words.copy()
            candidate_words[i] = syn
            candidate_text = " ".join(candidate_words)
            diff = abs(count_syllables(candidate_text, es_dic) - target_syllables)
            if diff < current_diff:
                spanish_text = candidate_text
                words = candidate_words
                current_diff = diff
                if current_diff == 0:
                    return spanish_text
    return spanish_text


def lipsync_translate_phrase(english_phrase, n=NUM_CANDIDATES):
    if not english_phrase.strip():
        return None
    target = count_syllables(english_phrase, en_dic)
    raw = translator(english_phrase, num_beams=n, num_return_sequences=n, max_new_tokens=128)
    scored = [{"spanish": r["translation_text"],
               "syllables": count_syllables(r["translation_text"], es_dic)} for r in raw]
    for s in scored:
        s["diff"] = abs(target - s["syllables"])
    scored.sort(key=lambda x: x["diff"])
    best = scored[0]
    if best["diff"] > SYLLABLE_TOLERANCE:
        improved = synonym_swap(best["spanish"], target)
        improved_syls = count_syllables(improved, es_dic)
        improved_diff = abs(target - improved_syls)
        if improved_diff < best["diff"]:
            scored.insert(0, {"spanish": improved, "syllables": improved_syls, "diff": improved_diff})
    return {
        "english": english_phrase,
        "english_syllables": target,
        "best_spanish": scored[0]["spanish"],
        "spanish_syllables": scored[0]["syllables"],
        "syllable_diff": scored[0]["diff"],
        "alternatives": " | ".join(s["spanish"] for s in scored[1:5]),
    }


def process_file(path):
    phrases = parse_input_file(path)
    if not phrases:
        print(f"  WARNING: No usable content in {path.name}")
        return []
    rows = []
    for i, ph in enumerate(phrases, 1):
        result = lipsync_translate_phrase(ph["english"])
        if result is None:
            continue
        result["phrase_id"] = f"{path.stem}_{i:03d}"
        result["clip_id"] = path.stem
        result["phrase_start"] = round(ph["start"], 3)
        result["phrase_end"] = round(ph["end"], 3)
        result["phrase_duration"] = round(ph["end"] - ph["start"], 3)
        result["emotion"] = ph["emotion"] or ""
        result["emotion_confidence"] = ph["confidence"] if ph["confidence"] is not None else ""
        result["source_file"] = path.name
        rows.append(result)
    return rows


COLUMNS = ["phrase_id", "clip_id", "phrase_start", "phrase_end", "phrase_duration",
           "english", "best_spanish", "english_syllables", "spanish_syllables",
           "syllable_diff", "alternatives", "emotion", "emotion_confidence", "source_file"]


def main():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files in {INPUT_DIR}.")
        return
    print(f"\nFound {len(txt_files)} input file(s)\n")
    all_rows = []
    for path in txt_files:
        print(f"Processing {path.name}...")
        rows = process_file(path)
        if not rows:
            print()
            continue
        df = pd.DataFrame(rows)[COLUMNS]
        out = OUTPUT_DIR / f"{path.stem}_lipsync.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out}")
        for r in rows[:3]:
            emo = f"  {r['emotion']}" if r['emotion'] else ""
            print(f"    [{r['phrase_id']}]  {r['phrase_start']}s - {r['phrase_end']}s  ({r['phrase_duration']}s){emo}")
            print(f"      EN ({r['english_syllables']}): {r['english']}")
            print(f"      ES ({r['spanish_syllables']}): {r['best_spanish']}  [diff {r['syllable_diff']}]")
        if len(rows) > 3:
            print(f"    ... +{len(rows) - 3} more phrase(s)")
        print()
        all_rows.extend(rows)
    if all_rows:
        pd.DataFrame(all_rows)[COLUMNS].to_csv(OUTPUT_DIR / "all_lipsync_summary.csv", index=False)
        print(f"\nSummary: {OUTPUT_DIR / 'all_lipsync_summary.csv'}")
        print(f"Total phrases processed: {len(all_rows)}")


if __name__ == "__main__":
    main()