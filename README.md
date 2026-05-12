# video-translation

ACM AI Spring 2026 Project — translates spoken-English video while preserving the speaker's vocal tone.

## Components

- **`tone_analysis/`** — Speech emotion recognition pipeline that classifies the vocal tone of each audio clip. See [tone_analysis/README.md](tone_analysis/README.md) for setup and usage.
- **`translation/`** — Lip-sync translation pipeline. Reads word-level or phrase-level timestamp transcripts, translates each phrase to Spanish using a Hugging Face model with beam search, picks the candidate whose syllable count best matches the English source so the dub fits the original mouth movements. Falls back to synonym swapping when no candidate is close. See [translation/README.md](translation/README.md) for details.
