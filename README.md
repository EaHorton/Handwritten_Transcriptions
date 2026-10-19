# Handwritten Transcriptions

This command-line tool transcribes handwritten document images with an OpenAI
vision model. It requests the original language, spelling, line breaks, and
visible layout, then saves one UTF-8 transcription per image. Suspicious or
low-quality inputs are marked for human review.

## Privacy and responsibility

Images are transmitted to the OpenAI API for processing. Do not use this tool
with documents you are not authorized to upload. Review your organization’s
privacy, legal, and retention requirements before processing sensitive records.

AI-generated transcriptions are an aid, not a legal or archival authority.
Always compare the result with the source image, especially for deeds and
other legal documents. API usage may incur charges, and model pricing can
change.

## Repository and data layout

Keep reusable code in this repository and private documents elsewhere:

```text
handwritten-transcriptions/       # cloned GitHub repository
  ai_vision.py
  hallucination_detector.py
  requirements.txt

~/my-documents/                   # private workspace, not this repository
  images/
    page-001.jpg
  transcriptions/
```

The repository does not contain document images or generated output. The
included `.gitignore` protects common local document and output directories,
but check `git status` before every commit.

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/handwritten-transcriptions.git
cd handwritten-transcriptions

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export OPENAI_API_KEY="your-api-key"
```

Use `.env.example` as a reference for optional environment defaults. The
script reads `OPENAI_API_KEY` and `OPENAI_MODEL` from the shell environment;
it does not load `.env` files automatically.

## Recommended usage

Process images directly from any directory and write results outside the
repository:

```bash
python ai_vision.py \
  --input-dir ~/my-documents/images \
  --output-dir ~/my-documents/transcriptions
```

For another user's documents:

```bash
python ai_vision.py \
  --input-dir ~/my-documents/images \
  --output-dir ~/my-documents/transcriptions
```

Use `--recursive` for nested image directories. Their relative directory
structure is preserved in the output:

```bash
python ai_vision.py \
  --input-dir ~/my-documents \
  --output-dir ~/my-transcriptions \
  --recursive
```

## Options

* `--input-dir PATH` is the recommended generic mode.
* `--output-dir PATH` chooses where transcriptions and summaries are written.
* `--recursive` searches all subdirectories below `--input-dir`.
* `--model MODEL` selects the vision model. The default is `OPENAI_MODEL` or
  `gpt-4o`.
* `--max-image-size PIXELS` limits the longest image dimension sent to the API.
* `--delay SECONDS` controls the pause between requests.

## Output

Each input image produces a `<name>_transcription.txt` file. The output
directory also contains `summaries/usage_summary.json` and
`summaries/usage_summary.txt` with token usage and estimated cost. Estimates
are available only for models listed in `ai_vision.py`; verify current pricing
before relying on them.

`hallucination_detector.py` intentionally does not rewrite model output.
Quality signals are warnings, not proof of an error; preserving the raw
transcription is important for archival review.
