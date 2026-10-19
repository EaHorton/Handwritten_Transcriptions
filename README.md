# Handwritten Transcriptions

This program uses an OpenAI vision model to transcribe handwritten document
images. It preserves the original language, spelling, punctuation, line
breaks, and visible layout as closely as possible.

You do not need to know how to program to use it, but you will need to:

1. Install Python.
2. Copy this project to your computer.
3. Install the project’s dependencies.
4. Create an OpenAI API key and add billing to your OpenAI account.
5. Run one command from a terminal.

The instructions below explain each step.

## Important privacy and accuracy information

Images are sent to the OpenAI API for processing. Only upload documents that
you are authorized to send to an outside service. Check your organization’s
privacy, legal, and records-retention requirements before using this tool.

The result is an AI-assisted transcription, not a certified transcription.
Always compare it with the original image. This is especially important for
deeds, contracts, legal records, names, dates, numbers, and financial
information.

API requests may cost money. The amount depends on the model, image size, and
number of images. Review OpenAI’s current pricing before processing a large
collection.

## What you need

### A computer

The instructions work on:

- macOS
- Windows 10 or later
- Linux

### Python

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

On Windows, enable **Add Python to PATH** during installation.

After installation, open a terminal and check that Python is available:

macOS/Linux:

```bash
python3 --version
```

Windows:

```powershell
python --version
```

You should see a version such as `Python 3.11.9`. If the command is not
recognized, reinstall Python and make sure it was added to your PATH.

### An OpenAI API key

This project uses the OpenAI API, which is separate from simply having access
to the ChatGPT website.

1. Sign in at [platform.openai.com](https://platform.openai.com/).
2. Create an API key from the API keys area.
3. Make sure your account has an available payment method or API credits.
4. Copy the key somewhere temporarily. It normally begins with `sk-`.

Treat the key like a password. Do not post it on GitHub, put it in a
screenshot, or send it to another person.

## Step 1: Download the project

If you are comfortable using Git, open a terminal and run:

```bash
git clone https://github.com/EaHorton/Handwritten_Transcriptions.git
cd handwritten-transcriptions
```

If you do not have Git:

1. Open the repository page on GitHub.
2. Select the green **Code** button.
3. Select **Download ZIP**.
4. Extract the ZIP file.
5. Open a terminal in the extracted folder.

The folder must contain files such as:

```text
ai_vision.py
hallucination_detector.py
requirements.txt
README.md
```

## Step 2: Create a private document folder

Keep your source images and transcriptions separate from the code repository.
For example:

```text
my-documents/
├── images/
│   ├── page-001.jpg
│   ├── page-002.jpg
│   └── page-003.png
└── transcriptions/
```

You do not need to create the `transcriptions` folder in advance; the script will create it.

The script accepts these image types:

- `.jpg` and `.jpeg`
- `.png`
- `.tiff`
- `.bmp`
- `.webp`

Use clear, high-resolution images where possible. Make sure the writing is
upright, well lit, and not unnecessarily cropped. The script can resize very
large images before sending them to the API, but it cannot fix an image that
is too blurry or dark to read.

## Step 3: Create an isolated Python environment

An isolated environment keeps this project’s packages separate from other
Python programs on your computer.

From inside the project folder, run the command for your operating system.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

When the environment is active, you will usually see `(.venv)` at the
beginning of the terminal prompt.

## Step 4: Install the required packages

With the virtual environment active, run:

macOS/Linux:

```bash
python -m pip install -r requirements.txt
```

Windows:

```powershell
python -m pip install -r requirements.txt
```

Wait for the installation to finish before continuing.

## Step 5: Add your API key

Set the key in the current terminal session.

### macOS/Linux

```bash
export OPENAI_API_KEY="paste-your-api-key-here"
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="paste-your-api-key-here"
```

Do not include the quotation marks as part of the key. The key is temporary
for the current terminal session. If you close the terminal, set it again
the next time you use the script.

You can optionally choose a model:

macOS/Linux:

```bash
export OPENAI_MODEL="gpt-4o"
```

Windows PowerShell:

```powershell
$env:OPENAI_MODEL="gpt-4o"
```

The program uses `gpt-4o` if `OPENAI_MODEL` is not set. Model names and
pricing can change, so check OpenAI’s current documentation if a model is
unavailable.

## Step 6: Run your first transcription

From the project folder, run the following command.

### macOS/Linux

```bash
python ai_vision.py \
  --input-dir ~/my-documents/images \
  --output-dir ~/my-documents/transcriptions
```

### Windows PowerShell

```powershell
python ai_vision.py `
  --input-dir "$HOME\my-documents\images" `
  --output-dir "$HOME\my-documents\transcriptions"
```

Replace the example paths with the actual locations of your folders.

If a path contains spaces, surround it with quotation marks:

```bash
python ai_vision.py \
  --input-dir "/Users/Your Name/My Documents/images" \
  --output-dir "/Users/Your Name/My Documents/transcriptions"
```

The program prints its progress, including the image currently being
processed. Do not close the terminal until it reports that processing is
complete.

## Processing subfolders

By default, only images directly inside `--input-dir` are processed. Add
`--recursive` when images are stored in folders below the input directory:

```bash
python ai_vision.py \
  --input-dir ~/my-documents \
  --output-dir ~/my-transcriptions \
  --recursive
```

The output keeps the same relative folder structure. For example:

```text
my-documents/
├── volume-1/page-001.jpg
└── volume-2/page-001.jpg

my-transcriptions/
├── volume-1/page-001_transcription.txt
└── volume-2/page-001_transcription.txt
```

## Understanding the output

For an image named `page-001.jpg`, the script creates:

```text
page-001_transcription.txt
```

Each transcription includes:

- The source image name.
- The processing date.
- Quality-review information.
- The model’s transcribed text.

The output directory also contains:

```text
summaries/
├── usage_summary.json
└── usage_summary.txt
```

These files report the number of API requests, token usage, processed images,
and estimated cost. The estimate is only available for models whose prices
are listed in `ai_vision.py`.

If the program reports a quality concern, it preserves the transcription and
marks the file for review. It does not silently delete uncertain text.

## Useful options

### Process a nested folder tree

```bash
python ai_vision.py --input-dir PATH_TO_DOCUMENTS --output-dir PATH_TO_OUTPUT --recursive
```

### Process without a pause between images

The default one-second pause can help avoid rate-limit problems. For a small
test run, you can disable it:

```bash
python ai_vision.py --input-dir PATH_TO_IMAGES --output-dir PATH_TO_OUTPUT --delay 0
```

### Change the maximum image dimension

Very large images are resized before upload. The default maximum width or
height is 2048 pixels:

```bash
python ai_vision.py \
  --input-dir PATH_TO_IMAGES \
  --output-dir PATH_TO_OUTPUT \
  --max-image-size 3000
```

Larger images may preserve more detail but can increase processing cost.

### See all available options

```bash
python ai_vision.py --help
```

## Running the program again

To run another collection:

1. Open a new terminal or reactivate the virtual environment.
2. Set `OPENAI_API_KEY` again if you opened a new terminal.
3. Run the command with the new input and output folders.

To reactivate the environment:

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

To leave the environment when finished:

```bash
deactivate
```

## Troubleshooting

### “OPENAI_API_KEY environment variable not set”

Set the API key in the same terminal window where you run the script. If you
opened a new terminal, set it again.

### “input directory not found”

Check that:

1. The folder exists.
2. The path is spelled correctly.
3. You used the path to the folder containing the images.
4. Paths containing spaces are in quotation marks.

### “No image files found”

Confirm that the files use one of the supported extensions listed above. Add
`--recursive` if the images are inside subfolders.

### “ModuleNotFoundError”

Activate the virtual environment and reinstall the dependencies:

```bash
python -m pip install -r requirements.txt
```

### “Permission denied” on Windows PowerShell

If PowerShell refuses to activate the environment, open PowerShell as your
normal user and run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try the activation command again.

### The transcription is inaccurate

Try images with better resolution, stronger lighting, and less background
noise. Review the original image alongside the output. For difficult pages,
process a smaller group of images and consider increasing
`--max-image-size`.

## Keeping private information out of GitHub

Never commit:

- API keys.
- Original document images.
- Generated transcriptions.
- Private notes or metadata.

The repository includes a `.gitignore` file for common private and generated
directories. Before committing changes, run:

```bash
git status
```

If a key has accidentally been committed, revoke it immediately in the OpenAI
platform and create a replacement. Removing the file in a later commit does
not make an exposed key safe.
