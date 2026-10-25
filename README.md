# Test Case Generator

This project generates a Product Requirement Document (PRD), extracts requirement details from that PRD, and converts the resulting requirements into user stories and test cases.

## Project Structure

- `prdGenerator.py` – creates a formal PRD document in PDF format from a user-provided product name.
- `requirementGatheringAgent.py` – reads the generated PRD PDF, extracts the requirement text, and converts it into structured user stories.
- `testCaseGenerator.py` – transforms the generated user stories into test cases in an Excel-style workbook format.
- `run_pipeline.py` – runs the full workflow in sequence: PRD -> requirements -> test cases.
- `generated/` – output folder for generated PDFs and Excel files.

## Prerequisites

- Python 3.10+
- A valid OpenAI API key in a local `.env` file

Example `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your `.env` file is present in the project root and contains your OpenAI key.

## How to Run

### Generate the PRD

```bash
python prdGenerator.py
```

When prompted, enter the product name for which you want to generate the PRD.

### Run the requirement extraction flow

```bash
python requirementGatheringAgent.py
```

This reads the generated PRD PDF from the `generated/` folder and creates user stories.

### Generate test cases

```bash
python testCaseGenerator.py
```

This reads the generated user story data and creates an Excel workbook with test cases.

### Run the full pipeline

```bash
python run_pipeline.py
```

This runs the workflow in this order:

1. PRD generation
2. Requirement extraction
3. Test case generation

## Output Files

Generated files are stored in the `generated/` folder, including:

- `prd_document.pdf`
- `user_stories_document.pdf`
- `user_stories.json`
- `test_cases_document.xlsx`

## Notes

- WeasyPrint may show native library warnings on macOS if system dependencies are missing. The project includes a fallback path that still generates PDF output when possible.
- The generated test case document is saved as an Excel workbook with columns such as Serial Number, Test Case Summary, Test Case Steps, and Expected Result.
