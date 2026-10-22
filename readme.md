# AI Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and gives concrete suggestions to improve it, powered by Google Gemini Flash.

## Features

- Upload a resume as PDF, DOCX or TXT
- Overall ATS score (0-100) plus a breakdown: keywords, formatting, content, structure, readability
- Strengths, weaknesses and missing keywords
- Prioritised improvements with example rewrites
- Optional job description for a tailored match
- Download the report as JSON

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey and either paste it in the sidebar, or save it in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# optional: pin a model instead of the default "gemini-flash-latest"
# GEMINI_MODEL = "gemini-2.5-flash"
```

Never commit `secrets.toml` (see `.gitignore` below).

## Project files

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `.gitignore` | Keeps secrets and caches out of GitHub |

Suggested `.gitignore`:

```
venv/
__pycache__/
.streamlit/secrets.toml
.env
```

## Push to GitHub (using the website)

1. Sign in at https://github.com and click **+** (top right) -> **New repository**.
2. Name it (e.g. `resume-ats-checker`), choose Public or Private, and click **Create repository**.
3. On the new repo page click **uploading an existing file**.
4. Drag in `app.py`, `requirements.txt`, `README.md` and `.gitignore`. Do **not** upload any file containing your API key.
5. Write a commit message such as "Initial commit" and click **Commit changes**.

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** -> **Deploy a public app from GitHub** (or choose your existing repo).
3. Select your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** -> **Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. After a minute or two you get a public `*.streamlit.app` URL.

To update the app later, edit or upload the changed file on GitHub; Streamlit redeploys automatically.

## Notes

- Scanned/image-only resumes have no extractable text; export a text-based PDF or DOCX.
- The ATS score is an AI estimate, not the output of a real ATS. Use it as guidance.
- Resumes are sent to the Gemini API for analysis; the app does not store them.
- If the default model name stops working, set `GEMINI_MODEL` to a current Flash model.
