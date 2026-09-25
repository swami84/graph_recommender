# Google Docs-ready article files

Use the `.docx` files in this directory when working in Google Docs. Each figure is embedded in the
document package; no local Markdown paths need to be resolved.

1. Upload a `.docx` file to Google Drive.
2. Right-click it and select **Open with > Google Docs**.
3. Optionally choose **File > Save as Google Docs** to create a native Google document.

The matching PDF files are fixed-layout previews for checking the expected appearance. The
Markdown source files remain in `../drafts/` for version-controlled editing.

Regenerate the DOCX files from the Markdown sources with:

```bash
cd /home/swami/Work/Projects/foodie_revamp
/home/swami/venv/dev_env/bin/python article/build_google_docs_ready.py
```
