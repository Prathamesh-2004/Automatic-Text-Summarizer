"""
utils/file_handler.py
---------------------
MODULE 1 – USER INPUT (file upload part)
Reads an uploaded .txt file and returns its text.
"""


def read_uploaded_file(uploaded_file) -> str:
    """Read a Streamlit UploadedFile (.txt) as text.
    Tries UTF-8 first; falls back to Latin-1 so odd characters never crash the app."""
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")
