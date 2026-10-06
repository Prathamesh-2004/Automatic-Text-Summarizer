"""
utils/nltk_setup.py
-------------------
NLTK needs some extra data files (sentence tokenizer, stopword list,
WordNet dictionary for lemmatization). They are downloaded ONCE and
saved on your computer.

Run once manually:      python -m utils.nltk_setup
It is also called automatically by the preprocessing module, so if
something is missing it gets downloaded the first time you run the project.
"""

import nltk

# (path NLTK looks for, package name to download)
REQUIRED_RESOURCES = [
    ("tokenizers/punkt", "punkt"),          # sentence tokenizer (older NLTK)
    ("tokenizers/punkt_tab", "punkt_tab"),  # sentence tokenizer (NLTK 3.8.2+)
    ("corpora/stopwords", "stopwords"),     # list of stopwords
    ("corpora/wordnet", "wordnet"),         # dictionary for lemmatization
    ("corpora/omw-1.4", "omw-1.4"),         # extra WordNet data
]


def _is_installed(path: str) -> bool:
    """Some NLTK packages (like wordnet) are stored as .zip files,
    so we look for both 'corpora/wordnet' and 'corpora/wordnet.zip'."""
    for candidate in (path, path + ".zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_data(verbose: bool = False) -> None:
    """Download any NLTK resource that is not already installed."""
    for path, package in REQUIRED_RESOURCES:
        if _is_installed(path):
            if verbose:
                print(f"  [OK]   {package} already installed")
        else:
            print(f"  [....] downloading {package} ...")
            nltk.download(package, quiet=True)


if __name__ == "__main__":
    print("Checking NLTK data...")
    ensure_nltk_data(verbose=True)
    print("Done. NLTK is ready.")
