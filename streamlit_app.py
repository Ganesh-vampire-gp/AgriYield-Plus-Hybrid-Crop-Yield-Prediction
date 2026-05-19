from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Importing the module executes the existing Streamlit app logic.
import agriyield.app.app  # noqa: F401
