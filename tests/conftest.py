# rlgymbotv2/tests/conftest.py
import os, sys
# Add project root to sys.path so `import mysim` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
