from .controller import L20Controller
import sys
import os

# Ensure the root of the repo is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

def main():
    controller = L20Controller()
    controller.start()

if __name__ == "__main__":
    main()
