import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uvicorn

if __name__ == "__main__":
    uvicorn.run("caisen.api:app", host="127.0.0.1", port=8301, reload=False)
