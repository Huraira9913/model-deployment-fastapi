# Model Deployment with FastAPI

Example of packaging a trained scikit-learn model and serving predictions via FastAPI.

## Structure

```
app/
  main.py
models/
  (generated on first run)
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then visit http://127.0.0.1:8000/docs

Author: Muhammad AbuHuraira
