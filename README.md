# FTIR ML Streamlit App (Railway-ready)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app_k.py
```

## Deploy on Railway
1. Push this repo to GitHub
2. In Railway: New Project → Deploy from GitHub Repo
3. Railway will use:
   - `requirements.txt` for deps
   - `Procfile` for the start command

The app will start with:
`streamlit run app_k.py --server.port $PORT --server.address 0.0.0.0`
