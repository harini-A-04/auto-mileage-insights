# Deploying the MPG Predictor on Streamlit Community Cloud

## 1. Run locally first
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## 2. Push to GitHub
```bash
git init
git add app.py requirements.txt auto-mpg.csv
git commit -m "MPG predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/mpg-predictor.git
git push -u origin main
```
`auto-mpg.csv` is your dataset and must be in the repo, next to `app.py`.

## 3. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Pick the repo, branch `main`, main file path `app.py`.
4. (Advanced settings) choose Python 3.11 or 3.12.
5. Click **Deploy**. First boot takes a few minutes (installs requirements,
   downloads data, trains models). Later visits are fast because of caching.

## 4. Updating
Push to GitHub; the app redeploys automatically.

## Troubleshooting
- Build error: check package names/versions in requirements.txt.
- "Could not load the dataset": make sure `auto-mpg.csv` was committed
  in the same folder as `app.py`, or upload a CSV from the sidebar.
