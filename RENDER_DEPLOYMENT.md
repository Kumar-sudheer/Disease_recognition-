# Deploying to Render.com

## Overview
This Flask application provides unified crop disease detection using deep learning models. Follow these steps to deploy it on Render.

---

## Prerequisites
- GitHub account with your repository pushed
- Render.com account (free tier available)
- ML model files in the correct locations

---

## Step-by-Step Deployment Guide

### Step 1: Prepare Your Repository
1. Make sure all changes are committed to git:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. Verify your repository structure:
   ```
   ├── render.yaml
   ├── Procfile
   ├── .gitignore
   ├── requirements.txt
   └── sudheer/
       ├── app.py
       ├── requirements.txt
       ├── templates/
       ├── static/
       ├── rice_severity_model.pth
       ├── wheat_disease_resnet18.pth
       └── ...
   ```

### Step 2: Upload Model Files
**IMPORTANT:** Large ML model files should be downloaded at runtime, not stored in git.

**Option A: Upload models manually (easier for first deployment)**
1. After deployment, access your Render service's bash shell
2. Upload the model files to the container

**Option B: Host models in cloud storage (recommended)**
1. Upload your `.pth` files to AWS S3, or another cloud storage
2. Modify `app.py` to download them at startup

### Step 3: Create Render Service
1. Go to [Render.com](https://render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name:** crop-disease-detection
   - **Environment:** Python 3
   - **Build Command:** `pip install -r sudheer/requirements.txt`
   - **Start Command:** `cd sudheer && gunicorn --workers 1 --worker-class sync --bind 0.0.0.0:$PORT --timeout 120 app:app`
   - **Plan:** Free (or upgrade as needed)
   - **Region:** Choose closest to you
   - **Python Version:** 3.11

5. Click **"Create Web Service"**

### Step 4: Configure Environment Variables (if needed)
1. In Render dashboard, go to your service
2. Navigate to **Settings** → **Environment**
3. Add any required environment variables

### Step 5: Wait for Deployment
- Render will automatically build and deploy your app
- Check the **Logs** tab for any errors
- Once the status shows "Live", your app is running!

### Step 6: Access Your Application
- Your app URL will be displayed as: `https://crop-disease-detection.onrender.com`
- Share this URL with users

---

## Important Notes

### Model Files
- **Small models** (< 100MB): Can be stored in git/Render
- **Large models** (> 100MB): 
  - Store on cloud (AWS S3, Google Drive)
  - Download at app startup
  - Set cache/timeout appropriately

### Free Tier Limitations
- Deployments take longer (5-10 minutes)
- Spins down after 15 minutes of inactivity
- Limited to 1 worker process
- 50GB/month bandwidth limit
- Sufficient for testing/small traffic

### Performance Tuning
For production:
1. Upgrade to paid plan
2. Increase workers: `--workers 4`
3. Use PostgreSQL for data persistence
4. Enable CDN for static files

---

## Troubleshooting

### Build Fails
- Check Python version compatibility
- Verify `requirements.txt` in `sudheer/` folder exists
- Check Render logs for specific error

### App Starts but Crashes
Look for:
- Missing model files: Download from cloud storage at startup
- Port binding issues: Ensure using `$PORT` environment variable
- Memory issues: Free tier has limited RAM

### Models Not Loading
Add download logic at app startup:
```python
def download_model_if_missing():
    import urllib.request
    path = Path("rice_severity_model.pth")
    if not path.exists():
        url = "https://your-cloud-storage.com/rice_severity_model.pth"
        urllib.request.urlretrieve(url, path)
```

---

## Best Practices
1. ✅ Keep `.gitignore` updated for large files
2. ✅ Use environment variables for sensitive data
3. ✅ Test locally before pushing
4. ✅ Monitor Render logs regularly
5. ✅ Set up automated redeploy on git push
6. ✅ Use caching for inference when possible

---

## Next Steps
- Add custom domain
- Set up monitering alerts
- Implement error logging
- Add API rate limiting
- Cache predictions for common inputs

