# 🚀 Deployment Guide

This guide will help you deploy the Flight Delay Predictor to various cloud platforms.

## 📋 Pre-Deployment Checklist

✅ **Lightweight models included**: 
- `trained_model_lite.pkl` (~40MB)
- `airport_encoder_lite.pkl` (~50KB)
- **Total app size: ~45MB** ✨

✅ **Test locally**:
```bash
python app_new.py
```

✅ **Verify dependencies in `requirements.txt`**

## 🌐 Free Deployment Options (Perfect Size!)

Your app is now **~45MB** - perfect for all free hosting platforms! 🎉

### 1. Railway (Recommended - Free Tier Available)

1. **Create Railway Account**: Visit [railway.app](https://railway.app)
2. **Connect GitHub**: Link your GitHub repository
3. **Deploy**: 
   - Select your repository
   - Railway auto-detects Python and uses `Procfile`
   - Deployment starts automatically
4. **Custom Domain**: Available on paid plans

**Pros**: Easy setup, generous free tier, automatic HTTPS
**Cons**: Limited free hours per month

### 2. Render (Good Free Option)

1. **Create Account**: Visit [render.com](https://render.com)
2. **New Web Service**: Connect your GitHub repo
3. **Configuration**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app_new.py`
4. **Deploy**: Click "Create Web Service"

**Pros**: Always-on free tier, custom domains
**Cons**: Slower cold starts on free tier

### 3. Heroku (Classic Option)

1. **Install Heroku CLI**: [Download here](https://devcenter.heroku.com/articles/heroku-cli)
2. **Login**: `heroku login`
3. **Create App**: `heroku create your-app-name`
4. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy flight predictor"
   git push heroku main
   ```

**Pros**: Mature platform, lots of add-ons
**Cons**: No free tier (starting $5/month)

### 4. Vercel (Serverless)

1. **Install Vercel CLI**: `npm i -g vercel`
2. **Create `vercel.json`**:
   ```json
   {
     "builds": [{"src": "app_new.py", "use": "@vercel/python"}],
     "routes": [{"src": "/(.*)", "dest": "app_new.py"}]
   }
   ```
3. **Deploy**: `vercel --prod`

**Pros**: Fast deployment, good for static sites
**Cons**: Cold start delays, complexity with ML models

## 🔧 Environment Variables

For production, consider setting:
- `FLASK_ENV=production`
- `FLASK_DEBUG=False`

## 📊 Model Files

The pre-trained model files (`trained_model.pkl`, `airport_encoder.pkl`) should be included in your repository for deployment. They're small enough (~20MB total) to fit in Git.

If they're too large, consider:
1. Using Git LFS (Large File Storage)
2. Storing in cloud storage and downloading during startup
3. Training model on first startup (slower)

## 🔒 Security Notes

- The app uses external weather APIs (no API keys required)
- No sensitive data is stored
- Consider rate limiting in production
- Use HTTPS (most platforms provide this automatically)

## 🚨 Troubleshooting

**Common Issues:**

1. **Module Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **Model Loading Fails**: Check that `.pkl` files are in repository
3. **Port Issues**: Make sure app binds to `0.0.0.0` and uses `$PORT`
4. **Memory Errors**: ML models need ~512MB RAM minimum

**Performance Tips:**
- Models are loaded once at startup (good for serverless)
- Weather API calls are cached
- Consider Redis for production caching

## 📱 Final Steps

1. **Test thoroughly** on the deployed URL
2. **Update README** with live demo link
3. **Monitor performance** using platform dashboards
4. **Set up custom domain** (optional)

---

🎉 **Congratulations!** Your Flight Delay Predictor is now live and helping users make informed travel decisions!
