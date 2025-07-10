# Netlify Deployment Guide

## Quick Deploy to Netlify

### Option 1: Auto-deploy from GitHub (Recommended)

1. **Connect to Netlify:**
   - Go to [Netlify](https://netlify.com)
   - Sign up/Login with your GitHub account
   - Click "New site from Git"
   - Choose GitHub and authorize Netlify

2. **Select Repository:**
   - Find and select `CassiusFam/predictor_final`
   - Branch: `main`

3. **Build Settings:**
   - Build command: `cp -r dist/* .` (copies static files to root)
   - Publish directory: `.` (root directory)
   - Functions directory: `netlify/functions`

4. **Environment Variables:**
   - Add `OPENWEATHER_API_KEY` with your OpenWeatherMap API key
   - Get API key from: https://openweathermap.org/api

5. **Deploy:**
   - Click "Deploy site"
   - Your site will be available at a random URL like `https://amazing-site-name.netlify.app`

### Option 2: Manual Deploy

1. **Prepare Files:**
   ```bash
   # In your project directory
   cp -r dist/* .
   zip -r site.zip . -x "*.git*" "*.vscode*" "__pycache__*" "templates/*" "data/*" "tests/*"
   ```

2. **Upload to Netlify:**
   - Go to [Netlify](https://netlify.com)
   - Drag and drop the `site.zip` file
   - Configure environment variables in site settings

## Important Files for Netlify

- `netlify.toml` - Netlify configuration
- `dist/index.html` - Main application page
- `dist/about.html` - About page
- `dist/airports.js` - Airport data
- `netlify/functions/predict.py` - Serverless prediction function
- Model files: `trained_model_lite.pkl`, `airport_encoder_lite.pkl`

## Environment Variables Needed

- `OPENWEATHER_API_KEY` - Your OpenWeatherMap API key (optional, will use default weather if not provided)

## Troubleshooting

1. **Function errors:** Check Netlify function logs in the dashboard
2. **Missing dependencies:** Ensure all required packages are in `netlify/functions/requirements.txt`
3. **Model not found:** Make sure the pickle files are in the repository root
4. **CORS issues:** Functions include CORS headers, but check browser console for errors

## Testing Locally

To test the Netlify functions locally:

```bash
npm install -g netlify-cli
netlify dev
```

This will start a local server that mimics Netlify's environment.

## Custom Domain

After deployment:
1. Go to your site settings in Netlify
2. Click "Domain management"
3. Add your custom domain
4. Follow the DNS configuration instructions

Your app is now ready for deployment! 🚀
