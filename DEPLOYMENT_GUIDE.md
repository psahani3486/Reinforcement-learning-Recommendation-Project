# Deployment Guide: Vercel + Render/Railway

This guide explains how to deploy the RL Recommendation System with the frontend on **Vercel** and the backend on **Render** or **Railway**.

---

## Part 1: Backend Deployment (Render or Railway)

### Prerequisites
- MongoDB Atlas account (free tier available: https://mongodb.com/cloud/atlas)
- Git repository pushed to GitHub
- Render or Railway account

### Option A: Deploy on Render

#### Step 1: Set up MongoDB Atlas
1. Go to https://mongodb.com/cloud/atlas
2. Create a free cluster
3. In Database Access, create a user (username/password)
4. In Network Access, allow all IPs (0.0.0.0/0) or your Render IP
5. Click Connect and copy the connection string
   - Replace `<username>` and `<password>` with your credentials
   - Example: `mongodb+srv://user:pass@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority`

#### Step 2: Deploy Backend to Render
1. Go to https://render.com and sign in with GitHub
2. Click **New** → **Web Service**
3. Select your GitHub repository
4. Fill in the form:
   - **Name:** `rl-backend` (or any name)
   - **Root Directory:** `backend`
   - **Environment:** `Node`
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
5. Click **Create Web Service**
6. Once created, go to **Environment** tab and add these variables:
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority
   JWT_SECRET=generate-a-strong-random-secret-key-here
   NODE_ENV=production
   CORS_ORIGIN=https://your-frontend-domain.vercel.app
   ```
7. Render will auto-deploy when you push to `main` branch
8. Your backend URL will be shown in the dashboard (e.g., `https://rl-backend.onrender.com`)

---

### Option B: Deploy on Railway

#### Step 1: Set up MongoDB Atlas
(Same as Render - follow Step 1 above)

#### Step 2: Deploy Backend to Railway
1. Go to https://railway.app and sign in with GitHub
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your repository and authorize
4. Choose to deploy from GitHub
5. Railway auto-detects `backend/package.json`; if not, select **Node** environment
6. Add environment variables in Project Settings → Variables:
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority
   JWT_SECRET=generate-a-strong-random-secret-key-here
   NODE_ENV=production
   CORS_ORIGIN=https://your-frontend-domain.vercel.app
   PORT=5000
   ```
7. Deploy and your backend URL will be provided (e.g., `https://rl-backend.up.railway.app`)

---

## Part 2: Frontend Deployment (Vercel)

### Prerequisites
- Vercel account (https://vercel.com)
- Backend URL from Render/Railway (from Part 1)
- GitHub repository

### Step 1: Update Vercel Configuration
The `vercel.json` file is already configured. It specifies:
- Root directory: `frontend`
- Build output: `dist`
- SPA routing: all routes point to `index.html`

### Step 2: Deploy to Vercel
1. Go to https://vercel.com and sign in with GitHub
2. Click **Add New** → **Project**
3. Select your GitHub repository
4. Configure project settings:
   - **Framework Preset:** `Vite`
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`
5. Click **Deploy**
6. Once deployed, go to **Settings** → **Environment Variables** and add:
   ```
   VITE_API_URL=https://rl-backend.onrender.com
   ```
   (Replace with your Render/Railway backend URL)
7. Redeploy to apply the environment variable:
   - Go to **Deployments**
   - Click the three dots on the latest deployment
   - Select **Redeploy**

Your frontend will be live at the Vercel URL (e.g., `https://rl-recommendation.vercel.app`)

---

## Part 3: Verification & Testing

### Test Backend Health
```bash
curl https://rl-backend.onrender.com
# Should return:
# {
#   "message": "RL Recommendation System API",
#   "version": "2.0.0",
#   ...
# }
```

### Test Frontend
1. Visit your Vercel URL in the browser
2. Open DevTools → Network tab
3. Check that API calls go to your backend URL
4. Verify no CORS errors in Console

### Troubleshooting

#### CORS Errors
- Backend CORS is set to allow all origins by default
- If issues, update `backend/src/server.js`:
  ```javascript
  app.use(cors({
    origin: process.env.CORS_ORIGIN || '*'
  }));
  ```

#### Environment Variables Not Loading
- Redeploy after adding variables (both Vercel and Render/Railway)
- Check variable names match exactly (case-sensitive)
- For Vercel: restart the deployment for `VITE_*` variables to be embedded

#### MongoDB Connection Failures
- Verify IP is whitelisted in MongoDB Atlas Network Access
- Check username/password in connection string
- Ensure database/collection names match

---

## Part 4: Continuous Deployment

Both Vercel and Render/Railway support automatic deployments:
- **Frontend (Vercel):** Auto-deploys on push to `main` branch
- **Backend (Render/Railway):** 
  - Render: Auto-deploys by default
  - Railway: Configure auto-deploy in Project Settings

To disable auto-deploy, adjust settings in the respective dashboards.

---

## Part 5: Environment Variables Summary

### Backend (.env on Render/Railway)
| Variable | Example | Notes |
|----------|---------|-------|
| `MONGODB_URI` | `mongodb+srv://user:pass@cluster.mongodb.net/db?retryWrites=true&w=majority` | MongoDB Atlas connection string |
| `JWT_SECRET` | `super-secret-key-min-32-chars` | Use a strong random string |
| `NODE_ENV` | `production` | Set to production on Render/Railway |
| `CORS_ORIGIN` | `https://rl-recommendation.vercel.app` | Frontend URL |
| `PORT` | `5000` | Leave default; Render/Railway overrides |

### Frontend (.env on Vercel)
| Variable | Example | Notes |
|----------|---------|-------|
| `VITE_API_URL` | `https://rl-backend.onrender.com` | Backend API base URL |

---

## Part 6: Custom Domain (Optional)

### Vercel
1. Go to Project Settings → Domains
2. Add your custom domain
3. Update DNS records (instructions in Vercel dashboard)

### Render/Railway
1. Similar process in respective dashboards
2. Follow platform-specific DNS setup guides

---

## Additional Resources

- **Vercel Docs:** https://vercel.com/docs
- **Render Docs:** https://render.com/docs
- **Railway Docs:** https://docs.railway.app
- **MongoDB Atlas:** https://docs.atlas.mongodb.com

---

## Rollback & Maintenance

### Render: Rollback Deployment
- Dashboard → Deployment History → Redeploy from previous version

### Railway: Rollback Deployment
- Project → Deployments → Select previous version → Deploy

### Vercel: Rollback Deployment
- Deployments tab → Click three dots on previous deployment → Redeploy

---

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Deploy backend (Render/Railway)
3. ✅ Deploy frontend (Vercel)
4. ✅ Test API connectivity
5. ✅ Set up custom domain (optional)
6. ✅ Configure monitoring/logging (optional)

**Happy deploying!** 🚀
