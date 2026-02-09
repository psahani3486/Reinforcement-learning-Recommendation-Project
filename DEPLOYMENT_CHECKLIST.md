# Quick Deployment Checklist

## Pre-Deployment Setup (Once)

### 1. Prepare Your GitHub Repository
```bash
git add .
git commit -m "Add deployment configs: vercel.json, .env.example, DEPLOYMENT_GUIDE.md"
git push origin main
```

### 2. MongoDB Atlas Setup
- [ ] Create MongoDB Atlas account (https://mongodb.com/cloud/atlas)
- [ ] Create a free cluster
- [ ] Create database user (save username/password)
- [ ] Whitelist IPs (allow 0.0.0.0/0 for simplicity)
- [ ] Get connection string: `mongodb+srv://user:pass@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority`

### 3. Backend Deployment (Choose ONE)

#### Option A: Render
- [ ] Create Render account (https://render.com)
- [ ] Connect GitHub
- [ ] New Web Service
  - Repository: Your repo
  - Root Directory: `backend`
  - Build Command: `npm install`
  - Start Command: `npm start`
- [ ] Add Environment Variables:
  - `MONGODB_URI` = MongoDB connection string
  - `JWT_SECRET` = (generate random string, min 32 chars: `openssl rand -base64 32`)
  - `NODE_ENV` = `production`
  - `CORS_ORIGIN` = `https://YOUR-FRONTEND.vercel.app` (update after frontend deploy)
- [ ] Deploy (auto on git push)
- [ ] Copy backend URL (e.g., `https://rl-backend.onrender.com`)

#### Option B: Railway
- [ ] Create Railway account (https://railway.app)
- [ ] Connect GitHub
- [ ] New Project → Deploy from repository
- [ ] Select your repo
- [ ] Add Environment Variables (same as Render above)
- [ ] Deploy
- [ ] Copy backend URL

### 4. Frontend Deployment (Vercel)
- [ ] Create Vercel account (https://vercel.com)
- [ ] Import GitHub project
- [ ] Project Settings:
  - Framework Preset: `Vite`
  - Root Directory: `frontend`
  - Build Command: `npm run build`
  - Output Directory: `dist`
- [ ] Add Environment Variable:
  - `VITE_API_URL` = `https://rl-backend.onrender.com` (YOUR backend URL from step 3)
- [ ] Deploy
- [ ] Copy frontend URL (e.g., `https://rl-recommendation.vercel.app`)
- [ ] Update backend `CORS_ORIGIN` to this URL on Render/Railway and redeploy

---

## Post-Deployment Testing

### 1. Backend Health Check
```bash
curl https://rl-backend.onrender.com
# Expected: JSON response with API endpoints
```

### 2. Frontend Load
- [ ] Open frontend URL in browser
- [ ] Check DevTools > Console (no CORS errors)
- [ ] Try login/signup to verify API calls work
- [ ] Check Network tab to confirm requests go to correct backend URL

### 3. Fix Any Issues
- CORS Error? → Update `CORS_ORIGIN` on backend
- API Not Found? → Check `VITE_API_URL` on frontend
- Can't Connect to DB? → Verify MongoDB Atlas IP whitelist + credentials

---

## Production Environment Variables

### Backend (Render/Railway)
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority
JWT_SECRET=your-strong-random-secret-key-min-32-chars
NODE_ENV=production
CORS_ORIGIN=https://your-frontend.vercel.app
```

### Frontend (Vercel)
```
VITE_API_URL=https://your-backend.onrender.com
```

---

## Useful Commands

### Generate Strong Secret
```bash
# On Mac/Linux:
openssl rand -base64 32

# On Windows PowerShell:
[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Random -Count 32 -InputObject (33..126) | ForEach-Object {[char]$_}) -join ''))
```

### Test API Locally Before Deploy
```bash
# Terminal 1: Backend
cd backend
npm install
npm run dev

# Terminal 2: Frontend
cd frontend
npm install
VITE_API_URL=http://localhost:5000 npm run dev
```

---

## Monitoring & Logs

- **Vercel:** Dashboard → Deployments → Click deployment → Logs
- **Render:** Dashboard → Logs (bottom right)
- **Railway:** Dashboard → Deployments → Logs

---

## Auto-Redeploy on Git Push
- Vercel: Automatic (no config needed)
- Render: Automatic (no config needed)
- Railway: Configure in Project Settings → Auto-Deploy

---

## Rollback Instructions

If something breaks after deployment:

**Vercel:**
1. Go to Deployments tab
2. Find previous working deployment
3. Click three dots → Redeploy

**Render/Railway:**
1. Go to Deployment History
2. Click previous successful build
3. Redeploy

---

## Need Help?
- **Vercel Docs:** https://vercel.com/docs
- **Render Docs:** https://render.com/docs
- **Railway Docs:** https://docs.railway.app
- **MongoDB:** https://docs.atlas.mongodb.com

Happy deploying! 🚀
