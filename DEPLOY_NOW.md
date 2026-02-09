# 🚀 Deploy Now: Railway + Vercel Setup Guide

**Status:** Ready to deploy ✅  
**Last Updated:** Feb 9, 2026  
**Time to Deploy:** ~15 minutes

This guide walks you through deploying this project **step-by-step** with Railway (backend) and Vercel (frontend).

---

## 📋 Prerequisites (5 min setup)

You need three things:
1. **GitHub Account** - Your repo is already pushed: https://github.com/psahani3486/Reinforcement-learning-Recommendation-Project
2. **MongoDB Atlas Account** - Free cloud MongoDB (2 min signup)
3. **Railway Account** - Sign up with GitHub (1 min)
4. **Vercel Account** - Sign up with GitHub (1 min)

---

## Part 1: MongoDB Atlas Setup (2 min)

MongoDB will store all your app data in the cloud.

### Step 1.1: Create MongoDB Atlas Account
1. Go to https://mongodb.com/cloud/atlas
2. Click **Sign Up**
3. Fill in your details
4. Verify email

### Step 1.2: Create a Free Cluster
1. After login, click **Create Deployment** → **Build a Cluster**
2. Choose **Shared** (Free tier)
3. Select region closest to you
4. Click **Create**
5. Wait ~3 minutes for cluster to start

### Step 1.3: Create Database User
1. In **Security** → **Quickstart**
2. Click **Create a Database User**
3. Save username and password (e.g., `admin` / `your-strong-password`)
4. Click **Create User**

### Step 1.4: Whitelist IP
1. Go to **Security** → **Network Access**
2. Click **Add IP Address**
3. Enter `0.0.0.0/0` (allows all IPs - fine for development)
4. Click **Confirm**

### Step 1.5: Get Connection String
1. Go to **Deployment** → **Databases**
2. Click **Connect** on your cluster
3. Select **Drivers** → **Node.js**
4. Copy the connection string (looks like):
   ```
   mongodb+srv://admin:PASSWORD@cluster0.mongodb.net/?retryWrites=true&w=majority
   ```
5. Replace `<password>` with your actual password
6. **Final URL** (example):
   ```
   mongodb+srv://admin:MyStrong123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
   ```
7. Save this! ⭐ You'll use it in Railway

---

## Part 2: Deploy Backend on Railway (3 min)

### Step 2.1: Sign Up to Railway with GitHub
1. Go to https://railway.app
2. Click **Login** → **GitHub**
3. Authorize Railway to access your GitHub
4. Done! You're logged in

### Step 2.2: Create New Project
1. Click **Create New Project**
2. Select **Deploy from GitHub repo**
3. Click **Configure GitHub App** (if needed)
4. Select repository: `Reinforcement-learning-Recommendation-Project`
5. Click **Deploy**

Railway will automatically detect `backend/package.json` and start deploying.

### Step 2.3: Add Environment Variables
1. Once the deployment starts, click on the **backend** service (on the right panel)
2. Go to **Variables** tab
3. Add these variables (copy-paste, one by one):

   | Key | Value |
   |-----|-------|
   | `MONGODB_URI` | `mongodb+srv://admin:MyStrong123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority` |
   | `JWT_SECRET` | `your-super-secret-key-min-32-chars-xyz123abc` |
   | `NODE_ENV` | `production` |
   | `CORS_ORIGIN` | `https://rl-recommendation.vercel.app` |

   ➡️ **Replace values with yours!**

4. After adding each variable, Railway auto-saves
5. The deployment should restart automatically
6. Wait for **green checkmark** ✅ in deployment status

### Step 2.4: Get Your Backend URL
1. Click on your backend service
2. Look for **Public URL** (top right)
3. Copy it (looks like): `https://railway-production-xxxx.up.railway.app`
4. Save this! ⭐ You'll use it in Vercel

### ✅ Backend is now LIVE!
Test it: Open your railway URL in browser → should see `{"message": "RL Recommendation System API", ...}`

---

## Part 3: Deploy Frontend on Vercel (2 min)

### Step 3.1: Sign Up to Vercel with GitHub
1. Go to https://vercel.com
2. Click **Sign Up** → **GitHub**
3. Authorize Vercel to access your GitHub
4. Done! You're logged in

### Step 3.2: Import Your Project
1. Click **Add New** → **Project**
2. Click **Import Git Repository**
3. Paste your repo URL: `https://github.com/psahani3486/Reinforcement-learning-Recommendation-Project`
4. Click **Continue**

### Step 3.3: Configure Project Settings
Vercel will auto-detect settings. Verify these:

- **Framework Preset:** `Vite`
- **Root Directory:** `frontend` (select from dropdown if needed)
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Install Command:** `npm install`

Everything should be auto-filled. Click **Deploy** to proceed.

### Step 3.4: Add Environment Variable
1. While deploying (or after), go to **Settings** → **Environment Variables**
2. Add:
   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | `https://railway-production-xxxx.up.railway.app` |

   ⭐ Use the Railway backend URL from **Part 2, Step 2.4**

3. Click **Save**

### Step 3.5: Redeploy with Environment Variable
1. Go to **Deployments** tab
2. Find the latest deployment (top one)
3. Click **...** (three dots) → **Redeploy**
4. Click **Redeploy** again to confirm
5. Wait for green **Ready** status ✅

### Step 3.6: Get Your Frontend URL
1. Click on the deployment
2. Your frontend URL is shown (top, something like): `https://rl-recommendation.vercel.app`
3. Save it! ⭐

### ✅ Frontend is now LIVE!

---

## Part 4: Update CORS on Backend (1 min)

Now that we have the frontend URL, update the backend CORS setting:

### Step 4.1: Go Back to Railway
1. Open https://railway.app
2. Click on your backend service
3. Go to **Variables** tab
4. Find `CORS_ORIGIN`
5. Change the value to your **Vercel frontend URL** (from Part 3, Step 3.6)
   - Example: `https://rl-recommendation.vercel.app`
6. Save (auto-saves)
7. Railway auto-redeploys ✅

---

## Part 5: Test Your Live Deployment (2 min)

### Test 1: Open Frontend
1. Go to your **Vercel URL** (from Part 3, Step 3.6)
2. You should see the app load!
3. Try to login/signup

### Test 2: Check Network Call Works
1. Open **DevTools** (F12)
2. Go to **Console** tab
3. Try to login
4. In **Network** tab, check API call to backend
5. Should show **Status 200** (success)
6. No **CORS errors** in Console 🎉

### ⚠️ If You See CORS Error
1. Go back to Railway
2. Variables tab
3. Change `CORS_ORIGIN` to `*` (allow all)
4. Save and redeploy
5. Test again

---

## 🎉 You're Done!

### URLs to Share
- **Frontend:** https://rl-recommendation.vercel.app (or your URL)
- **Backend API:** https://railway-production-xxxx.up.railway.app (private)

### Next Steps (Optional)
1. **Custom Domain** - Add your own domain to Vercel
2. **Monitoring** - Check logs in Railway dashboard
3. **Auto-Deploy** - Both services redeploy on git push 🔄

---

## 🚨 Troubleshooting

### "Cannot connect to API"
- Check `VITE_API_URL` in Vercel is correct
- Check `CORS_ORIGIN` in Railway is correct (should match Vercel URL)
- Redeploy both services

### "MongoDB Connection Failed"
- Check `MONGODB_URI` in Railway variables
- Verify database user password in MongoDB Atlas
- Check IP whitelist in MongoDB Atlas (should include `0.0.0.0/0`)

### "Deployment Failed"
- Check Railway/Vercel logs
- Ensure `backend/package.json` and `frontend/package.json` exist
- Try redeploying manually

### "Port Already in Use" (local testing)
```bash
# Kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

---

## 📚 Reference

### Commands (if needed)
```bash
# Test backend locally before deploy
cd backend
npm install
npm start

# Test frontend locally before deploy
cd frontend
npm install
npm run dev
```

### Environment Variables Summary
**Railway (Backend)**
```
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority
JWT_SECRET=min-32-char-random-string
NODE_ENV=production
CORS_ORIGIN=https://your-vercel-url.vercel.app
```

**Vercel (Frontend)**
```
VITE_API_URL=https://your-railway-url.up.railway.app
```

---

## ✅ Deployment Checklist

- [ ] MongoDB Atlas cluster created
- [ ] MongoDB user created with password
- [ ] IP whitelist set to `0.0.0.0/0`
- [ ] MongoDB connection string copied
- [ ] Railway backend deployed
- [ ] Backend environment variables added
- [ ] Vercel frontend deployed  
- [ ] Frontend `VITE_API_URL` variable added
- [ ] Frontend redeployed with env var
- [ ] Backend `CORS_ORIGIN` updated with Vercel URL
- [ ] Backend redeployed
- [ ] Frontend URL loads in browser
- [ ] API calls work (check Network tab)
- [ ] No CORS errors

---

## 💬 Questions?
1. Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for more details
2. Railway docs: https://docs.railway.app
3. Vercel docs: https://vercel.com/docs
4. MongoDB docs: https://docs.atlas.mongodb.com

**Happy deploying! 🚀**
