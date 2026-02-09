# ⚡ DEPLOY NOW - 5 Minute Quick Deploy

Follow these EXACT steps. Do NOT skip any. Copy-paste values when shown.

---

## 🚀 PART 1: MongoDB Atlas Setup (Get Connection String)

### Step 1: Create Free MongoDB Atlas Account
1. Go to: https://mongodb.com/cloud/atlas
2. Click **Sign Up**
3. Fill in: Email, Password, Accept terms
4. Click **Create account**
5. Verify email (check inbox)

### Step 2: Create Free Cluster
1. After login, click **Create Deployment** → **Build a Cluster**
2. Choose: **Shared (Free tier)**
3. Click **Create**
4. Wait 3 minutes for it to start (refresh if needed)

### Step 3: Create Database User
1. Left menu → **Database Access**
2. Click **Add New Database User**
3. Fill in:
   - **Username:** `admin`
   - **Password:** `MyMongo123` (⚠️ NO special characters - keep it simple!)
4. Click **Create User**

**⚠️ IMPORTANT:** Use ONLY letters, numbers, and hyphens. NO `!@#$%^&*` etc.

### Step 4: Whitelist IP
1. Left menu → **Network Access**
2. Click **Add IP Address**
3. Enter: `0.0.0.0/0` (allows all)
4. Click **Confirm**

### Step 5: Get Connection String
1. Go to **Databases** (left menu)
2. Click **Connect** on your cluster
3. Select **Drivers** → **Node.js**
4. Copy the connection string (looks like below)
5. **Replace** `<password>` with `MyMongo123`
6. **Replace** `<username>` with `admin`
7. Make sure it ends with: `/rl-recommendation?retryWrites=true&w=majority`

**Final URL should look like:**
```
mongodb+srv://admin:MyMongo123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

**Save this! ⭐ You need it for Railway.**

---

## 🚀 PART 2: Deploy Backend on Railway (3 min)

### Step 1: Go to Railway Dashboard
Open: https://railway.app/dashboard

You should already be logged in (from `railway login`).

### Step 2: Create New Project
1. Click **Create New Project**
2. Select **Deploy from GitHub repo**
3. A dropdown appears → Select `Reinforcement-learning-Recommendation-Project`
4. Click **Deploy**

Railway starts building automatically! ⏳ Wait for it...

### Step 3: Add Environment Variables (WHILE IT'S BUILDING)
1. Look at the right panel → you should see your service
2. Click on **backend** service (if you see it)
3. Go to **Variables** tab
4. Click **Add Variable**
5. Add these one by one (copy-paste exactly):

| Key | Value |
|-----|-------|
| `MONGODB_URI` | `mongodb+srv://admin:MyMongo123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority` |
| `JWT_SECRET` | `your-super-secret-jwt-key-12345678901234567890` |
| `NODE_ENV` | `production` |
| `CORS_ORIGIN` | `https://rl-recommendation.vercel.app` |

**⚠️ IMPORTANT:** 
- Replace `MyMongo123` with YOUR actual MongoDB password
- Replace `cluster0` with YOUR cluster name
- Press Enter after each one - Railway auto-saves

### Step 4: Wait for Green Checkmark ✅
- Deployment should complete in ~2-3 minutes
- You'll see a green checkmark when done
- Look for **Public URL** (top right) - copy it!

**Example:** `https://railway-production-abc123.up.railway.app`

**Save this! ⭐ You need it for Vercel.**

### ✅ Backend is LIVE!

Test: Paste your Railway URL in browser → should see JSON.

---

## 🚀 PART 3: Deploy Frontend on Vercel (2 min)

### Step 1: Go to Vercel Dashboard
Open: https://vercel.com/dashboard

If not logged in, click **Sign Up** → **GitHub** to connect.

### Step 2: Import GitHub Project
1. Click **Add New** → **Project**
2. Click **Import Git Repository**
3. Paste: `https://github.com/psahani3486/Reinforcement-learning-Recommendation-Project`
4. Click **Continue**

### Step 3: Configure (Should be auto-filled)
Verify these settings (they should be automatic):

- **Framework Preset:** `Vite`
- **Root Directory:** `frontend`
- **Build Command:** `npm run build`
- **Output Directory:** `dist`

All correct? Click **Deploy** ✅

### Step 4: Add Environment Variable (While deploying)
1. While it's deploying, go to **Settings** → **Environment Variables** (top tabs)
2. Click **Add New**
3. Fill in:
   - **Name:** `VITE_API_URL`
   - **Value:** Your Railway backend URL (from Part 2, Step 4)
   - **Example:** `https://railway-production-abc123.up.railway.app`
4. Click **Save**

### Step 5: Redeploy with Environment Variable
1. Go back to **Deployments** tab
2. Find the first deployment (top row)
3. Click **...** (three dots) → **Redeploy**
4. Click **Redeploy** to confirm
5. Wait for green **Ready** ✅

### Step 6: Get Your Frontend URL
- Click on the **Ready** deployment
- Your frontend URL is shown at the top
- **Example:** `https://rl-recommendation.vercel.app`
- **Copy it! ⭐**

### ✅ Frontend is LIVE!

---

## 🔄 FINAL STEP: Update Backend CORS

Now that you have your **Vercel frontend URL**, update the backend:

### Go Back to Railway
1. Open: https://railway.app/dashboard
2. Click on your backend service
3. Go to **Variables** tab
4. Find **CORS_ORIGIN**
5. Change value from `https://rl-recommendation.vercel.app` to your actual Vercel URL
6. Save (auto-saves)
7. Railway auto-redeploys ✅

---

## ✅ VERIFICATION: Test Your Live App

### Test 1: Open Frontend
1. Go to your **Vercel URL** in browser
2. See the app load? ✅

### Test 2: Check Network
1. Open **DevTools** (F12)
2. Go to **Network** tab
3. Try to log in or use any feature
4. Check that API calls go to your Railway URL
5. Should see **Status 200** (success)
6. No CORS errors? ✅✅✅

### 🎉 YOU'RE DONE!

---

## 📋 Reference: Your URLs

**Frontend:** `https://rl-recommendation.vercel.app` (or your Vercel URL)

**Backend API:** `https://railway-production-abc123.up.railway.app` (or your Railway URL)

**Database:** MongoDB Atlas (automatically working)

---

## 🚨 If Something Goes Wrong

### "API not reachable"
- Check `VITE_API_URL` in Vercel (correct Railway URL?)
- Check `CORS_ORIGIN` in Railway (correct Vercel URL?)
- Redeploy both

### "MongoDB connection failed"
- Check `MONGODB_URI` in Railway (correct connection string?)
- Check IP whitelist in MongoDB Atlas (should be `0.0.0.0/0`)
- Check username/password in connection string

### "Deployment failed"
- Check logs in Railway/Vercel dashboard
- Ensure `backend/package.json` and `frontend/package.json` exist
- Try redeploying manually

---

## 📚 Need Help?

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- MongoDB Docs: https://docs.atlas.mongodb.com

---

**That's it! Your app is now live on the internet.** 🚀

Share your frontend URL with anyone and they can use your app! 🎉
