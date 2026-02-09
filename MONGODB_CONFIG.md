# MongoDB Configuration - Your Connection String

Your MongoDB Atlas connection string:
```
mongodb+srv://psahani729_db_user:<db_password>@psahani729.aaqecg1.mongodb.net/?appName=psahani729
```

## Step 1: Complete Your Connection String

Add the database name and parameters:
```
mongodb+srv://psahani729_db_user:<db_password>@psahani729.aaqecg1.mongodb.net/rl-recommendation?retryWrites=true&w=majority&appName=psahani729
```

Replace `<db_password>` with your actual MongoDB password.

**Example (with fake password):**
```
mongodb+srv://psahani729_db_user:MyPassword123@psahani729.aaqecg1.mongodb.net/rl-recommendation?retryWrites=true&w=majority&appName=psahani729
```

## Step 2: Add to Railway

1. **Go to [railway.app/dashboard](https://railway.app/dashboard)**

2. **Open your project:** "Reinforcement-learning-Recommendation-Project"

3. **Click on the backend service** (striking-energy)

4. **Go to Variables tab**

5. **Find or Create `MONGODB_URI` variable:**
   - If it exists, click the pencil ✏️ to edit
   - If not, click **Add Variable**

6. **Paste your complete connection string:**
   ```
   mongodb+srv://psahani729_db_user:YOUR_PASSWORD@psahani729.aaqecg1.mongodb.net/rl-recommendation?retryWrites=true&w=majority&appName=psahani729
   ```

7. **Press Enter to save**

## Step 3: Redeploy

1. Go to **Deployments** tab
2. Click the **...** (three dots) on the latest deployment
3. Select **Redeploy**
4. Wait for green checkmark ✅

## Verification

Check the Deploy Logs for:
```
✓ MongoDB Connected: psahani729.aaqecg1.mongodb.net
✓ Server running on port 5000
```

## Credentials Summary

- **Username:** `psahani729_db_user`
- **Cluster:** `psahani729.aaqecg1`
- **Database:** `rl-recommendation`
- **Password:** (Get from your MongoDB Atlas dashboard)

