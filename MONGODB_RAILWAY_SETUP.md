# Quick MongoDB Setup for Railway

## Step 1: Get Your MongoDB Connection String

If you already have MongoDB Atlas set up, your connection string should look like:
```
mongodb+srv://admin:MyMongo123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

### ⚠️ Important: Handle Special Characters
If your password has special characters (like `!`, `@`, `#`, etc.), URL-encode them:
- `!` → `%21`
- `@` → `%40`
- `#` → `%23`
- `$` → `%24`

So if your password is `MyMongo123!`, use: `MyMongo123%21`

## Step 2: Add to Railway Variables

1. **Open Railway Dashboard**
   - Go to [railway.app](https://railway.app)
   - Click on your project: "Reinforcement-learning-Recommendation-Project"

2. **Go to Your Service**
   - Click on the "striking-energy" service (or your backend service name)

3. **Open Variables Tab**
   - Click on the **Variables** tab
   - Look for existing `MONGODB_URI` variable

4. **Update or Create Variable**
   - If it exists, click the pencil icon to edit
   - If not, click **Add Variable**
   
5. **Set the Values**
   - **Key:** `MONGODB_URI`
   - **Value:** `mongodb+srv://admin:MyMongo123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority`
   
   Replace with your actual:
   - `admin` = your MongoDB username
   - `MyMongo123` = your MongoDB password (URL-encoded if special chars)
   - `cluster0` = your cluster name
   - `rl-recommendation` = your database name

6. **Save**
   - Click the checkmark or press Enter

## Step 3: Redeploy

1. Go back to **Deployments** tab
2. Find the latest failed deployment
3. Click the **...** (three dots) menu
4. Select **Redeploy**
5. Wait for build and deployment to complete

## Verification

Once deployed, check the Deploy Logs for:
```
✓ MongoDB Connected: cluster0.mongodb.net
✓ Server running on port 5000
```

## Example MongoDB Strings

### MongoDB Atlas (Cloud) - Free Tier
```
mongodb+srv://admin:password@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

### With URL-Encoded Password
If password is `Pass!word@123`:
```
mongodb+srv://admin:Pass%21word%40123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

### Local Development (when testing locally)
```
mongodb://localhost:27017/rl-recommendation
```

## Need Help Finding Your Connection String?

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Login to your account
3. Select your organization and project
4. Click "Connect" on your cluster
5. Select "Drivers" (not Shell)
6. Copy the connection string
7. Add your password and database name

