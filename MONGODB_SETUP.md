# MongoDB Setup for Railway Deployment

## Option 1: Use MongoDB Atlas (Recommended)

### Step 1: Create a MongoDB Atlas Account
1. Go to [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up for a free account
3. Create a new organization and project

### Step 2: Create a Cluster
1. Click "Create" to deploy a cluster
2. Select **M0 Tier** (Free forever, 0.512 GB storage)
3. Choose your preferred region
4. Wait for the cluster to be created

### Step 3: Create Database User
1. Go to **Database Access** in the sidebar
2. Click **Add New Database User**
3. Set username: `admin`
4. Set password (keep it secure, you'll need this)
5. Click **Add User**

### Step 4: Get Connection String
1. Go to **Database** and click **Connect**
2. Select **Drivers** (not Shell)
3. Copy the connection string, it should look like:
   ```
   mongodb+srv://admin:PASSWORD@cluster0.mongodb.net/?retryWrites=true&w=majority
   ```

### Step 5: Add Database Name
Replace the connection string with:
```
mongodb+srv://admin:PASSWORD@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

## Option 2: Set Up in Railway

### Step 1: Navigate to Your Service
1. Go to your Railway project dashboard
2. Click on your "Reinforcement-learning..." service
3. Click on the **Variables** tab

### Step 2: Add MongoDB URI
1. Click **Add Variable**
2. **Key:** `MONGODB_URI`
3. **Value:** Paste your MongoDB Atlas connection string with the password filled in
   ```
   mongodb+srv://admin:MyMongo123@cluster0.mongodb.net/rl-recommendation?retryWrites=true&w=majority
   ```
4. Click **Add**

### Step 3: Deploy
1. Go back to **Deployments**
2. Click the three dots on the latest deployment
3. Select **Redeploy**
4. Wait for the new deployment to complete

## Verification

Check the build logs to confirm:
```
MongoDB Connected: cluster0.mongodb.net
```

## Troubleshooting

### Error: "Protocol and host list are required"
- Your `MONGODB_URI` is incomplete
- Ensure it includes: `mongodb+srv://username:password@hostname/dbname`

### Error: "Authentication failed"
- Check your MongoDB username and password
- Ensure special characters in password are not causing issues
- Try URL-encoding your password if it contains special characters

### Connection Timeout
- Check if your IP is whitelisted in MongoDB Atlas
- Go to **Network Access** → **Add IP Address** → **Allow Access from Anywhere** (0.0.0.0/0)

