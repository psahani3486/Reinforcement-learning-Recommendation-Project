import mongoose from 'mongoose';

const connectDB = async () => {
  try {
    const mongoUri = process.env.MONGODB_URI;
    
    if (!mongoUri) {
      throw new Error(
        'MONGODB_URI is not defined. Please set it in your environment variables.\n' +
        'Example: mongodb+srv://username:password@cluster.mongodb.net/dbname?retryWrites=true&w=majority'
      );
    }

    if (!mongoUri.includes('@')) {
      throw new Error(
        'Invalid MONGODB_URI format. It appears incomplete. ' +
        'Ensure it includes username, password, and cluster details.\n' +
        'Expected format: mongodb+srv://username:password@cluster.mongodb.net/dbname'
      );
    }

    const conn = await mongoose.connect(mongoUri);

    console.log(`MongoDB Connected: ${conn.connection.host}`);
    return conn;
  } catch (error) {
    console.error(`MongoDB Connection Error: ${error.message}`);
    process.exit(1);
  }
};

export default connectDB;
