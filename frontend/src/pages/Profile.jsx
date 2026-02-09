import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { User, Mail, Settings, Tag, Save } from 'lucide-react';
import Navbar from '../components/Navbar';
import { useAuthStore } from '../context/authStore';
import { authService } from '../services/api';
import toast from 'react-hot-toast';

const Profile = () => {
  const { user, updateUser } = useAuthStore();
  const [genres, setGenres] = useState(user?.preferences?.genres || []);
  const [newGenre, setNewGenre] = useState('');
  const [saving, setSaving] = useState(false);

  const availableGenres = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'Western', 'Documentary'
  ];

  const handleAddGenre = (genre) => {
    if (!genres.includes(genre)) {
      setGenres([...genres, genre]);
    }
  };

  const handleRemoveGenre = (genre) => {
    setGenres(genres.filter((g) => g !== genre));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const { data } = await authService.updatePreferences({ genres });
      updateUser(data);
      toast.success('Preferences saved successfully!');
    } catch (error) {
      toast.error('Failed to save preferences');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-2 gradient-text">Your Profile</h1>
          <p className="text-gray-400">Manage your account and preferences</p>
        </motion.div>

        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-dark-800 rounded-xl p-8 shadow-xl"
          >
            <div className="flex items-center gap-4 mb-6">
              <motion.div
                whileHover={{ scale: 1.1, rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="w-20 h-20 bg-gradient-to-r from-primary-600 to-purple-600 rounded-full flex items-center justify-center text-3xl font-bold"
              >
                {user?.username?.[0]?.toUpperCase()}
              </motion.div>
              <div>
                <h2 className="text-2xl font-bold">{user?.username}</h2>
                <p className="text-gray-400">{user?.email}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  <User size={16} />
                  Username
                </label>
                <div className="bg-dark-700 px-4 py-3 rounded-lg text-white">
                  {user?.username}
                </div>
              </div>

              <div>
                <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-2">
                  <Mail size={16} />
                  Email
                </label>
                <div className="bg-dark-700 px-4 py-3 rounded-lg text-white">
                  {user?.email}
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-dark-800 rounded-xl p-8 shadow-xl"
          >
            <div className="flex items-center gap-2 mb-6">
              <Settings size={24} className="text-primary-500" />
              <h2 className="text-2xl font-bold">Preferences</h2>
            </div>

            <div className="mb-6">
              <label className="flex items-center gap-2 text-sm font-medium text-gray-400 mb-3">
                <Tag size={16} />
                Favorite Genres
              </label>

              <div className="flex flex-wrap gap-2 mb-4">
                {genres.map((genre, index) => (
                  <motion.span
                    key={genre}
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ delay: index * 0.05 }}
                    exit={{ scale: 0, rotate: 180 }}
                    className="bg-gradient-to-r from-primary-600 to-purple-600 text-white px-4 py-2 rounded-full font-medium flex items-center gap-2 group"
                  >
                    {genre}
                    <motion.button
                      whileHover={{ scale: 1.2, rotate: 90 }}
                      onClick={() => handleRemoveGenre(genre)}
                      className="w-5 h-5 bg-white/20 hover:bg-white/30 rounded-full flex items-center justify-center transition-colors"
                    >
                      ×
                    </motion.button>
                  </motion.span>
                ))}
              </div>

              <div className="mb-4">
                <p className="text-sm text-gray-500 mb-2">Select genres you enjoy:</p>
                <div className="flex flex-wrap gap-2">
                  {availableGenres
                    .filter((g) => !genres.includes(g))
                    .map((genre, index) => (
                      <motion.button
                        key={genre}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3 + index * 0.02 }}
                        whileHover={{ scale: 1.05, y: -2 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => handleAddGenre(genre)}
                        className="bg-dark-700 hover:bg-dark-600 text-gray-300 px-3 py-1.5 rounded-full text-sm font-medium transition-colors"
                      >
                        + {genre}
                      </motion.button>
                    ))}
                </div>
              </div>
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSave}
              disabled={saving}
              className="w-full bg-gradient-to-r from-primary-600 to-purple-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg hover:shadow-primary-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <Save size={20} />
              {saving ? 'Saving...' : 'Save Preferences'}
            </motion.button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-dark-800 rounded-xl p-8 shadow-xl"
          >
            <h2 className="text-2xl font-bold mb-4">Account Stats</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-dark-700 p-4 rounded-lg">
                <p className="text-gray-400 text-sm mb-1">Total Interactions</p>
                <p className="text-2xl font-bold">{user?.totalInteractions || 0}</p>
              </div>
              <div className="bg-dark-700 p-4 rounded-lg">
                <p className="text-gray-400 text-sm mb-1">Total Reward</p>
                <p className="text-2xl font-bold text-primary-500">{user?.totalReward || 0}</p>
              </div>
              <div className="bg-dark-700 p-4 rounded-lg">
                <p className="text-gray-400 text-sm mb-1">Member Since</p>
                <p className="text-sm font-medium">
                  {user?.createdAt
                    ? new Date(user.createdAt).toLocaleDateString()
                    : 'N/A'}
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
