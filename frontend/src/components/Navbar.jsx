import { motion } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import { Home, BarChart3, History, User, LogOut, Sparkles, Search, Bookmark, Users, Activity, PlayCircle } from 'lucide-react';
import { useAuthStore } from '../context/authStore';
import ThemeToggle from './ThemeToggle';

const Navbar = () => {
  const location = useLocation();
  const { user, logout } = useAuthStore();

  const navItems = [
    { path: '/dashboard', icon: Home, label: 'Home' },
    { path: '/search', icon: Search, label: 'Search' },
    { path: '/watchlist', icon: Bookmark, label: 'Watchlist' },
    { path: '/social', icon: Users, label: 'Social' },
    { path: '/stats', icon: BarChart3, label: 'Stats' },
    { path: '/analytics', icon: Activity, label: 'Analytics' },
    { path: '/sessions', icon: PlayCircle, label: 'Sessions' },
    { path: '/history', icon: History, label: 'History' },
    { path: '/profile', icon: User, label: 'Profile' },
  ];

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100 }}
      className="bg-dark-800/80 backdrop-blur-lg border-b border-dark-700 sticky top-0 z-50"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/dashboard" className="flex items-center gap-2">
            <motion.div
              whileHover={{ rotate: 180, scale: 1.1 }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-r from-primary-600 to-purple-600 p-2 rounded-lg"
            >
              <Sparkles size={24} className="text-white" />
            </motion.div>
            <motion.span
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xl font-bold gradient-text"
            >
              RL Recommender
            </motion.span>
          </Link>

          <div className="flex items-center gap-6">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              return (
                <Link key={item.path} to={item.path}>
                  <motion.div
                    whileHover={{ scale: 1.1, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-primary-600 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-dark-700'
                    }`}
                  >
                    <Icon size={20} />
                    <span className="hidden md:block font-medium">{item.label}</span>
                  </motion.div>
                </Link>
              );
            })}

            <div className="flex items-center gap-3">
              <ThemeToggle />
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3 }}
                className="flex items-center gap-2 bg-dark-700 px-4 py-2 rounded-full"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-primary-600 to-purple-600 rounded-full flex items-center justify-center text-sm font-bold">
                  {user?.username?.[0]?.toUpperCase()}
                </div>
                <span className="hidden md:block text-sm font-medium">{user?.username}</span>
              </motion.div>
            </div>

            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              onClick={logout}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white transition-colors"
            >
              <LogOut size={20} />
              <span className="hidden md:block font-medium">Logout</span>
            </motion.button>
          </div>
        </div>
      </div>

      <motion.div
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.5 }}
        className="h-0.5 bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600"
        style={{ originX: 0 }}
      />
    </motion.nav>
  );
};

export default Navbar;
