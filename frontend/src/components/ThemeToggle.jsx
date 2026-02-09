import { motion } from 'framer-motion';
import { Sun, Moon } from 'lucide-react';
import { useThemeStore } from '../context/themeStore';

const ThemeToggle = () => {
  const { isDark, toggleTheme } = useThemeStore();

  return (
    <motion.button
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      onClick={toggleTheme}
      className="relative p-2 rounded-lg bg-dark-700 hover:bg-dark-600 transition-colors"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <motion.div
        animate={{ rotate: isDark ? 0 : 180 }}
        transition={{ duration: 0.3 }}
      >
        {isDark ? (
          <Moon size={18} className="text-yellow-400" />
        ) : (
          <Sun size={18} className="text-yellow-500" />
        )}
      </motion.div>
    </motion.button>
  );
};

export default ThemeToggle;
