import { motion } from 'framer-motion';

const LoadingSpinner = ({ size = 'md', fullScreen = false }) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
    xl: 'w-24 h-24',
  };

  const spinnerContent = (
    <div className="flex flex-col items-center justify-center gap-4">
      <motion.div
        className={`${sizeClasses[size]} border-4 border-primary-500 border-t-transparent rounded-full`}
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      />
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-gray-400 text-sm"
      >
        Loading recommendations...
      </motion.div>
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-dark-900/90 backdrop-blur-sm z-50">
        {spinnerContent}
      </div>
    );
  }

  return <div className="flex items-center justify-center p-8">{spinnerContent}</div>;
};

export default LoadingSpinner;
