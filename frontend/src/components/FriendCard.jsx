import { motion } from 'framer-motion';
import { UserPlus, UserCheck, UserX, User, Clock } from 'lucide-react';

const FriendCard = ({ user, status, friendshipId, onAction }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-dark-800 rounded-xl p-4 border border-dark-700 flex items-center gap-4"
    >
      <div className="w-12 h-12 bg-gradient-to-r from-primary-600 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
        <User size={22} className="text-white" />
      </div>

      <div className="flex-1 min-w-0">
        <p className="font-semibold truncate">{user.username}</p>
        <p className="text-sm text-gray-500 truncate">{user.email}</p>
      </div>

      <div className="flex gap-2 flex-shrink-0">
        {status === 'none' && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onAction('add', user._id)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm transition-colors"
          >
            <UserPlus size={16} />
            <span>Add</span>
          </motion.button>
        )}

        {status === 'pending' && (
          <div className="flex gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onAction('accept', friendshipId)}
              className="flex items-center gap-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm transition-colors"
            >
              <UserCheck size={16} />
              Accept
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onAction('reject', friendshipId)}
              className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm transition-colors"
            >
              <UserX size={16} />
              Reject
            </motion.button>
          </div>
        )}

        {status === 'friend' && (
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 px-3 py-1.5 bg-green-600/20 text-green-400 rounded-lg text-sm">
              <UserCheck size={16} />
              Friends
            </span>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onAction('remove', friendshipId)}
              className="px-2 py-1.5 text-gray-500 hover:text-red-400 transition-colors"
            >
              <UserX size={16} />
            </motion.button>
          </div>
        )}

        {status === 'sent' && (
          <span className="flex items-center gap-1 px-3 py-1.5 bg-yellow-600/20 text-yellow-400 rounded-lg text-sm">
            <Clock size={16} />
            Pending
          </span>
        )}
      </div>
    </motion.div>
  );
};

export default FriendCard;
