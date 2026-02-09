import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Users, UserPlus, Search, Bell, Activity, MessageCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar';
import FriendCard from '../components/FriendCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { socialService } from '../services/api';

const Social = () => {
  const [tab, setTab] = useState('friends');
  const [friends, setFriends] = useState([]);
  const [pending, setPending] = useState([]);
  const [activity, setActivity] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [friendsRes, pendingRes, activityRes] = await Promise.all([
        socialService.getFriends(),
        socialService.getPendingRequests(),
        socialService.getFriendActivity()
      ]);
      setFriends(friendsRes.data.friends);
      setPending(pendingRes.data.pending);
      setActivity(activityRes.data.activities);
    } catch (err) {
      toast.error('Failed to load social data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSearch = async () => {
    if (searchQuery.length < 2) return;
    try {
      const res = await socialService.searchUsers(searchQuery);
      setSearchResults(res.data.users);
    } catch (err) {
      toast.error('Search failed');
    }
  };

  const handleAction = async (action, id) => {
    try {
      if (action === 'add') {
        await socialService.sendFriendRequest(id);
        toast.success('Friend request sent!');
        setSearchResults(prev => prev.filter(u => u._id !== id));
      } else if (action === 'accept') {
        await socialService.respondFriendRequest(id, 'accepted');
        toast.success('Friend request accepted!');
        fetchData();
      } else if (action === 'reject') {
        await socialService.respondFriendRequest(id, 'rejected');
        toast.success('Friend request rejected');
        fetchData();
      } else if (action === 'remove') {
        await socialService.removeFriend(id);
        toast.success('Friend removed');
        fetchData();
      }
    } catch (err) {
      toast.error(err.response?.data?.message || 'Action failed');
    }
  };

  const tabs = [
    { id: 'friends', label: 'Friends', icon: Users, count: friends.length },
    { id: 'pending', label: 'Requests', icon: Bell, count: pending.length },
    { id: 'find', label: 'Find Friends', icon: UserPlus },
    { id: 'activity', label: 'Activity', icon: Activity },
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-900">
        <Navbar />
        <div className="flex justify-center items-center h-[60vh]"><LoadingSpinner /></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Users className="text-primary-500" />
            Social
          </h1>
          <p className="text-gray-400">Connect with friends and see what they're watching</p>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {tabs.map(t => (
            <motion.button
              key={t.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors whitespace-nowrap ${
                tab === t.id
                  ? 'bg-primary-600 text-white'
                  : 'bg-dark-800 text-gray-400 hover:bg-dark-700'
              }`}
            >
              <t.icon size={18} />
              {t.label}
              {t.count > 0 && (
                <span className="bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full">{t.count}</span>
              )}
            </motion.button>
          ))}
        </div>

        {/* Friends List */}
        {tab === 'friends' && (
          <div className="space-y-3">
            {friends.length === 0 ? (
              <div className="text-center py-16 text-gray-500">
                <Users size={48} className="mx-auto mb-3 opacity-50" />
                <p>No friends yet. Find and add friends!</p>
              </div>
            ) : (
              friends.map(friend => (
                <FriendCard
                  key={friend.friendshipId}
                  user={friend}
                  status="friend"
                  friendshipId={friend.friendshipId}
                  onAction={handleAction}
                />
              ))
            )}
          </div>
        )}

        {/* Pending Requests */}
        {tab === 'pending' && (
          <div className="space-y-3">
            {pending.length === 0 ? (
              <div className="text-center py-16 text-gray-500">
                <Bell size={48} className="mx-auto mb-3 opacity-50" />
                <p>No pending friend requests</p>
              </div>
            ) : (
              pending.map(req => (
                <FriendCard
                  key={req._id}
                  user={req.requester}
                  status="pending"
                  friendshipId={req._id}
                  onAction={handleAction}
                />
              ))
            )}
          </div>
        )}

        {/* Find Friends */}
        {tab === 'find' && (
          <div>
            <div className="flex gap-2 mb-6">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSearch()}
                  placeholder="Search users by username..."
                  className="w-full pl-12 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 text-white"
                />
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSearch}
                className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium"
              >
                Search
              </motion.button>
            </div>

            <div className="space-y-3">
              {searchResults.map(user => (
                <FriendCard
                  key={user._id}
                  user={user}
                  status="none"
                  onAction={handleAction}
                />
              ))}
              {searchQuery && searchResults.length === 0 && (
                <p className="text-center py-8 text-gray-500">No users found</p>
              )}
            </div>
          </div>
        )}

        {/* Friend Activity */}
        {tab === 'activity' && (
          <div className="space-y-4">
            {activity.length === 0 ? (
              <div className="text-center py-16 text-gray-500">
                <Activity size={48} className="mx-auto mb-3 opacity-50" />
                <p>No friend activity yet</p>
              </div>
            ) : (
              activity.map((item, idx) => (
                <motion.div
                  key={item._id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className="bg-dark-800 rounded-xl p-4 border border-dark-700 flex items-center gap-4"
                >
                  <div className={`p-2 rounded-lg ${
                    item.interactionType === 'purchase' ? 'bg-green-600/20 text-green-400' :
                    item.interactionType === 'skip' ? 'bg-red-600/20 text-red-400' :
                    'bg-blue-600/20 text-blue-400'
                  }`}>
                    <MessageCircle size={18} />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm">
                      <span className="font-semibold">{item.userId?.username || 'Someone'}</span>
                      {' '}{item.interactionType === 'purchase' ? 'liked' :
                            item.interactionType === 'skip' ? 'skipped' :
                            item.interactionType === 'click' ? 'viewed' : 'interacted with'}{' '}
                      <span className="text-primary-400">{item.movie?.title || `Movie #${item.movieId}`}</span>
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(item.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className={`text-sm font-semibold ${item.reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {item.reward > 0 ? '+' : ''}{item.reward}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Social;
