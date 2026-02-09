import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Activity, Target, Award } from 'lucide-react';
import Navbar from '../components/Navbar';
import AnimatedChart from '../components/AnimatedChart';
import LoadingSpinner from '../components/LoadingSpinner';
import { recommendationService } from '../services/api';
import toast from 'react-hot-toast';

const Stats = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const { data } = await recommendationService.getStats();
      setStats(data);
    } catch (error) {
      toast.error('Failed to fetch statistics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-900">
        <Navbar />
        <LoadingSpinner fullScreen />
      </div>
    );
  }

  const interactionData = stats?.interactionsByType?.map((item) => ({
    name: item._id,
    count: item.count,
    reward: item.totalReward,
  })) || [];

  const recentActivityData = stats?.recentActivity?.map((item, index) => ({
    day: `Day ${index + 1}`,
    reward: item.reward,
  })) || [];

  const statCards = [
    {
      title: 'Total Interactions',
      value: stats?.totalInteractions || 0,
      icon: Activity,
      color: 'from-blue-600 to-blue-700',
      iconColor: 'text-blue-200',
    },
    {
      title: 'Total Reward',
      value: stats?.totalReward || 0,
      icon: Award,
      color: 'from-purple-600 to-purple-700',
      iconColor: 'text-purple-200',
    },
    {
      title: 'Avg Reward',
      value: stats?.totalInteractions > 0
        ? (stats.totalReward / stats.totalInteractions).toFixed(2)
        : 0,
      icon: TrendingUp,
      color: 'from-green-600 to-green-700',
      iconColor: 'text-green-200',
    },
    {
      title: 'Preferences Set',
      value: stats?.preferences?.genres?.length || 0,
      icon: Target,
      color: 'from-pink-600 to-pink-700',
      iconColor: 'text-pink-200',
    },
  ];

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-2 gradient-text">Your Statistics</h1>
          <p className="text-gray-400">Track your engagement and performance metrics</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {statCards.map((card, index) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={card.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`bg-gradient-to-br ${card.color} rounded-xl p-6 shadow-xl`}
              >
                <div className="flex items-center justify-between mb-2">
                  <p className="text-white/80 text-sm font-medium">{card.title}</p>
                  <Icon size={24} className={card.iconColor} />
                </div>
                <motion.p
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.1 + 0.2, type: 'spring' }}
                  className="text-4xl font-bold text-white"
                >
                  {card.value}
                </motion.p>
              </motion.div>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <AnimatedChart
            type="bar"
            data={interactionData}
            dataKey="count"
            xKey="name"
            title="Interactions by Type"
            color="#0ea5e9"
          />

          <AnimatedChart
            type="pie"
            data={interactionData}
            dataKey="count"
            title="Interaction Distribution"
          />
        </div>

        {recentActivityData.length > 0 && (
          <div className="grid grid-cols-1 gap-6 mb-8">
            <AnimatedChart
              type="line"
              data={recentActivityData}
              dataKey="reward"
              xKey="day"
              title="Recent Activity Rewards"
              color="#8b5cf6"
            />
          </div>
        )}

        {stats?.preferences?.genres && stats.preferences.genres.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-dark-800 rounded-xl p-6 shadow-xl"
          >
            <h3 className="text-xl font-bold mb-4 gradient-text">Your Genre Preferences</h3>
            <div className="flex flex-wrap gap-3">
              {stats.preferences.genres.map((genre, index) => (
                <motion.span
                  key={genre}
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ delay: 0.6 + index * 0.1, type: 'spring' }}
                  className="bg-gradient-to-r from-primary-600 to-purple-600 text-white px-4 py-2 rounded-full font-medium"
                >
                  {genre}
                </motion.span>
              ))}
            </div>
          </motion.div>
        )}

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-8 text-center"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={fetchStats}
            className="bg-primary-600 hover:bg-primary-700 text-white px-6 py-3 rounded-lg font-semibold"
          >
            Refresh Stats
          </motion.button>
        </motion.div>
      </div>
    </div>
  );
};

export default Stats;
