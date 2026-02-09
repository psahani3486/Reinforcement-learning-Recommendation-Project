import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3, Users, Film, TrendingUp, Activity,
  PieChart as PieChartIcon, Target, Zap
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, AreaChart, Area, Legend
} from 'recharts';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import { analyticsService } from '../services/api';

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#ef4444', '#6366f1'];

const Analytics = () => {
  const [data, setData] = useState(null);
  const [ctrData, setCtrData] = useState([]);
  const [funnel, setFunnel] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [dashRes, ctrRes, funnelRes] = await Promise.all([
          analyticsService.getDashboardStats(),
          analyticsService.getCTRTrend(30),
          analyticsService.getUserFunnel()
        ]);
        setData(dashRes.data);
        setCtrData(ctrRes.data.ctrData);
        setFunnel(funnelRes.data.funnel);
      } catch (err) {
        toast.error('Failed to load analytics');
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-900">
        <Navbar />
        <div className="flex justify-center items-center h-[60vh]"><LoadingSpinner /></div>
      </div>
    );
  }

  const overview = data?.overview || {};

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <BarChart3 className="text-primary-500" />
            Analytics Dashboard
          </h1>
          <p className="text-gray-400">System-wide metrics and A/B test results</p>
        </motion.div>

        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {[
            { label: 'Total Users', value: overview.totalUsers, icon: Users, color: 'from-blue-500 to-cyan-500' },
            { label: 'Total Interactions', value: overview.totalInteractions, icon: Activity, color: 'from-purple-500 to-pink-500' },
            { label: 'Total Movies', value: overview.totalMovies, icon: Film, color: 'from-green-500 to-emerald-500' },
          ].map((card, idx) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-dark-800 rounded-xl p-6 border border-dark-700"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-400">{card.label}</p>
                  <p className="text-3xl font-bold mt-1">{card.value || 0}</p>
                </div>
                <div className={`p-3 rounded-xl bg-gradient-to-r ${card.color}`}>
                  <card.icon size={24} className="text-white" />
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Daily Interactions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-dark-800 rounded-xl p-6 border border-dark-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <TrendingUp size={20} className="text-primary-500" />
              Daily Interactions
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={data?.dailyInteractions || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="_id" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
                <Area type="monotone" dataKey="count" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.2} />
                <Area type="monotone" dataKey="avgReward" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.1} />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Interaction Distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-dark-800 rounded-xl p-6 border border-dark-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <PieChartIcon size={20} className="text-purple-500" />
              Interaction Distribution
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={(data?.interactionDistribution || []).map(d => ({ name: d._id, value: d.count }))}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {(data?.interactionDistribution || []).map((_, idx) => (
                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* CTR Trend */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-dark-800 rounded-xl p-6 border border-dark-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Target size={20} className="text-green-500" />
              Click-Through Rate Trend
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={ctrData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="_id" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="ctr" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Genre Popularity */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-dark-800 rounded-xl p-6 border border-dark-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap size={20} className="text-yellow-500" />
              Genre Popularity
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data?.genrePopularity || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" stroke="#9ca3af" />
                <YAxis dataKey="_id" type="category" stroke="#9ca3af" width={80} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* User Funnel & A/B Test Results */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-dark-800 rounded-xl p-6 border border-dark-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Users size={20} className="text-cyan-500" />
              User Funnel
            </h3>
            <div className="space-y-3">
              {funnel.map((stage, idx) => {
                const maxCount = funnel[0]?.count || 1;
                const pct = ((stage.count / maxCount) * 100).toFixed(0);
                return (
                  <div key={stage.stage}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">{stage.stage}</span>
                      <span className="font-semibold">{stage.count} ({pct}%)</span>
                    </div>
                    <div className="w-full h-3 bg-dark-700 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ delay: 0.7 + idx * 0.1 }}
                        className="h-full rounded-full"
                        style={{ background: COLORS[idx % COLORS.length] }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>

          {/* A/B Test */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="bg-dark-800 rounded-xl p-6 border border-dark-700"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity size={20} className="text-pink-500" />
              A/B Test Results
            </h3>
            {data?.abTestResults?.length > 0 ? (
              <div className="space-y-4">
                {data.abTestResults.map((group, idx) => (
                  <div key={group._id} className="bg-dark-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold capitalize">{group._id} Group</span>
                      <span className="text-sm text-gray-400">{group.sessions} sessions</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="text-gray-500">Avg Duration</p>
                        <p className="font-semibold">{(group.avgDuration || 0).toFixed(0)}s</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Avg Actions</p>
                        <p className="font-semibold">{(group.avgActions || 0).toFixed(1)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Activity size={40} className="mx-auto mb-2 opacity-50" />
                <p>No A/B test data yet. Start sessions to collect data.</p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Top Movies */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-dark-800 rounded-xl p-6 border border-dark-700"
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Film size={20} className="text-yellow-500" />
            Top Movies by Interactions
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-dark-700">
                  <th className="text-left py-3 px-2">#</th>
                  <th className="text-left py-3 px-2">Title</th>
                  <th className="text-right py-3 px-2">Interactions</th>
                  <th className="text-right py-3 px-2">Avg Reward</th>
                </tr>
              </thead>
              <tbody>
                {(data?.topMovies || []).map((item, idx) => (
                  <tr key={item._id} className="border-b border-dark-700/50 hover:bg-dark-700/30">
                    <td className="py-3 px-2 text-gray-500">{idx + 1}</td>
                    <td className="py-3 px-2 font-medium">{item.movie?.title || `Movie #${item._id}`}</td>
                    <td className="py-3 px-2 text-right text-primary-400">{item.count}</td>
                    <td className="py-3 px-2 text-right">
                      <span className={item.avgReward >= 0 ? 'text-green-400' : 'text-red-400'}>
                        {item.avgReward?.toFixed(1)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Analytics;
