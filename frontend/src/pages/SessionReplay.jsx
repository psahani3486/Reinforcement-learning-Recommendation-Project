import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { PlayCircle, Clock, Zap, Film, ChevronRight } from 'lucide-react';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import { sessionService } from '../services/api';

const SessionReplay = () => {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [replayData, setReplayData] = useState(null);
  const [replayIdx, setReplayIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const res = await sessionService.getUserSessions({ limit: 20 });
        setSessions(res.data.sessions);
      } catch (err) {
        toast.error('Failed to load sessions');
      } finally {
        setLoading(false);
      }
    };
    fetchSessions();
  }, []);

  const loadReplay = async (sessionId) => {
    try {
      const res = await sessionService.getSessionReplay(sessionId);
      setReplayData(res.data.session);
      setSelectedSession(sessionId);
      setReplayIdx(0);
      setIsPlaying(false);
    } catch (err) {
      toast.error('Failed to load session replay');
    }
  };

  useEffect(() => {
    if (!isPlaying || !replayData) return;
    if (replayIdx >= replayData.actions.length) {
      setIsPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setReplayIdx(prev => prev + 1);
    }, 1500);

    return () => clearTimeout(timer);
  }, [isPlaying, replayIdx, replayData]);

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
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <PlayCircle className="text-primary-500" />
            Session Replay
          </h1>
          <p className="text-gray-400">Review your browsing sessions and understand recommendations</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Session List */}
          <div className="lg:col-span-1">
            <h3 className="text-sm font-semibold text-gray-400 uppercase mb-3">Sessions</h3>
            <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-2">
              {sessions.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No sessions recorded yet</p>
              ) : (
                sessions.map((session, idx) => (
                  <motion.button
                    key={session._id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    onClick={() => loadReplay(session._id)}
                    className={`w-full text-left p-3 rounded-xl border transition-colors ${
                      selectedSession === session._id
                        ? 'bg-primary-600/20 border-primary-500'
                        : 'bg-dark-800 border-dark-700 hover:border-dark-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">
                          {new Date(session.createdAt).toLocaleDateString()}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {new Date(session.createdAt).toLocaleTimeString()}
                        </p>
                      </div>
                      <div className="text-right">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          session.abGroup === 'rl' ? 'bg-green-600/20 text-green-400' :
                          session.abGroup === 'baseline' ? 'bg-blue-600/20 text-blue-400' :
                          'bg-yellow-600/20 text-yellow-400'
                        }`}>
                          {session.abGroup}
                        </span>
                        {session.duration && (
                          <p className="text-xs text-gray-500 mt-1 flex items-center gap-1 justify-end">
                            <Clock size={10} /> {Math.round(session.duration)}s
                          </p>
                        )}
                      </div>
                    </div>
                  </motion.button>
                ))
              )}
            </div>
          </div>

          {/* Replay View */}
          <div className="lg:col-span-2">
            {!replayData ? (
              <div className="text-center py-20 text-gray-500">
                <PlayCircle size={64} className="mx-auto mb-4 opacity-30" />
                <p>Select a session to view replay</p>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold">Session Replay</h3>
                    <p className="text-sm text-gray-500">
                      Strategy: <span className="capitalize text-primary-400">{replayData.abGroup}</span>
                      {' · '}{replayData.actions.length} actions
                      {replayData.duration && ` · ${Math.round(replayData.duration)}s`}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => { setIsPlaying(!isPlaying); if (replayIdx >= replayData.actions.length) setReplayIdx(0); }}
                      className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium transition-colors"
                    >
                      {isPlaying ? 'Pause' : replayIdx >= replayData.actions.length ? 'Restart' : 'Play'}
                    </motion.button>
                  </div>
                </div>

                {/* Progress */}
                <div className="w-full h-2 bg-dark-700 rounded-full mb-6 overflow-hidden">
                  <motion.div
                    animate={{ width: `${replayData.actions.length > 0 ? (replayIdx / replayData.actions.length) * 100 : 0}%` }}
                    className="h-full bg-primary-500 rounded-full"
                  />
                </div>

                {/* Timeline */}
                <div className="space-y-3 max-h-[55vh] overflow-y-auto pr-2">
                  {replayData.actions.map((action, idx) => {
                    const isVisible = idx <= replayIdx;
                    const isCurrent = idx === replayIdx;

                    return (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{
                          opacity: isVisible ? 1 : 0.3,
                          x: isVisible ? 0 : -10,
                          scale: isCurrent ? 1.02 : 1
                        }}
                        className={`flex items-center gap-4 p-3 rounded-xl border transition-colors ${
                          isCurrent
                            ? 'bg-primary-600/10 border-primary-500'
                            : 'bg-dark-800 border-dark-700'
                        }`}
                      >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                          action.type === 'purchase' || action.type === 'click'
                            ? 'bg-green-600/20 text-green-400'
                            : action.type === 'skip'
                            ? 'bg-red-600/20 text-red-400'
                            : 'bg-blue-600/20 text-blue-400'
                        }`}>
                          {idx + 1}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium capitalize">{action.type}</p>
                          {action.movie && (
                            <p className="text-xs text-gray-500">{action.movie.title}</p>
                          )}
                        </div>
                        {action.timestamp && (
                          <span className="text-xs text-gray-600">
                            {new Date(action.timestamp).toLocaleTimeString()}
                          </span>
                        )}
                        <ChevronRight size={16} className="text-gray-600" />
                      </motion.div>
                    );
                  })}
                </div>

                {replayData.actions.length === 0 && (
                  <div className="text-center py-12 text-gray-500">
                    <Zap size={40} className="mx-auto mb-2 opacity-50" />
                    <p>No actions recorded in this session</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionReplay;
