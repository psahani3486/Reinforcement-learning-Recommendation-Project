import { motion, AnimatePresence } from 'framer-motion';
import { Lightbulb, ChevronDown, ChevronUp, Sparkles, Users, Star, TrendingUp, Compass } from 'lucide-react';
import { useState } from 'react';

const iconMap = {
  genre: Sparkles,
  collaborative: Users,
  rating: Star,
  popularity: TrendingUp,
  explore: Compass,
};

const colorMap = {
  genre: 'from-blue-500 to-cyan-500',
  collaborative: 'from-purple-500 to-pink-500',
  rating: 'from-yellow-500 to-orange-500',
  popularity: 'from-green-500 to-emerald-500',
  explore: 'from-indigo-500 to-violet-500',
};

const ExplanationPanel = ({ explanations = [] }) => {
  const [expanded, setExpanded] = useState(false);

  if (!explanations || explanations.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-primary-400 hover:text-primary-300 transition-colors"
      >
        <Lightbulb size={14} />
        <span>Why this?</span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2 space-y-1.5 overflow-hidden"
          >
            {explanations.map((exp, idx) => {
              const Icon = iconMap[exp.type] || Lightbulb;
              const gradient = colorMap[exp.type] || 'from-gray-500 to-gray-600';

              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="flex items-start gap-2 text-xs"
                >
                  <div className={`p-1 rounded bg-gradient-to-r ${gradient} flex-shrink-0`}>
                    <Icon size={10} className="text-white" />
                  </div>
                  <span className="text-gray-400">{exp.text}</span>
                  <div className="ml-auto flex-shrink-0">
                    <div className="w-12 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(exp.weight || 0) * 100}%` }}
                        transition={{ delay: idx * 0.1 + 0.2 }}
                        className={`h-full rounded-full bg-gradient-to-r ${gradient}`}
                      />
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ExplanationPanel;
