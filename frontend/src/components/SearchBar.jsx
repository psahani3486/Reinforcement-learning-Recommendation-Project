import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, SlidersHorizontal, X } from 'lucide-react';

const SearchBar = ({ onSearch, genres = [] }) => {
  const [query, setQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    genre: '',
    yearMin: '',
    yearMax: '',
    ratingMin: '',
    sortBy: 'relevance'
  });

  const handleSearch = (e) => {
    e?.preventDefault();
    onSearch({ q: query, ...filters });
  };

  const clearFilters = () => {
    setFilters({ genre: '', yearMin: '', yearMax: '', ratingMin: '', sortBy: 'relevance' });
    setQuery('');
    onSearch({});
  };

  return (
    <div className="w-full">
      <form onSubmit={handleSearch} className="relative">
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search movies by title..."
              className="w-full pl-12 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 text-white placeholder-gray-500"
            />
          </div>
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowFilters(!showFilters)}
            className={`px-4 py-3 rounded-xl border transition-colors ${
              showFilters ? 'bg-primary-600 border-primary-500 text-white' : 'bg-dark-800 border-dark-600 text-gray-400 hover:text-white'
            }`}
          >
            <SlidersHorizontal size={20} />
          </motion.button>
          <motion.button
            type="submit"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-semibold transition-colors"
          >
            Search
          </motion.button>
        </div>
      </form>

      {showFilters && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-4 p-4 bg-dark-800 rounded-xl border border-dark-700"
        >
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-300">Filters</h4>
            <button onClick={clearFilters} className="text-xs text-gray-500 hover:text-red-400 flex items-center gap-1">
              <X size={14} /> Clear all
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <select
              value={filters.genre}
              onChange={(e) => setFilters({ ...filters, genre: e.target.value })}
              className="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-primary-500 focus:outline-none"
            >
              <option value="">All Genres</option>
              {genres.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
            <input
              type="number"
              placeholder="Year from"
              value={filters.yearMin}
              onChange={(e) => setFilters({ ...filters, yearMin: e.target.value })}
              className="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />
            <input
              type="number"
              placeholder="Year to"
              value={filters.yearMax}
              onChange={(e) => setFilters({ ...filters, yearMax: e.target.value })}
              className="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />
            <input
              type="number"
              step="0.1"
              min="0"
              max="5"
              placeholder="Min rating"
              value={filters.ratingMin}
              onChange={(e) => setFilters({ ...filters, ratingMin: e.target.value })}
              className="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-primary-500 focus:outline-none"
            />
            <select
              value={filters.sortBy}
              onChange={(e) => setFilters({ ...filters, sortBy: e.target.value })}
              className="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-primary-500 focus:outline-none"
            >
              <option value="relevance">Relevance</option>
              <option value="rating">Rating</option>
              <option value="popularity">Popularity</option>
              <option value="year">Year</option>
              <option value="title">Title</option>
            </select>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default SearchBar;
