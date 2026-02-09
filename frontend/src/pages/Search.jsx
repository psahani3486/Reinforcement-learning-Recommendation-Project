import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Film, Search as SearchIcon } from 'lucide-react';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar';
import SearchBar from '../components/SearchBar';
import MovieCard from '../components/MovieCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { searchService, recommendationService } from '../services/api';

const Search = () => {
  const [movies, setMovies] = useState([]);
  const [genres, setGenres] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [pagination, setPagination] = useState({ page: 1, pages: 1, total: 0 });

  useEffect(() => {
    searchService.getGenres()
      .then(res => setGenres(res.data.genres))
      .catch(() => {});
  }, []);

  const handleSearch = async (params = {}) => {
    setLoading(true);
    setSearched(true);
    try {
      const res = await searchService.searchMovies({ ...params, limit: 12 });
      setMovies(res.data.movies);
      setPagination(res.data.pagination);
    } catch (err) {
      toast.error('Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleInteraction = async (movie, type) => {
    try {
      await recommendationService.recordInteraction({
        movieId: movie.movieId,
        interactionType: type,
        sessionId: Date.now().toString()
      });
      toast.success(`${type === 'purchase' ? 'Liked' : type === 'skip' ? 'Skipped' : 'Clicked'} ${movie.title}`);
    } catch (err) {
      toast.error('Failed to record interaction');
    }
  };

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
            <SearchIcon className="text-primary-500" />
            Search Movies
          </h1>
          <p className="text-gray-400">Find movies by title, genre, year, or rating</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8"
        >
          <SearchBar onSearch={handleSearch} genres={genres} />
        </motion.div>

        {loading && (
          <div className="flex justify-center py-20">
            <LoadingSpinner />
          </div>
        )}

        {!loading && searched && movies.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <Film size={64} className="mx-auto text-gray-600 mb-4" />
            <p className="text-gray-400 text-lg">No movies found. Try different search criteria.</p>
          </motion.div>
        )}

        {!loading && movies.length > 0 && (
          <>
            <p className="text-sm text-gray-500 mb-4">{pagination.total} results found</p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <AnimatePresence>
                {movies.map((movie, index) => (
                  <MovieCard
                    key={movie.movieId || movie._id}
                    movie={movie}
                    onInteraction={handleInteraction}
                    index={index}
                  />
                ))}
              </AnimatePresence>
            </div>

            {pagination.pages > 1 && (
              <div className="mt-8 flex justify-center gap-2">
                {Array.from({ length: pagination.pages }, (_, i) => (
                  <button
                    key={i}
                    onClick={() => handleSearch({ page: i + 1 })}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      pagination.page === i + 1
                        ? 'bg-primary-600 text-white'
                        : 'bg-dark-800 text-gray-400 hover:bg-dark-700'
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            )}
          </>
        )}

        {!searched && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <SearchIcon size={64} className="mx-auto text-gray-700 mb-4" />
            <p className="text-gray-500 text-lg">Search for movies above or browse by genre</p>
            <div className="flex flex-wrap justify-center gap-2 mt-6">
              {genres.slice(0, 8).map(genre => (
                <motion.button
                  key={genre}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handleSearch({ genre })}
                  className="px-4 py-2 bg-dark-800 hover:bg-primary-600/20 text-gray-400 hover:text-primary-400 rounded-full text-sm transition-colors border border-dark-700"
                >
                  {genre}
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Search;
