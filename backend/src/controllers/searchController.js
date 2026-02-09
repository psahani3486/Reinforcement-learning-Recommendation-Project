import Movie from '../models/Movie.js';

export const searchMovies = async (req, res) => {
  try {
    const { q, genre, year, minRating, maxRating, sortBy = 'popularity', order = 'desc', limit = 20, page = 1 } = req.query;
    const skip = (page - 1) * limit;
    const filter = {};

    if (q) {
      filter.$text = { $search: q };
    }

    if (genre) {
      const genres = genre.split(',');
      filter.genres = { $in: genres };
    }

    if (year) {
      filter.year = parseInt(year);
    }

    if (minRating || maxRating) {
      filter.averageRating = {};
      if (minRating) filter.averageRating.$gte = parseFloat(minRating);
      if (maxRating) filter.averageRating.$lte = parseFloat(maxRating);
    }

    const sortOptions = {};
    sortOptions[sortBy] = order === 'asc' ? 1 : -1;

    const movies = await Movie.find(filter)
      .sort(sortOptions)
      .skip(skip)
      .limit(parseInt(limit));

    const total = await Movie.countDocuments(filter);

    // Get all distinct genres for filter UI
    const allGenres = await Movie.distinct('genres');
    const yearRange = await Movie.aggregate([
      { $group: { _id: null, minYear: { $min: '$year' }, maxYear: { $max: '$year' } } }
    ]);

    res.json({
      movies,
      pagination: { total, page: parseInt(page), pages: Math.ceil(total / limit) },
      filters: {
        genres: allGenres.sort(),
        yearRange: yearRange[0] || { minYear: 1970, maxYear: 2025 }
      }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getGenres = async (req, res) => {
  try {
    const genres = await Movie.distinct('genres');
    res.json({ genres: genres.sort() });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
