import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useEffect } from 'react';
import { useAuthStore } from './context/authStore';
import { useThemeStore } from './context/themeStore';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Stats from './pages/Stats';
import History from './pages/History';
import Profile from './pages/Profile';
import Search from './pages/Search';
import Watchlist from './pages/Watchlist';
import Analytics from './pages/Analytics';
import Social from './pages/Social';
import MovieDetail from './pages/MovieDetail';
import SessionReplay from './pages/SessionReplay';

function App() {
  const { isAuthenticated } = useAuthStore();
  const { initTheme } = useThemeStore();

  useEffect(() => {
    initTheme();
  }, [initTheme]);

  return (
    <Router>
      <div className="App">
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 3000,
            style: {
              background: '#1f2937',
              color: '#fff',
              border: '1px solid #374151',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#fff',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />

        <Routes>
          <Route
            path="/login"
            element={isAuthenticated ? <Navigate to="/dashboard" /> : <Login />}
          />
          <Route
            path="/dashboard"
            element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" />}
          />
          <Route
            path="/stats"
            element={isAuthenticated ? <Stats /> : <Navigate to="/login" />}
          />
          <Route
            path="/history"
            element={isAuthenticated ? <History /> : <Navigate to="/login" />}
          />
          <Route
            path="/profile"
            element={isAuthenticated ? <Profile /> : <Navigate to="/login" />}
          />
          <Route
            path="/search"
            element={isAuthenticated ? <Search /> : <Navigate to="/login" />}
          />
          <Route
            path="/watchlist"
            element={isAuthenticated ? <Watchlist /> : <Navigate to="/login" />}
          />
          <Route
            path="/analytics"
            element={isAuthenticated ? <Analytics /> : <Navigate to="/login" />}
          />
          <Route
            path="/social"
            element={isAuthenticated ? <Social /> : <Navigate to="/login" />}
          />
          <Route
            path="/movie/:id"
            element={isAuthenticated ? <MovieDetail /> : <Navigate to="/login" />}
          />
          <Route
            path="/sessions"
            element={isAuthenticated ? <SessionReplay /> : <Navigate to="/login" />}
          />
          <Route
            path="/"
            element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
