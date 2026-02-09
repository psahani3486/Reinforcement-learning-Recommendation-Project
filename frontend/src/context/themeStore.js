import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useThemeStore = create(
  persist(
    (set) => ({
      isDark: true,

      toggleTheme: () => {
        set((state) => {
          const newIsDark = !state.isDark;
          if (newIsDark) {
            document.documentElement.classList.add('dark');
            document.documentElement.classList.remove('light');
          } else {
            document.documentElement.classList.add('light');
            document.documentElement.classList.remove('dark');
          }
          return { isDark: newIsDark };
        });
      },

      initTheme: () => {
        set((state) => {
          if (state.isDark) {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.add('light');
          }
          return state;
        });
      },
    }),
    {
      name: 'theme-storage',
    }
  )
);
