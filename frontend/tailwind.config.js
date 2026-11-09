/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        paper: {
          DEFAULT: '#F9F8F5', // subtle clean off-white lab paper
          dark: '#111417',
        },
        ink: {
          DEFAULT: '#16191D',
          dark: '#F0EFEA',
        },
        muted: {
          DEFAULT: '#5E666E',
          dark: '#9AA1A9',
        },
        rule: {
          DEFAULT: '#E2DFD7',
          dark: '#262B30',
        },
        panel: {
          DEFAULT: '#0F1316',
          dark: '#0A0D0F',
        },
        accent: {
          DEFAULT: '#C97D20', // sonar amber
          hover: '#B56E18',
          light: '#FDF5EC',
          dark: '#DF8C26',
        },
        teal: {
          sonar: '#2F7C73', // known classes
          light: '#EBF5F3',
        }
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', 'Inter', '-apple-system', 'sans-serif'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        serif: ['"Plus Jakarta Sans"', 'Inter', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'Menlo', 'Consolas', 'monospace'],
      },
      letterSpacing: {
        tighter: '-0.035em',
        tight: '-0.02em',
        snug: '-0.01em',
        normal: '0em',
        wide: '0.02em',
      },
      maxWidth: {
        reading: '72ch',
      },
      borderRadius: {
        DEFAULT: '3px',
        sm: '2px',
        md: '4px',
        lg: '6px',
      }
    },
  },
  plugins: [],
};
