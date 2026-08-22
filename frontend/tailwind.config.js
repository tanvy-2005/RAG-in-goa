/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Quicksand', 'sans-serif'],
      },
      colors: {
        forest: {
          900: '#06140e',
          800: '#072718',
          700: '#0d4027',
          600: '#0b2419',
          500: '#0f5132',
          400: '#144731',
          300: '#106941',
          200: '#15803d',
        },
      },
    },
  },
  plugins: [],
}
