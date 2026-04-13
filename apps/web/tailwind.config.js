/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f4ff",
          100: "#dce6fe",
          600: "#3b5bdb",
          700: "#2f4bc9",
        },
      },
    },
  },
  plugins: [],
};
