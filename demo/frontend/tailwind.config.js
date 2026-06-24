/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        party: {
          cdu: "#1A1A1A",
          spd: "#E3000F",
          gruene: "#64A12D",
          fdp: "#FFED00",
          linke: "#BE3075",
          afd: "#009EE0",
        },
        moderator: "#F59E0B",
      },
    },
  },
  plugins: [],
};
