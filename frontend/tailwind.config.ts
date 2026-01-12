import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "var(--background)",
                foreground: "var(--foreground)",
                "grow-green": {
                    DEFAULT: "#00b386",
                    light: "#00d6a0",
                    dark: "#008f6b"
                },
                "grow-dark": "#0f111a",
                "grow-card": "rgba(30, 41, 59, 0.7)", // Glassmorphic base
                "grow-gray": "#94a3b8",
                "grow-surface": "#1e293b",
            },
            backgroundImage: {
                "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
                "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
            },
        },
    },
    plugins: [],
};
export default config;
