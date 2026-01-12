import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "StockPro | Advanced Analytics",
  description: "Next-gen stock analysis platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="flex min-h-screen bg-background text-foreground">
          <Sidebar />
          <main className="flex-1 ml-64 min-h-screen relative overflow-hidden">

            {/* Background Gradients */}
            <div className="fixed top-0 left-0 w-full h-full pointer-events-none z-0">
              <div className="absolute top-[-20%] right-[-10%] w-[500px] h-[500px] bg-grow-green/5 rounded-full blur-3xl opacity-50" />
              <div className="absolute bottom-[-20%] left-[10%] w-[400px] h-[400px] bg-blue-500/5 rounded-full blur-3xl opacity-30" />
            </div>

            <div className="relative z-10 p-8">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
