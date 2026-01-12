"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion } from "framer-motion"
import { LayoutDashboard, LineChart, PieChart, Newspaper, Settings, User } from "lucide-react"
import { cn } from "@/utils/cn"

const navItems = [
    { name: "Dashboard", href: "/", icon: LayoutDashboard },
    { name: "Stock Analysis", href: "/stock-analysis", icon: LineChart },
    { name: "Portfolio", href: "/portfolio", icon: PieChart },
    { name: "Market News", href: "/news", icon: Newspaper },
]

export function Sidebar() {
    const pathname = usePathname()

    return (
        <aside className="fixed left-0 top-0 h-screen w-64 border-r border-white/5 bg-grow-dark/95 backdrop-blur-xl z-50">
            <div className="flex h-16 items-center px-6 border-b border-white/5">
                <div className="text-2xl font-bold bg-gradient-to-r from-grow-green to-teal-400 bg-clip-text text-transparent">
                    StockPro
                </div>
            </div>

            <nav className="flex-1 space-y-2 p-4">
                {navItems.map((item) => {
                    const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
                    return (
                        <Link key={item.href} href={item.href}>
                            <div
                                className={cn(
                                    "relative flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 group",
                                    isActive
                                        ? "text-white bg-white/5 shadow-lg shadow-grow-green/5"
                                        : "text-gray-400 hover:text-white hover:bg-white/5"
                                )}
                            >
                                {isActive && (
                                    <motion.div
                                        layoutId="activeNav"
                                        className="absolute left-0 w-1 h-6 bg-grow-green rounded-r-full"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                    />
                                )}
                                <item.icon className={cn("w-5 h-5", isActive ? "text-grow-green" : "group-hover:text-grow-green")} />
                                <span className="font-medium">{item.name}</span>
                            </div>
                        </Link>
                    )
                })}
            </nav>

            <div className="p-4 border-t border-white/5 space-y-2">
                <button className="flex w-full items-center space-x-3 px-4 py-3 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all">
                    <Settings className="w-5 h-5" />
                    <span className="font-medium">Settings</span>
                </button>
                <button className="flex w-full items-center space-x-3 px-4 py-3 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all">
                    <User className="w-5 h-5" />
                    <span className="font-medium">Profile</span>
                </button>
            </div>
        </aside>
    )
}
