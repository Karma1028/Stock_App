import { Card } from "@/components/ui/card";

export default function Loading() {
    return (
        <div className="space-y-8 animate-pulse">
            <div className="h-10 w-48 bg-white/5 rounded-lg mb-8" />

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                {[1, 2, 3, 4].map(i => (
                    <Card key={i} className="h-32 bg-white/5 border-white/5" />
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <Card className="h-[300px] bg-white/5 border-white/5" />
                <Card className="h-[300px] bg-white/5 border-white/5" />
            </div>
        </div>
    )
}
