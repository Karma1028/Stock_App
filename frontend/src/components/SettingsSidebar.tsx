import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Settings, Sliders } from "lucide-react";
import { useSettingsStore } from "@/store/settingsStore";
import { Input } from "@/components/ui/input";

const AI_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "alibaba/tongyi-deepresearch-30b-a3b:free",
    "arcee-ai/trinity-mini:free",
    "qwen/qwen3-4b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "xiaomi/mimo-v2-flash:free",
    "google/gemma-3-27b-it:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "tngtech/deepseek-r1t-chimera:free",
    "z-ai/glm-4.5-air:free",
    "amazon/nova-2-lite-v1:free",
    "openai/gpt-oss-20b:free"
];

export function SettingsSidebar() {
    const {
        aiModel, setAiModel,
        apiKey, setApiKey,
        chartType, setChartType,
        showSMA50, toggleSMA50,
        showSMA200, toggleSMA200,
        showBollingerBands, toggleBollingerBands
    } = useSettingsStore();

    return (
        <Sheet>
            <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
                    <Settings className="w-5 h-5" />
                </Button>
            </SheetTrigger>
            <SheetContent className="w-[400px] sm:w-[540px] overflow-y-auto">
                <SheetHeader>
                    <SheetTitle className="flex items-center gap-2">
                        <Sliders className="w-5 h-5" /> Global Settings
                    </SheetTitle>
                    <SheetDescription>
                        Configure AI models, charts, and application preferences.
                    </SheetDescription>
                </SheetHeader>

                <div className="grid gap-8 py-8">
                    {/* AI Settings */}
                    <div className="space-y-4">
                        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Artificial Intelligence</h3>

                        <div className="space-y-2">
                            <Label>AI Model</Label>
                            <Select value={aiModel} onValueChange={setAiModel}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Select Model" />
                                </SelectTrigger>
                                <SelectContent>
                                    {AI_MODELS.map(model => (
                                        <SelectItem key={model} value={model}>
                                            {model.split(':')[0]}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                            <p className="text-xs text-muted-foreground">Select the LLM used for summaries and planning.</p>
                        </div>

                        <div className="space-y-2">
                            <Label>OpenRouter API Key (Optional)</Label>
                            <Input
                                type="password"
                                placeholder="sk-or-..."
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                            />
                            <p className="text-xs text-muted-foreground">Overrides the system default key if provided.</p>
                        </div>
                    </div>

                    {/* Chart Settings */}
                    <div className="space-y-4">
                        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Chart Preferences</h3>

                        <div className="flex items-center justify-between">
                            <Label htmlFor="chart-type">Chart Style</Label>
                            <Select value={chartType} onValueChange={(v: any) => setChartType(v)}>
                                <SelectTrigger className="w-[180px]">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="area">Area Chart</SelectItem>
                                    <SelectItem value="candlestick">Candlestick</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label>SMA 50</Label>
                                <p className="text-xs text-muted-foreground">50-day Simple Moving Average</p>
                            </div>
                            <Switch checked={showSMA50} onCheckedChange={toggleSMA50} />
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label>SMA 200</Label>
                                <p className="text-xs text-muted-foreground">200-day Simple Moving Average</p>
                            </div>
                            <Switch checked={showSMA200} onCheckedChange={toggleSMA200} />
                        </div>

                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label>Bollinger Bands</Label>
                                <p className="text-xs text-muted-foreground">Volatility indicators</p>
                            </div>
                            <Switch checked={showBollingerBands} onCheckedChange={toggleBollingerBands} />
                        </div>
                    </div>
                </div>
            </SheetContent>
        </Sheet>
    );
}
