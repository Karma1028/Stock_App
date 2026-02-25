import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_meta_filter_chart():
    # 1. Create Synthetic "Crash" Data (Jan 15, 2025)
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=40, freq='D')
    
    # Simulating a crash: Start 100, drop to 85, chop, then recover
    price = [100]
    for i in range(1, 15): price.append(price[-1] * (0.98 + np.random.normal(0, 0.005))) # Crash
    for i in range(15, 25): price.append(price[-1] * (1.00 + np.random.normal(0, 0.01))) # Chop
    for i in range(25, 40): price.append(price[-1] * (1.01 + np.random.normal(0, 0.005))) # Recovery
    
    df = pd.DataFrame({'Close': price}, index=dates)
    
    # 2. Define Signals (Raw Buys vs Filtered)
    # Raw Buys: Trying to catch the knife on the way down
    raw_buy_dates = [dates[5], dates[8], dates[12], dates[14], dates[28]]
    raw_buy_prices = [df.loc[d, 'Close'] for d in raw_buy_dates]
    
    # Meta Filter Decisions
    # X = Vetoed (Volatility too high)
    vetoed_dates = [dates[5], dates[8], dates[12], dates[14]]
    vetoed_prices = [df.loc[d, 'Close'] for d in vetoed_dates]
    
    # O = Approved (Recovery confirmed)
    approved_dates = [dates[28]]
    approved_prices = [df.loc[d, 'Close'] for d in approved_dates]

    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], color='gray', alpha=0.6, label='Price (TMCV)')
    
    # Raw Signals (Red Dots)
    plt.scatter(raw_buy_dates, raw_buy_prices, color='red', s=100, label='Raw Buy Signal (RSI < 30)', zorder=5)
    
    # Vetoes (Black X)
    plt.scatter(vetoed_dates, vetoed_prices, color='black', marker='x', s=150, linewidth=3, label='Meta-Filter VETO (Vol > 90th %ile)', zorder=6)
    
    # Approved (Green Arrow)
    plt.scatter(approved_dates, approved_prices, color='green', marker='^', s=200, label='Meta-Filter APPROVED', zorder=7)
    
    # Formatting
    plt.title('Figure 14.3: Meta-Labeling forensic Analysis (Jan 2025)\n"The Sniper Filter"', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save to report/pdf_figures (same as FIG in generate_final_report.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_path = os.path.join(project_dir, 'report', 'pdf_figures', 'Figure_14_3_MetaFilter.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    generate_meta_filter_chart()
