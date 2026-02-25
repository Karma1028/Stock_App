import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

PROCESSED_DIR = '../data/processed'

def generate_technical_charts(file_path):
    print(f"Loading data from: {os.path.abspath(file_path)}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Ensure directory exists (relative to script execution might mean checking full path)
    if not os.path.exists(PROCESSED_DIR):
        # Fallback if running from a different root
        if os.path.exists('data/processed'):
             PROCESSED_DIR_LOCAL = 'data/processed'
        else:
             PROCESSED_DIR_LOCAL = PROCESSED_DIR
             os.makedirs(PROCESSED_DIR_LOCAL, exist_ok=True)
    else:
        PROCESSED_DIR_LOCAL = PROCESSED_DIR

    # Recalculate necessary columns for visualization if they don't exist
    if 'Price_Change' not in df.columns:
        df['Price_Change'] = df['Close'].diff()
        df['Gain'] = df['Price_Change'].apply(lambda x: x if x > 0 else 0)
        df['Loss'] = df['Price_Change'].apply(lambda x: abs(x) if x < 0 else 0)
        period = 14
        df['Avg_Gain'] = df['Gain'].ewm(alpha=1/period, min_periods=period).mean()
        df['Avg_Loss'] = df['Loss'].ewm(alpha=1/period, min_periods=period).mean()

    # --- Chart 1: Price Change ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(df.index, df['Close'], color='black', linewidth=1.2, label='Close Price')
    ax.set_title('Tata Motors — Price + Daily Change', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (₹)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ['#2ECC71' if x >= 0 else '#E74C3C' for x in df['Price_Change']]
    ax.bar(df.index, df['Price_Change'], color=colors, width=1.0)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Daily Change (₹)')
    ax.set_title('Daily Price Change (Green = Up, Red = Down)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PROCESSED_DIR_LOCAL, 'price_change_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # --- Chart 2: Gains vs Losses ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(df.index, df['Close'], color='black', linewidth=1.2)
    ax.set_title('Tata Motors — Price + Gains vs Losses', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (₹)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df.index, df['Gain'], color='#2ECC71', linewidth=1, label='Gain (Up Days)', alpha=0.7)
    ax.plot(df.index, df['Loss'], color='#E74C3C', linewidth=1, label='Loss (Down Days)', alpha=0.7)
    ax.fill_between(df.index, 0, df['Gain'], color='#2ECC71', alpha=0.1)
    ax.fill_between(df.index, 0, df['Loss'], color='#E74C3C', alpha=0.1)

    ax.set_ylabel('Magnitude (₹)')
    ax.set_title('Magnitude of Daily Gains vs Losses')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PROCESSED_DIR_LOCAL, 'gains_losses_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # --- Chart 3: Smoothed Averages ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(df.index, df['Close'], color='black', linewidth=1.2)
    ax.set_title('Tata Motors — Price + Smoothed Averages (For RSI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (₹)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df.index, df['Avg_Gain'], color='#27AE60', linewidth=1.5, label='Avg Gain (14-day smoothed)')
    ax.plot(df.index, df['Avg_Loss'], color='#C0392B', linewidth=1.5, label='Avg Loss (14-day smoothed)')

    ax.fill_between(df.index, df['Avg_Gain'], df['Avg_Loss'], 
                    where=(df['Avg_Loss'] > df['Avg_Gain']), 
                    color='#E74C3C', alpha=0.1, label='Bearish Momentum')

    ax.fill_between(df.index, df['Avg_Gain'], df['Avg_Loss'], 
                    where=(df['Avg_Gain'] > df['Avg_Loss']), 
                    color='#2ECC71', alpha=0.1, label='Bullish Momentum')

    ax.set_ylabel('Smoothed Value')
    ax.set_title('14-Day Smoothed Average Gain vs Loss')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PROCESSED_DIR_LOCAL, 'smoothed_averages_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Use absolute path for reliability
    data_file = r'd:/stock/stock project dnyanesh/stock_app/data/processed/tata_motors_clean.csv'
    generate_technical_charts(data_file)
