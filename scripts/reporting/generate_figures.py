import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = "report/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_figures():
    print("Loading data...")
    try:
        equity_df = pd.read_csv("data/EQUITY.csv")
        training_df = pd.read_csv("data/training_data.csv")
        if 'Date' in training_df.columns:
            training_df['Date'] = pd.to_datetime(training_df['Date'])
            training_df.set_index('Date', inplace=True)
    except FileNotFoundError:
        print("Error: content files not found. Ensure 'data/EQUITY.csv' and 'data/training_data.csv' exist.")
        return

    print("Generating Figure 1: Sector Distribution...")
    if 'Sector' in equity_df.columns:
        plt.figure(figsize=(12, 8))
        sector_counts = equity_df['Sector'].value_counts().head(10)
        sns.barplot(x=sector_counts.values, y=sector_counts.index, palette="viridis")
        plt.title("Top 10 Sectors by Company Count")
        plt.xlabel("Number of Companies")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/01_sector_distribution.png", dpi=300)
        plt.close()
    else:
        print("Warning: 'Sector' column not found in EQUITY.csv. Creating placeholder.")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Sector Data Not Available", ha='center', va='center', fontsize=20)
        plt.axis('off')
        plt.savefig(f"{OUTPUT_DIR}/01_sector_distribution.png", dpi=300)
        plt.close()

    print("Generating Figure 2: Price Distribution...")
    if 'Close' in training_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(training_df['Close'], kde=True, bins=50, color='blue')
        plt.title("Distribution of Closing Prices")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/02_price_distribution.png", dpi=300)
        plt.close()
    
    print("Generating Figure 3: Correlation Heatmap...")
    # Select numeric columns relevant for correlation
    numeric_cols = training_df.select_dtypes(include=[np.number]).columns
    # Filter for some key features to make heatmap readable
    key_features = [c for c in numeric_cols if any(x in c for x in ['Close', 'Volume', 'RSI', 'MACD', 'SMA', 'Return'])]
    if len(key_features) > 1:
        corr = training_df[key_features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=300)
        plt.close()
        
    print("Generating Figure 4: Time Series Trend...")
    if 'Close' in training_df.columns:
        # Plot last 1 year of data if available, else full
        recent_df = training_df.last('365D') if len(training_df) > 365 else training_df
        plt.figure(figsize=(14, 7))
        plt.plot(recent_df.index, recent_df['Close'], label='Close Price', linewidth=1)
        if 'SMA_50' in recent_df.columns:
             plt.plot(recent_df.index, recent_df['SMA_50'], label='SMA 50', linestyle='--', alpha=0.7)
        plt.title("Price Trend Analysis (Last Year)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/04_price_trend.png", dpi=300)
        plt.close()

    print("Figures generated successfully.")

if __name__ == "__main__":
    generate_figures()
