import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

def scrapeData(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    return df

def handleNullvalues(df):
    before = len(df)
    # Drop rows missing values
    df = df.dropna()

    removed = before - len(df)
    print(f"Removed {removed} rows with missing data")
    print(f"Remaining: {len(df)}")  

# Ensure the plotting style is clean and modern
sns.set_theme(style="whitegrid")

def price_movingAvg(df):
    plt.figure(figsize=(14, 7))

    # Plot the raw closing price and rolling averages
    plt.plot(df.index, df['Close'], label='TSLA Close Price', color='royalblue', alpha=0.6, linewidth=1.5)
    plt.plot(df.index, df['Close'].rolling(window=50).mean(), label='50-day SMA', color='darkorange', linewidth=2)
    plt.plot(df.index, df['Close'].rolling(window=200).mean(), label='200-day SMA', color='crimson', linewidth=2)

    # Title and labels
    plt.title('Tesla ($TSLA) Macro-Trend Analysis (2015 - 2026)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.show()

def daily_return(df):
    mean = df['Returns'].mean()
    std = df['Returns'].std()
    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    plt.plot(x, stats.norm.pdf(x, mean, std), color='darkgray', linestyle='--', linewidth=2, label='Theoretical Normal Dist')

    # Annotate Skewness and Kurtosis onto the chart window
    textstr = f"Skewness: {df['Returns'].skew():.2f}\nKurtosis: {df['Returns'].kurt():.2f}"
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Title and labels
    plt.title('Distribution of TSLA Daily Returns vs. Normal Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Daily Percentage Return', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-0.15, 0.15) # Zoom in slightly on the core distribution to focus on the tails
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.show()