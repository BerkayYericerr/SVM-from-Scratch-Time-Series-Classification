import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# create folder for plots
if not os.path.exists('plots'):
    os.makedirs('plots')


def summary_stats(df, col, name):
    print(f"\n--- {name} Summary ---")
    print(df[col].describe())
    print(f"Variance: {df[col].var():.2f}")


def plot_timeseries(df, col, name):
    plt.figure(figsize=(12, 4))
    plt.plot(df['Date'], df[col], linewidth=0.5)
    plt.title(f'{name} over time')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'plots/{name.lower().replace(" ", "_")}_timeseries.png')
    plt.close()


def plot_class_dist(y, name):
    counts = np.bincount(y)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(['Class 0', 'Class 1'], counts)
    plt.title(f'{name} - Class Distribution')
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=['Class 0', 'Class 1'], autopct='%1.1f%%')
    plt.title('Proportions')
    
    plt.tight_layout()
    plt.savefig(f'plots/{name.lower().replace(" ", "_")}_classes.png')
    plt.close()
    
    return counts


def check_imbalance(y, name):
    counts = np.bincount(y)
    ratio = max(counts) / min(counts)
    minority_pct = min(counts) / len(y) * 100
    
    print(f"\n--- {name} Imbalance Check ---")
    print(f"Class 0: {counts[0]} ({counts[0]/len(y)*100:.1f}%)")
    print(f"Class 1: {counts[1]} ({counts[1]/len(y)*100:.1f}%)")
    print(f"Ratio: {ratio:.2f}")
    
    is_imbalanced = minority_pct < 30
    if is_imbalanced:
        print("WARNING: Imbalanced dataset!")
    
    return {'ratio': ratio, 'is_imbalanced': is_imbalanced}


def run_eda(data, save=True):
    name = data.get('dataset_name', 'Dataset')
    col = data['value_col']
    df = data['daily_df']
    X = data['X']
    y = data['y']
    
    print(f"\n{'='*50}")
    print(f"EDA: {name}")
    print(f"{'='*50}")
    
    # summary stats
    summary_stats(df, col, name)
    
    # plots
    if save:
        plot_timeseries(df, col, name)
        plot_class_dist(y, name)
    
    # imbalance
    imb = check_imbalance(y, name)
    
    return imb
    
    print("\nPlots saved to plots/ folder")
