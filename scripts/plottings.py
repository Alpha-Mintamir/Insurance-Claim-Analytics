import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, columns, log_scale_threshold=1000):
    num_cols = len(columns)
    plt.figure(figsize=(15, num_cols * 4))

    for i, col in enumerate(columns):
        plt.subplot(num_cols, 1, i + 1)
        
        # Check if the column contains large values
        if df[col].max() > log_scale_threshold:
            sns.histplot(np.log1p(df[col]), kde=True, bins=30)
            plt.title(f'Log Distribution of {col}', fontsize=16)
            plt.xlabel(f'Log of {col}', fontsize=12)
        else:
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}', fontsize=16)
            plt.xlabel(col, fontsize=12)
        
        plt.ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_bar_charts(df, columns):
    num_cols = len(columns)
    plt.figure(figsize=(15, num_cols * 4))

    for i, col in enumerate(columns):
        plt.subplot(num_cols, 1, i + 1)
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Count of Categories in {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    
    plt.tight_layout()
    plt.show()