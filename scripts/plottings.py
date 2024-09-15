import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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




def analyze_monthly_changes(df, postal_code_col, month_col, premium_col, claims_col):
    # Step 1: Sort data by PostalCode and TransactionMonth
    df = df.sort_values(by=[postal_code_col, month_col])
    
    # Step 2: Calculate monthly changes in TotalPremium and TotalClaims
    df['MonthlyChange_TotalPremium'] = df.groupby(postal_code_col)[premium_col].diff()
    df['MonthlyChange_TotalClaims'] = df.groupby(postal_code_col)[claims_col].diff()

    # Step 3: Drop rows where changes are NaN (first month in each group)
    df = df.dropna(subset=['MonthlyChange_TotalPremium', 'MonthlyChange_TotalClaims'])

    # Step 4: Scatter plot of monthly changes with PostalCode color-coded
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='MonthlyChange_TotalPremium', y='MonthlyChange_TotalClaims', hue=postal_code_col, palette='viridis', alpha=0.7)
    plt.title('Monthly Changes in Total Premium vs Total Claims by PostalCode')
    plt.xlabel('Monthly Change in Total Premium')
    plt.ylabel('Monthly Change in Total Claims')
    plt.show()

    # Step 5: Calculate correlation between monthly changes for each PostalCode
    correlations = []
    postal_codes = df[postal_code_col].unique()
    
    for code in postal_codes:
        temp_df = df[df[postal_code_col] == code]
        corr = temp_df[['MonthlyChange_TotalPremium', 'MonthlyChange_TotalClaims']].corr().iloc[0, 1]
        correlations.append((code, corr))
    
    # Create a DataFrame with PostalCode and the correlation values
    correlation_df = pd.DataFrame(correlations, columns=[postal_code_col, 'Correlation'])
    
    return correlation_df




def bivariate_analysis(df, x_col, y_col, show_regression_line=False):
    # Check if columns exist in DataFrame
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in DataFrame.")
    
    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5, label='Data points')
    
    # Regression Line (optional)
    if show_regression_line:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color='red')
    
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
    
    # Correlation Coefficient
    correlation = df[x_col].corr(df[y_col])
    print(f"Correlation between {x_col} and {y_col}: {correlation:.2f}")