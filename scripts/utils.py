import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_data_types = df.dtypes
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values',2:'Otype'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns



def detect_outliers(df, numerical_columns):
    outliers_dict = {}
    
    for column in numerical_columns:
        if column in df.columns:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers_z = np.where(z_scores > 3)[0]
            outliers_dict[column] = outliers_z.tolist()
            
            # Box plot method
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[column])
            plt.title(f'Box Plot for {column}')
            plt.show()
            
    return outliers_dict



def get_numerical_columns(df):
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numerical_columns




def fix_outlier(df, col):
    df[col] = np.where(df[col] > df[col].quantile(0.95), df[col].quantile(0.95), df[col])
    df[col] = np.where(df[col] < df[col].quantile(0.05), df[col].quantile(0.05), df[col])
    return df

def remove_outliers(df, column_to_process, z_score_threshold):
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process * '_outlier'
    df[outlier_column] = (np.abs(z_scores) > z_score_threshold).astype(int)
    df = df[df[outlier_column] == 0]

    df = df.drop(columns=[outlier_column], errors='ignore')
    return df
