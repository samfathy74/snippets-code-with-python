# 1.Data collection and see Overall
# 2.Handle Missing values
# 3.Handle Redundantly (Repeating)
# 4.Handle Outliers
# 5.Handle Skewness
# 6.Handle Imbalance Classes in Data
# 7.Detected highly correlation
# 8.Detected symbols
# 9.Make Normalizes for dataset
# 10.Encoding Categories columns
# 11.Feature selection

import pandas as pd
def read_dataset(file_path, column_map=None, print_info=False, null_info=None):
    # Read the dataset into a pandas DataFrame
    dataset = pd.read_csv(file_path)
    
    # Rename columns if column_map is provided {'Co1':'new_col'}
    if column_map is not None:
        dataset = dataset.rename(columns=column_map)
    
    # Print dataset information if print_info is True
    if print_info:
        print(dataset.info())
    
    # Calculate the sum or mean of all non-null values if agg_func is provided
    if null_info == 'sum':
        result = dataset.isna().sum()
        print(result)
    elif null_info == 'mean':
        result = dataset.isna().mean()
        print(result)
    
    return dataset
# ==============================================================================================================================
import pandas as pd
from scipy import stats

def handle_missing_values(df, fill_type='mean', apply_changes=False, columns=None):
    # Make a copy of the original dataframe
    df_copy = df.copy()
    
    # Fill missing values based on the specified fill type
    if fill_type == 'mean':
        fill_values = df_copy.mean()
    elif fill_type == 'mode':
        fill_values = df_copy.mode().iloc[0]
    elif fill_type == 'median':
        fill_values = df_copy.median()
    
    if columns is None:
        # Fill missing values for all columns
        df_copy.fillna(fill_values, inplace=True)
    else:
        # Fill missing values for specified columns
        for col in columns:
            if col in df_copy.columns:
                df_copy[col].fillna(fill_values[col], inplace=True)
    
    # Check if any changes were made
    changes_made = not df_copy.equals(df)
    
    # Apply changes to the original dataframe if specified
    if apply_changes:
        df = df_copy
    
    return df, changes_made
# ==============================================================================================================================
import pandas as pd
import numpy as np

def remove_redundant_features(df, threshold=0.95, apply_changes=False, columns=None):
    # Make a copy of the original dataframe
    df_copy = df.copy()
    
    # Create a correlation matrix
    corr_matrix = df_copy.corr().abs()
    
    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    if columns is None:
        # Check all columns for redundancy
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    else:
        # Check specified columns for redundancy
        to_drop = [column for column in columns if column in upper.columns and any(upper[column] > threshold)]
    
    # Drop the redundant features
    df_copy = df_copy.drop(to_drop, axis=1)
    
    # Apply changes to the original dataframe if specified
    if apply_changes:
        df = df_copy
    
    return df
# ======================================================================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_outliers(df, columns, figsize=(10, 6), column_names=None, showfliers=True):
    # Set the figure size
    plt.figure(figsize=figsize)
    
    # Create a boxplot of the specified columns
    sns.boxplot(data=df[columns], showfliers=showfliers)
    
    # Set the column names if specified
    if column_names is not None:
        plt.xticks(range(len(columns)), column_names)
    
    # Show the plot
    plt.show()
# ======================================================
import pandas as pd
from scipy import stats

def remove_outliers(df, threshold=3):
    # Calculate the Z-scores for each value in the dataset
    z_scores = stats.zscore(df)
    
    # Identify the rows and columns where the absolute Z-score is greater than the threshold
    rows, cols = np.where(np.abs(z_scores) > threshold)
    
    # Remove the rows containing outliers
    df = df.drop(rows)
    
    return df
# ============== another method =======================
import pandas as pd

def remove_outliers_iqr(df, apply_changes=False, column=None):
    # Make a copy of the original dataframe
    df_copy = df.copy()
    
    if column is None:
        # Calculate the IQR for each column in the dataset
        Q1 = df_copy.quantile(0.25)
        Q3 = df_copy.quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify rows where any value is an outlier
        rows = ((df_copy < (Q1 - 1.5 * IQR)) | (df_copy > (Q3 + 1.5 * IQR))).any(axis=1)
    else:
        # Calculate the IQR for the specified column
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify rows where the value in the specified column is an outlier
        rows = ((df_copy[column] < (Q1 - 1.5 * IQR)) | (df_copy[column] > (Q3 + 1.5 * IQR)))
    
    # Remove rows containing outliers
    df_copy = df_copy[~rows]
    
    # Apply changes to the original dataframe if specified
    if apply_changes:
        df = df_copy
    
    return df
# ================================================================================================================================
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def plot_skewness(dataset, column, method='log'):
    # Create a copy of the dataset to avoid modifying the original data
    data = dataset.copy()
    
    # Apply transformation to specified column
    if method == 'log':
        data[column] = np.log1p(data[column])
    elif method == 'sqrt':
        data[column] = np.sqrt(data[column])
    elif method == 'cbrt':
        data[column] = np.cbrt(data[column])
    elif method == 'reciprocal':
        data[column] = 1 / data[column]
    elif method == 'boxcox':
        data[column], _ = stats.boxcox(data[column])
    elif method == 'exp':
        data[column] = np.exp(data[column])
    
    # Plot original and transformed data
    fig, ax = plt.subplots(1, 2)
    sns.histplot(dataset[column], kde=True, ax=ax[0])
    ax[0].set_title('Original Data')
    sns.histplot(data[column], kde=True, ax=ax[1])
    ax[1].set_title(f'{method.capitalize()} Transformed Data')
    plt.show()
# ==============================================
import numpy as np
from scipy import stats

def handle_skewness(dataset, columns=None, modify=False, method='log'):
    if columns is None:
        # Handle skewness for all columns
        columns = dataset.columns
    if modify:
        # Apply transformation to specified columns
        for column in columns:
            if method == 'log':
                dataset[column] = np.log1p(dataset[column])
            elif method == 'sqrt':
                dataset[column] = np.sqrt(dataset[column])
            elif method == 'cbrt':
                dataset[column] = np.cbrt(dataset[column])
            elif method == 'reciprocal':
                dataset[column] = 1 / dataset[column]
            elif method == 'boxcox':
                dataset[column], _ = stats.boxcox(dataset[column])
            elif method == 'exp':
                dataset[column] = np.exp(dataset[column])
    return dataset
# ========================================================================================================================
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

def handle_imbalance(dataset, target_column, modify=False, strategy='undersample', plot=False):
    if plot:
        # Plot original data
        fig, ax = plt.subplots(1, 2)
        sns.countplot(x=target_column, data=dataset, ax=ax[0])
        ax[0].set_title('Original Data')
    
    if modify:
        X = dataset.drop(target_column, axis=1)
        y = dataset[target_column]
        
        if strategy == 'undersample':
            # Undersample the majority class
            undersampler = RandomUnderSampler()
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        elif strategy == 'oversample':
            # Oversample the minority class
            X_resampled, y_resampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0])
            X_resampled = np.concatenate((X[y == 0], X_resampled))
            y_resampled = np.concatenate((y[y == 0], y_resampled))
        elif strategy == 'smote':
            # Generate synthetic samples using SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Update the dataset with the resampled data
        dataset = pd.concat([X_resampled, y_resampled], axis=1)
    
    if plot:
        # Plot resampled data
        sns.countplot(x=target_column, data=dataset, ax=ax[1])
        ax[1].set_title(f'{strategy.capitalize()} Resampled Data')
        plt.show()
    
    return dataset
# =========================================================================================================================
import seaborn as sns

def plot_correlation_matrix(dataset):
    # Calculate the correlation matrix
    corr_matrix = dataset.corr()
    
    # Plot the correlation matrix as a heatmap
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
# =============================================
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def handle_correlation(dataset, threshold=0.9, modify=False, strategy='pca'):
    if modify:
        # Calculate the correlation matrix
        corr_matrix = dataset.corr().abs()
        
        # Select the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        
        # Find index of feature columns with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if strategy == 'pca':
            # Apply PCA to the dataset
            pca = PCA()
            dataset = pca.fit_transform(dataset)
        elif strategy == 'variance_threshold':
            # Apply variance threshold to the dataset
            variance_threshold = VarianceThreshold(threshold)
            dataset = variance_threshold.fit_transform(dataset)
        elif strategy == 'remove':
            # Remove highly correlated features from the dataset
            dataset = dataset.drop(dataset.columns[to_drop], axis=1)
    return dataset
# ==========================================================================================================================
import re

def remove_symbols(dataset, columns, symbols, strategy='translate'):
    symbols = '!@#$%^&*()_-+=[]{}|\:;"<>,.?/~`'
    if strategy == 'translate':
        # Create a translation table to remove the specified symbols
        trans = str.maketrans('', '', symbols)
        
        # Remove the symbols from the specified columns
        for column in columns:
            dataset[column] = dataset[column].str.translate(trans)
    elif strategy == 'regex':
        # Create a regular expression pattern to match the specified symbols
        pattern = '[' + re.escape(symbols) + ']'
        
        # Remove the symbols from the specified columns
        for column in columns:
            dataset[column] = dataset[column].str.replace(pattern, '', regex=True)
    
    return dataset
# ======================================================================================================================================
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# k is a parameter that specifies the number of top features to select
def apply_feature_selection(X, y, k=2, apply_fs=True, strategy='f_classif'):
    if apply_fs:
        if strategy in ['f_classif', 'mutual_info_classif']:
            # Choose the scoring function based on the specified strategy
            if strategy == 'f_classif':
                score_func = f_classif
            elif strategy == 'mutual_info_classif':
                score_func = mutual_info_classif

            # Apply feature selection using the filter method with the chosen scoring function
            selector = SelectKBest(score_func, k=k)
            X_new = selector.fit_transform(X, y)
        elif strategy == 'rfe':
            # Apply feature selection using the RFE method with a linear SVC estimator
            estimator = SVC(kernel='linear')
            selector = RFE(estimator, n_features_to_select=k)
            X_new = selector.fit_transform(X, y)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        return X_new
    else:
        # Do not apply feature selection
        return X
# ====================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt

def plot_columns(df):
    # Create bar plots for categorical columns
    nums_cols = df.select_dtypes(include='number')
    cats_cols = df.select_dtypes(exclude='number')
 
    print("* Categories Columns:",list(cats_cols.columns))
    print("* Numericals Columns:",list(nums_cols.columns))
 
    plt.bar(['Categorical', 'Numerical'], [len(cats_cols.columns), len(nums_cols.columns)], color=['orange', 'purple'], width=0.4, label=['Categorical', 'Numerical'])
    plt.xticks(rotation=45)
    plt.legend()
    plt.title('Number of Categorical and Numerical Columns')
    plt.show()

    return nums_cols, cats_cols
# ================================================
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(df, columns, apply_enc=True, strategy='one_hot'):
    if apply_enc:
        df_encoded = df.copy()
        for col in columns:
            if strategy == 'one_hot':
                # Apply one-hot encoding to the column
                encoder = OneHotEncoder(sparse=False)
                encoded_cols = encoder.fit_transform(df[[col]])
                col_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), pd.DataFrame(encoded_cols, columns=col_names)], axis=1)
            elif strategy == 'label':
                # Apply label encoding to the column
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col])
            else:
                raise ValueError(f"Invalid strategy: {strategy}")
        return df_encoded
    else:
        # Do not apply encoding
        return df
# =====================================================================================================================
# HINT For some regression problems, normalizing the target variable can improve the performance of the model. 
# For classification problems, normalizing the target variable is not necessary.
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def normalize_data(df, apply_norm=True, strategy='min_max'):
    if apply_norm:
        df_normalized = df.copy()
        if strategy == 'min_max':
            # Apply min-max normalization to the data
            scaler = MinMaxScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)
        elif strategy == 'standard':
            # Apply standard normalization to the data
            scaler = StandardScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)
        elif strategy == 'robust':
            # Apply robust normalization to the data
            scaler = RobustScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        return df_normalized
    else:
        # Do not apply normalization
        return df
# ========================================================================================
