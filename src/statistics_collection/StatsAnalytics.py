import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import napari
from skimage.io import imread
from scipy import stats
from copy import deepcopy
from typing import Optional, Iterable, Tuple



#------------------------------------------------------------------------------------------------------------
def _load_dataframes(
    files: Iterable[str]
) -> Tuple[pd.DataFrame]:
    '''
    Load dataframes from .csv files.

    Parameters:
    -----------
    files: (Iterable[str])
        A collection of paths to .csv file that store the dataframes.

    Returns:
    --------
    dfs: (Tuple[pd.DataFrame])
        The dataframes loaded from file.

    '''

    dfs = ()
    for file in files:
        df = pd.read_csv(file, index_col=0)
        dfs = dfs + df 
    
    return dfs

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _merge_dataframes(
    dfs: Iterable[pd.DataFrame]
) -> pd.DataFrame:
    '''
    Merge previously loaded dataframes.

    Parameters:
    -----------
    dfs: (Iterable[pd.DataFrame])
        A collection of dataframes.
    
    Returns:
    --------
    merged_df: (pd.DataFrame)
        The dataframe obtained from the merge.
    '''

    num_dfs = len(dfs)
    
    if num_dfs == 0:
        raise ValueError("You must pass at least one DataFrame.")
    elif num_dfs == 1:
        if isinstance(dfs, pd.DataFrame):
            merged_df = dfs
        else:
            merged_df = dfs[0]
    else:
        merged_df = pd.concat(objs=dfs, axis=0, ignore_index=True)
    
    return merged_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def get_numerical_df(
    df: pd.DataFrame,
    numeric_features: Iterable[str] = [
        'volume', 'num_neighbors', 'elongation',
        'isoperimetric_ratio', 'surface_area'
    ],
    remove_na: bool = True
) -> pd.DataFrame:
    
    #keep only id features plus numerical ones
    id_features = ['cell_ID', 'tissue', 'tissue_type']
    keep_features = id_features + numeric_features
    drop_features = [feat for feat in df.columns if feat not in keep_features]

    numeric_df = df.copy()
    numeric_df = df.drop(columns=drop_features) 

    #drop NAs
    len_before = len(numeric_df)
    numeric_df = numeric_df.dropna()
    len_after = len(numeric_df)
    print(f'Dropped {len_after-len_before} records containing NAs.')

    return numeric_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def standardize_numerical_df(
    numeric_df: pd.DataFrame,
    numeric_features: Iterable[str]
) -> pd.DataFrame:
    
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_df[numeric_features].values)
    scaled_df = numeric_df.copy()
    scaled_df[numeric_features] = scaled_values
    
    return scaled_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def mark_volume_outliers(
    df: pd.DataFrame,
    quantile_level: Optional[float] = 0.05
) -> pd.DataFrame:
    '''
    Mark cells in the dataset with volume under a certain
    quantile of the volume empirical distribution adding
    a boolean column to the cell dataframe.

    Parameters:
    -----------
    df: (pd.DataFrame)
        The cell statistics dataframe.

    quantile_level: (float)
        Cells below this quantile_level are marked as small, cells
        above are marked as large.

    Returns:
    --------
    out_df: (pd.Dataframe)
        The same dataframe as the one in input, except for additional
        boolean columns `is_small_cell` and `is_large_cell` that mark
        cells with outlying values of volume.

    '''

    tissues = df['tissue'].unique()
    df['small_cell'] = np.zeros(len(df), dtype=bool)
    df['large_cell'] = np.zeros(len(df), dtype=bool)
    for tissue in tissues:
        tissue_df = df[df['tissue'] == tissue]
        lower_threshold = np.quantile(tissue_df["volume"][~np.isnan(df['volume'])], quantile_level)
        upper_threshold = np.quantile(tissue_df["volume"][~np.isnan(df['volume'])], 1-quantile_level)
        df['is_small_cell'] = np.logical_or(
            df['is_small_cell'],
            (df['tissue'] == tissue) * (df["volume"] < lower_threshold)
        )
        df['is_large_cell'] = np.logical_or(
            df['is_large_cell'],
            (df['tissue'] == tissue) * (df["volume"] > upper_threshold)
        )

    return df
        
#------------------------------------------------------------------------------------------------------------
    