import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
from skimage.io import imread
from scipy import stats
from copy import deepcopy
from typing import Optional, Iterable, Tuple, Union, Literal, Callable



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
    dfs: (List[pd.DataFrame])
        The dataframes loaded from file.

    '''

    dfs = []
    for file in files:
        df = pd.read_csv(file, index_col=0)
        dfs.append(df) 
    
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
        The dataframe obtained after merging.
    '''

    num_dfs = len(dfs)
    
    if num_dfs == 0:
        raise ValueError("You must pass at least one DataFrame object.")
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
def prepare_df(
    paths_to_df: Iterable[str]
) -> pd.DataFrame:
    '''
    Load dataframes relative to different samples from .csv files and 
    merge them in a single dataframe using previously defined functions.

    Parameters:
    -----------
    paths_to_df: (Iterable[str])
        A collection of one or more paths to .csv file that store the dataframes.

    Returns:
    --------
    merged_df: (pd.DataFrame)
        The dataframe obtained after merging.

    '''

    dataframes = _load_dataframes(paths_to_df)
    return _merge_dataframes(dataframes)

#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def _detect_volume_outliers(
    df: pd.DataFrame,
    quantile_level: Optional[float] = 0.05,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
) -> pd.DataFrame:
    '''
    Detect cells in the dataset with outlying volume.
    The outlyingness can be checked with a quantile approach, or by
    setting manually lower and upper bounds.
    Deafult is the quantile approach.

    Parameters:
    -----------
    df: (pd.DataFrame)
        The cell statistics dataframe.

    quantile_level: (Optional[float])
        Cells below this quantile_level are marked as small, cells
        above are marked as large.

    lower_bound: (Optional[float])
        Cells below this bound are marked as small cells.

    upper_bound: (Optional[float])
        Cells above this bound are marked as large cells.

    Returns:
    --------
    out_df: (pd.Dataframe)
        The same dataframe as the one in input, except for additional
        boolean columns `is_small_cell` and `is_large_cell` that mark
        cells with outlying values of volume.

    '''

    tissues = df['tissue'].unique()
    df['is_small_cell'] = np.zeros(len(df), dtype=bool)
    df['is_large_cell'] = np.zeros(len(df), dtype=bool)
    for tissue in tissues:
        tissue_df = df[df['tissue'] == tissue]
        
        if quantile_level:
            lower_bound = np.quantile(tissue_df["volume"][~np.isnan(df['volume'])], quantile_level)
            upper_bound = np.quantile(tissue_df["volume"][~np.isnan(df['volume'])], 1-quantile_level)
        
        lower_volume_mask = df["volume"] < lower_bound
        df['is_small_cell'] = np.logical_or(
            df['is_small_cell'],
            (df['tissue'] == tissue) * (lower_volume_mask)
        )
        upper_volume_mask = df["volume"] > upper_bound
        df['is_large_cell'] = np.logical_or(
            df['is_large_cell'],
            (df['tissue'] == tissue) * (upper_volume_mask)
        )

        num_upper_outliers = np.sum(upper_volume_mask)
        num_lower_outliers = np.sum(lower_volume_mask) 
        print(
        f'''\
Found a total of {num_lower_outliers + num_upper_outliers} volume outliers,
of which
    - Below lower bound: {num_lower_outliers},
    - Above upper bound: {num_upper_outliers}. 
        '''
        )

    return df
        
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def detect_outliers(
    df: pd.DataFrame,
    methods: Optional[Iterable[Literal['volume']]] = ('volume'),
    inplace: Optional[bool] = False,
    *args, 
    **kwargs,
) -> Union[pd.DataFrame, None]:
    '''
    Apply the outliers detection functions decided by the user.
    The default is volume outliers search.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The cell statistics dataframe to apply outliers detection to.
    
        methods: (Optional[Iterable[str]], default='volume')
            An iterable storing the names of methods used for outlier detection.
        
        inplace: (Optional[bool], default=False)
            If False, return a copy. Otherwise, do operation inplace.
    
        *args, **kwargs: 
            The parameters required by the currently selected methods.
    
    Returns:
    --------
        df: (Union[pd.Dataframe, None])
            The input dataframe with an additional boolean column `is_outlier`,
            which is set to `True` for outlying records.

    '''
    is_outlier = np.zeros(len(df))

    if 'volume' in methods:
        volume_outliers_df = _detect_volume_outliers(df=df, *args, **kwargs)
        is_outlier_vol = np.logical_or(
            volume_outliers_df['is_small_cell'], 
            volume_outliers_df['is_large_cell']
        )
    else:
        raise NotImplementedError()
    
    is_outlier = np.logical_or(is_outlier, is_outlier_vol)

    if inplace:
        df['is_outlier'] = is_outlier
        return 
    else:
        out_df = df.copy()
        out_df['is_outlier'] = is_outlier
        return out_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def extract_numerical(
    df: pd.DataFrame,
    numeric_features: Optional[Iterable[str]] = [
        'volume', 'num_neighbors', 'elongation',
        'isoperimetric_ratio', 'area'
    ],
    remove_na: Optional[bool] = True,
) -> pd.DataFrame:
    '''
    Extract a copy of the input dataframe only containing numerical features.
    Optionally remove also NA's from it, since numerical features are 
    intended to be used for plotting and analysis purposes.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The dataframe to extract numerical features from.

        numeric_features: (Optional[Iterable[str]])
            A list of numerical features to be extracted from the input dataframe.
        
        remove_na (Optional[bool])
            If `True`, remove all the NA's from the numerical dataframe.

    Returns:
    --------
        numeric_df: (pd.DataFrame)
            A dataframe containing only cell and tissue ids, plus the 
            chosen numerical feature.

    '''
    #keep only id features plus numerical ones
    id_features = ['cell_ID', 'tissue', 'tissue_type']
    keep_features = id_features + numeric_features
    drop_features = [feat for feat in df.columns if feat not in keep_features]

    numeric_df = df.copy()
    numeric_df = df.drop(columns=drop_features) 

    #drop NAs
    if remove_na:
        len_before = len(numeric_df)
        numeric_df = numeric_df.dropna()
        len_after = len(numeric_df)
        print(f'Dropped {len_after-len_before} records containing NAs.')

    return numeric_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def standardize(
    df: pd.DataFrame,
    numeric_features: Optional[Iterable[str]] = [
        'volume', 'num_neighbors', 'elongation',
        'isoperimetric_ratio', 'area'
    ],
    scaler: Optional[Callable] = None
) -> pd.DataFrame:
    '''
    Standardize the numerical features of the input dataframe.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The dataframe whose numerical features needs standardization.

        numeric_features: (Optional[Iterable[str]])
            A list of numerical features to be extracted from the input dataframe.
        
        scaler: (Optional[Callable])
            A callable object that performs standardization of numerical features.
            It is required that `scaler` supports fit_transform method, otherwise 
            an exception is returned.
            If scaler is set to `None`, `sklearn.preprocessing.StandardScaler` is 
            used by deafult.

    Returns:
    --------
        scaled_df: (pd.DataFrame)
            The input dataframe with standardized numerical features.

    '''
    
    if scaler ==  None:
        scaler = StandardScaler()

    try:
        scaled_values = scaler.fit_transform(df[numeric_features].values)
    except:
        raise ValueError(f'Chosen scaler `{scaler}` does not support fit_transform method.')
    scaled_df = df.copy()
    scaled_df[numeric_features] = scaled_values
    
    return scaled_df

#------------------------------------------------------------------------------------------------------------

