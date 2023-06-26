import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    paths_to_dfs: Iterable[str]
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

    dataframes = _load_dataframes(paths_to_dfs)
    return _merge_dataframes(dataframes)

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def rename_features(
    df: pd.DataFrame,
    old_names: Iterable[str],
    new_names: Iterable[str],
) -> pd.DataFrame:
    '''
    Rename columns of the input dataframe.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The cell statistics dataframe.

        old_names: (Iterable[str])
            An iterable storing the column names to change.
        
        new_names: (Iterable[str])
            An iterable storing the new names to give to the columns.

    Returns:
    --------
        out_df: (pd.Dataframe)
            A copy of the input dataframe with modified names.
    '''

    out_df = df.copy()

    for old_name, new_name in zip(old_names, new_names):
        assert old_name in df.columns, f'Column {old_name} not in the dataframe.'
        out_df[new_name] = out_df[old_name]
    
    out_df = out_df.drop(columns=old_names)
    return out_df

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _detect_volume_outliers(
    df: pd.DataFrame,
    quantile_level: Optional[float] = 0.025,
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
        
        lower_volume_mask = np.logical_and((df['tissue'] == tissue), (df["volume"] < lower_bound))
        df['is_small_cell'] = np.logical_or(df['is_small_cell'], lower_volume_mask)
        # print(f'Lower bound: {lower_bound}')
        # print(f'Lower mask: \n {lower_volume_mask}')
        upper_volume_mask = np.logical_and((df['tissue'] == tissue), (df["volume"] > upper_bound))
        df['is_large_cell'] = np.logical_or(df['is_large_cell'], upper_volume_mask)
        # print(f'Upper bound: {upper_bound}')
        # print(f'Upper mask: \n {upper_volume_mask}')

        num_upper_outliers = np.sum(upper_volume_mask)
        num_lower_outliers = np.sum(lower_volume_mask) 

        print(
        f'''\
Found a total of {num_lower_outliers + num_upper_outliers} volume outliers in {tissue} sample,
of which:
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
            If False, return a copy. Otherwise, do operation in place.
    
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
def _exclude_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Return a copy of the input dataframe without the records marked as outliers.
    (Mainly used when computing summary statistics and plotting).

    Parameters:
    -----------
        df: (pd.DataFrame)
            The input dataframe. It must have an `is_outlier` boolean column,
            which is `True` if the correspondent record should be removed.

    Returns:
    --------
        out_df: (pd.DataFrame)
            The input dataframe without outliers.  
    '''

    out_df = df[~df['is_outlier']]
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
    id_features = ['cell_ID', 'tissue', 'tissue_type', 'is_outlier']
    keep_features = id_features + numeric_features
    drop_features = [feat for feat in df.columns if feat not in keep_features]

    numeric_df = df.copy()
    numeric_df = df.drop(columns=drop_features) 

    #drop NAs
    if remove_na:
        len_before = len(numeric_df)
        numeric_df = numeric_df.dropna()
        len_after = len(numeric_df)
        print(f'Dropped {len_before-len_after} records containing NAs.')

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



#------------------------------------------------------------------------------------------------------------
def apply_PCA(
    df: pd.DataFrame,
    numeric_features: Optional[Iterable[str]] = [
        'volume', 'num_neighbors', 'elongation',
        'isoperimetric_ratio', 'area'
    ],
    n_comps: Optional[int] = 2,
    standardize_data: Optional[bool] = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Return the principal components and their loading computed on
    the chosen numerical features of the input dataframe.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The dataframe to compute principal components on.

        numeric_features: (Optional[Iterable[str]])
            A list of numerical features to be extracted from the input dataframe.
        
        n_comps: (Optional[int], default=2)
            The number of principal components to extract from the data.
        
        standardize_data: (Optional[bool], deafult=True)
            If true standardize the input data before applying PCA.
     
    Returns:
    --------
        pca_data: (np.ndarray)
            An array of size `len(df) * n_components`, containing the 
            projections on the principal components of each record.

        pca_loadings: (np.ndarray)
            An array of size `len(numeric_features) * n_components`, in which each 
            columns is the set of weights associated to each principal component.

        explained_var: (np.ndarray)
            The variance explained (ratio) by each principal component.
    '''

    if standardize_data:
        df = standardize(df, numeric_features)

    pca = PCA(n_components=n_comps)
    pca_fit = pca.fit(df[numeric_features])

    pca_data =  pca_fit.transform(df[numeric_features])
    pca_loadings = np.array([pca_fit.components_[i] for i in range(n_comps)])
    explained_var = pca_fit.explained_variance_ratio_

    return pca_data, pca_loadings, explained_var

#------------------------------------------------------------------------------------------------------------

