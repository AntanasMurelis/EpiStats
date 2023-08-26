import numpy as np
import pandas as pd
import ast
import re
import warnings
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import Optional, Iterable, Tuple, Union, Literal, Callable, Dict, List



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
    if num_dfs == 1:
        if isinstance(dfs, pd.DataFrame):
            merged_df = dfs
        else:
            merged_df = dfs[0]
    else:
        for i in range(num_dfs - 1):
            assert set(dfs[i].columns) == set(dfs[i + 1].columns), f"The input datasets {dfs[i]['tissue'][0]} and {dfs[i+1]['tissue'][0]} " +\
                "do not have the same columns, hence it's not possible to merge them."

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
    Transform all the loaded features in the expected data type.

    Parameters:
    -----------
    paths_to_df: (Iterable[str])
        A collection of one or more paths to .csv file that store the dataframes.

    Returns:
    --------
    merged_df: (pd.DataFrame)
        The dataframe obtained after merging.

    '''

    # Load and merge
    dataframes = _load_dataframes(paths_to_dfs)
    merged_df = _merge_dataframes(dataframes)
    
    # Features saved as lists are loaded as 'str'. Convert them back to 'list' type
    list_columns = [
        'neighbors', 'neighbors_2D', 'area_2D', 
        'contact_area_distribution', 'num_neighbors_2D', 'slices',
        'neighbors_2D_principal', 'area_2D_principal', 'slices_principal',
        'neighbors_of_neighbors_2D_principal', 
        'num_neighbors_2D_principal'
    ]

    for column in list_columns:
        if column not in merged_df.columns:
            warnings.warn(f"Column {column} not present in the merged dataframe.")
            continue
        merged_df[column] = merged_df[column].apply(lambda x: re.sub(r'(\d)\s', r'\1,', x))
        merged_df[column] = merged_df[column].apply(lambda x: ast.literal_eval(x))

    return merged_df
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
    (pd.Dataframe)
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
def _detect_num_neighbors_outliers(
        df: pd.DataFrame,
        freq_threshold: Optional[float] = 0.025,
) -> List[int]:
    """
    In the input cell statistics dataframe find the number of (3D) neighbors outliers, by excluding less 
    frequent values, separately for each tissue.

    Parameters:
    -----------

    df: (pd.DataFrame)
        The cell statistics dataframe.

    freq_threshold: (Optional[float] = 0.05)
        Compute relative frequency of each number of neighbors. Values under this threshold are marked as outliers.

    Returns:
    --------

    (pd.Dataframe)
        The same dataframe as the one in input, except for additional
        boolean columns `is_small_cell` and `is_large_cell` that mark
        cells with outlying values of volume.
    """

    tissues = df['tissue'].unique()
    df['unfreqent_number_of_neighbors'] = np.zeros(len(df), dtype=bool)
    for tissue in tissues:
        tissue_df = df[df['tissue'] == tissue]

        num_neighbors = tissue_df['num_neighbors'].tolist()

        values, counts = np.unique(num_neighbors, return_counts=True)
        rel_frequencies = counts / len(num_neighbors)
        unfreq_values = values[rel_frequencies < freq_threshold]

        outliers_mask = np.logical_and(
            df['tissue'] == tissue, np.isin(tissue_df['num_neighbors'], unfreq_values)
        )
        df['unfreqent_number_of_neighbors'] = np.logical_or(
            df['unfreqent_number_of_neighbors'], outliers_mask
        )

    return df
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def detect_outliers(
    df: pd.DataFrame,
    methods: Optional[Iterable[Literal['volume', 'num_neighbors']]] = ('volume', 'num_neighbors'),
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
    
        methods: (Optional[Iterable[Literal['volume', 'num_neighbors']]], default=('volume', 'num_neighbors'))
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

    ### HANDLE ARGS ###

    is_outlier = np.zeros(len(df))

    if 'volume' in methods:
        volume_outliers_df = _detect_volume_outliers(df=df, *args, **kwargs)
        is_outlier_vol = np.logical_or(
            volume_outliers_df['is_small_cell'], 
            volume_outliers_df['is_large_cell']
        )
        is_outlier = np.logical_or(is_outlier, is_outlier_vol)
    elif 'num_neighbors' in methods:
        num_neigh_outliers_df = num_neigh_outliers_df(df=df, *args, **kwargs)
        is_outlier_num_neigh = num_neigh_outliers_df['unfreq_num_neighbors']
        is_outlier = np.logical_or(is_outlier, is_outlier_num_neigh)
    else:
        NotImplementedError()
    
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
    Also remove outlying cell indexes from neighbors lists.
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

    if 'neighbors' in df.columns:
        # Remove outlying indexes from neighbors lists
        tissues = df['tissue'].unique()
        for tissue in tissues:
            tissue_df = df[df['tissue'] == tissue]
            out_idxs = tissue_df[tissue_df['is_outlier']]['cell_ID'].values
            for idx, row in tissue_df.iterrows():
                if row['exclude_cell']:
                    continue
                else:
                    # 3D neighbors
                    mask = np.isin(np.asarray(row['neighbors']), out_idxs)
                    # rm_idxs = np.where(mask)[0]
                    tissue_df.at[idx, 'neighbors'] = list(np.asarray(row['neighbors'])[~mask])
                    # for idx in rm_idxs:
                    #     del tissue_df.loc[idx, 'neighbors'][idx]
                    tissue_df.at[idx, 'num_neighbors'] -= np.sum(mask)
                    # 2D neighbors
                    new_neighs, new_num_neighs = [], []
                    for neighs, num_neighs in zip(row['neighbors_2D'], row['num_neighbors_2D']):
                        mask = np.isin(np.asarray(neighs), out_idxs)
                        # rm_idxs = np.where(mask)[0]
                        new_neighs.append(list(np.asarray(neighs)[~mask]))
                        # for idx in rm_idxs:
                        #     del neighs[idx]
                        # new_neighs.append(neighs)
                        new_num_neighs.append(num_neighs - np.sum(mask))
                    tissue_df.at[idx, 'neighbors_2D'] = new_neighs
                    tissue_df.at[idx, 'num_neighbors_2D'] = new_num_neighs
            
            df[df['tissue'] == tissue] = tissue_df

    # Remove outlying records
    out_df = df[~df['is_outlier']].copy()

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
    id_features = ['cell_ID', 'tissue', 'tissue_type', 'exclude_cell', 'is_outlier']
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



#------------------------------------------------------------------------------------------------------------
def _get_lewis_law_2D_stats(
    df: pd.DataFrame,
    num_neighbors_lower_threshold: Optional[int] = 3,
    principal_axis: Optional[bool] = True,
    freq_threshold: Optional[int] = 0.025
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    '''
    Compute the statistics needed for checking 2D Lewis' Law.

    Parameters:
    -----------

    df: (pd.DataFrame)
        The dataframe to compute statistics from.
    
    num_neighbors_lower_threshold: (Optional[int], default=3)
        The threshold under which a cell is excluded from computation.
    
    principal_axis (Optional[bool], default=True)
        If True compute the lewis law statistics for the 2D statistics 
        collected along cells' principal_axes.

    freq_threshold: (Optional[float] = 0.05)
        Values number of neighbors whose frequency is below this threshold
        are excluded from computation.

    Returns:
    --------
    
    lewis_law_dict: (Dict[str, Dict[int, Tuple[float, float]]])
        A nested dictionary. The statistics computed for different tissues are
        stored in different dictionaries, each associated to the correspondent tissue names key.
        Each inner dictionary then has number of neighbors values as keys and tuples 
        of normalized averages and standard errors as values.
    '''

    if principal_axis:
        col_num_neighbors = 'num_neighbors_2D_principal'
        col_area = 'area_2D_principal'
    else:
        col_num_neighbors = 'num_neighbors_2D'
        col_area = 'area_2D'

    tissues = df['tissue'].unique()

    # Gather cell 2D areas by tissue and by number of neighbors
    stats_2D_dict = {}
    for tissue in tissues:
        tissue_df = df[df['tissue'] == tissue]
        tissue_2D_dict = defaultdict(list)
        for _, row in tissue_df.iterrows():
            if row['exclude_cell']:
                continue
            else:
                assert len(row[col_num_neighbors]) == len(row[col_area]), 'Bug in the 2D stats code!'
                for area, num_neigh in zip(row[col_area], row[col_num_neighbors]):
                    if num_neigh < num_neighbors_lower_threshold:
                        continue
                    else:
                        tissue_2D_dict[num_neigh].append(area)

        # Reject unfrequent number of neighbors (if threshold is > 0)
        if freq_threshold > 0.0:
            num_neigh_vals = np.asarray(tissue_2D_dict.keys())
            freqs = np.zeros(len(tissue_2D_dict.keys()))
            tot = 0
            for i, num_neigh in enumerate(num_neigh_vals):
                freqs[i] = len(num_neigh)
                tot += freqs[i]
            rel_freqs = freqs / tot
            to_remove = num_neigh_vals[rel_freqs < freq_threshold]
            for key in to_remove:
                del tissue_2D_dict[key]
            print(f"Tissue {tissue} -> rejected data for number of neighbors {to_remove} due to low occurrency.")

        stats_2D_dict[tissue] = tissue_2D_dict
    
    # Compute area averages
    lewis_law_dict = {}
    for tissue, tissue_dict in stats_2D_dict.items():
        total_avg = np.sum([np.mean(areas)*len(areas) for areas in tissue_dict.values()])
        total_avg = total_avg / np.sum([len(areas) for areas in tissue_dict.values()])
        local_avgs = {num: np.mean(areas)/total_avg for num, areas in tissue_dict.items()}
        local_errs = {num: np.std(areas/total_avg) for num, areas in tissue_dict.items()}
        lewis_law_dict[tissue] = {num: (avg, err) for num, avg, err in zip(local_avgs.keys(), local_avgs.values(), local_errs.values())}
    
    return lewis_law_dict

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _get_aboav_law_2D_stats(
    df: pd.DataFrame,
    num_neighbors_lower_threshold: Optional[int] = 3,
    principal_axis: Optional[bool] = True,
    freq_threshold: Optional[int] = 0.025,
    show_logs: Optional[bool] = False
    ) -> Dict[str, Dict[int, float]]:
    '''
    Parameters:
    -----------

    df: (pd.DataFrame)
        The dataframe to compute statistics from.
    
    num_neighbors_lower_threshold: (Optional[int], default=3)
        The threshold under which a cell is excluded from computation.

    principal_axis (Optional[bool], default=True)
        If True compute the lewis law statistics for the 2D statistics 
        collected along cells' principal_axes.

    freq_threshold: (Optional[float] = 0.05)
        Values number of neighbors whose frequency is below this threshold
        are excluded from computation.
    
    show_logs: (Optional[bool], deafult=False)
        If true messages are print for debugging purpose.

    Returns:
    --------
    
    aboav_law_dict: (Dict[str, Dict[int, Tuple[float, float]]])
            A nested dictionary. The statistics computed for different tissues are stored 
            in different dictionaries, each associated to the correspondent tissue names key.
            Each inner dictionary then has number of neighbors values as keys and tuples 
            of normalized averages and standard errors as values.
    '''

    tissues = df['tissue'].unique()

    aboav_law_dict = {}
    for tissue in tissues:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'TISSUE: {tissue}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        tissue_df = df[df['tissue'] == tissue]
        tissue_dict = defaultdict(list)
        if principal_axis:
            for _, row in tqdm(tissue_df.iterrows(), total=len(tissue_df)):
                curr_dict = row['neighbors_of_neighbors_2D_principal']
                for num, neighs in curr_dict.items():
                    tissue_dict[num] = tissue_dict[num] + neighs
            tissue_dict = {
                num : [item for l in lst for item in l] 
                for num, lst in tissue_dict.items()
            }
        else:
            for _, row in tqdm(tissue_df.iterrows(), total=len(tissue_df)):
                if row['exclude_cell']:
                    continue
                else:
                    if show_logs:
                        print('---------------------------------------')
                        print(f"cell_ID: {row['cell_ID']}")
                    for neighs, num_neighs, slc in zip(row['neighbors_2D'], row['num_neighbors_2D'], row['slices']):
                        if show_logs:
                            print(f'Curr slice: {slc}, neighbors: {neighs}, num neighbors: {num_neighs}')
                        if num_neighs < num_neighbors_lower_threshold:
                            continue
                        else: 
                            others_num_neighs = []
                            for neigh in neighs:
                                neigh_row = tissue_df[tissue_df['cell_ID'] == neigh]
                                if neigh_row['exclude_cell'].bool(): #no complete neighborhood!
                                    break
                                slice_idx = neigh_row['slices'].item().index(slc)
                                others_num_neighs.append(neigh_row['num_neighbors_2D'].item()[slice_idx])
                                if show_logs:    
                                    print(f"Current other: {neigh}, slice: {slc}, num neighbors: {neigh_row['num_neighbors_2D'].item()[slice_idx]}")

                            if len(others_num_neighs) == num_neighs: 
                                tissue_dict[num_neighs] = tissue_dict[num_neighs] + others_num_neighs
                                if show_logs:
                                    print(f'Current tissue_dict: {tissue_dict}')
        
        # Reject unfrequent number of neighbors (if threshold is > 0)
        if freq_threshold > 0.0:
            num_neigh_vals = np.asarray(tissue_dict.keys())
            freqs = np.zeros(len(tissue_dict.keys()))
            tot = 0
            for i, num_neigh in enumerate(num_neigh_vals):
                freqs[i] = len(num_neigh)
                tot += freqs[i]
            rel_freqs = freqs / tot
            to_remove = num_neigh_vals[rel_freqs < freq_threshold]
            for key in to_remove:
                del tissue_dict[key]
            print(f"Tissue {tissue} -> rejected data for number of neighbors {to_remove} due to low occurrency.")

        tissue_dict = dict(sorted(tissue_dict.items()))
        if show_logs:
            print(tissue_dict)
        aboav_law_dict[tissue] = {k: (np.mean(v), np.std(v)/np.sqrt(len(v))) for k, v in tissue_dict.items()}

    return aboav_law_dict
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _get_area_CV(
    df: pd.DataFrame,
    num_neighbors_lower_threshold: Optional[int] = 3,
    principal_axis: Optional[bool] = True
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    '''
    Compute the statistics needed for checking 2D Lewis' Law.

    Parameters:
    -----------
        df: (pd.DataFrame)
            The dataframe to compute statistics from.
        
        num_neighbors_lower_threshold: (Optional[int], default=3)
            The threshold under which a cell is excluded from computation.
        
        principal_axis (Optional[bool], default=True)
            If True compute the lewis law statistics for the 2D statistics 
            collected along cells' principal_axes.

    Returns:
    --------
        area_cv_dict: (Dict[str, Dict[int, float]])
            A nested dictionary. The statistics computed for different tissues are
            stored in different dictionaries, each associated to the correspondent tissue names key.
            Each inner dictionary then has number of neighbors values as keys and coefficients of
            variation of area as values.
    '''

    if principal_axis:
        col_num_neighbors = 'num_neighbors_2D_principal'
        col_area = 'area_2D_principal'
    else:
        col_num_neighbors = 'num_neighbors_2D'
        col_area = 'area_2D'

    tissues = df['tissue'].unique()

    # Gather cell 2D areas by tissue and by number of neighbors
    stats_2D_dict = {}
    for tissue in tissues:
        tissue_df = df[df['tissue'] == tissue]
        tissue_2D_dict = defaultdict(list)
        for _, row in tissue_df.iterrows():
            if row['exclude_cell']:
                continue
            else:
                assert len(row[col_num_neighbors]) == len(row[col_area]), 'Bug in the 2D stats code!'
                for area, num_neigh in zip(row[col_area], row[col_num_neighbors]):
                    if num_neigh < num_neighbors_lower_threshold:
                        continue
                    else:
                        tissue_2D_dict[num_neigh].append(area)
        stats_2D_dict[tissue] = tissue_2D_dict
    
    # Compute area CVs
    area_cv_dict = {}
    for tissue, tissue_dict in stats_2D_dict.items():
        area_cv_dict[tissue] = {num: np.std(areas)/np.mean(areas) for num, areas in tissue_dict.items()}
    
    return area_cv_dict
#------------------------------------------------------------------------------------------------------------