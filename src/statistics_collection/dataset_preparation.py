import os
import pandas as pd 
import numpy as np
import json
import scripts.StatsAnalytics as sa
from typing import List, Optional, Tuple

"""
Run this script to postprocess the statistics datasets associated to the different epithelial tissues.
Post-processing consists of:
- Merging datasets,
- Extracting a dataset of only selected numerical features for plotting,
- Computing PCs (and related information) for plotting,
- Extracting statistics from 2D slices for checking Lewis Law and Aboav-Wearie law.

Once the data are polished and extracted, the following files are saved in the output directory:
1. Dataframe containing numerical features + PCs
2. JSON file containing PC coefficients and explained variance
3. JSON file containing data for Lewis law
4. JSON file containing data for Aboav-Wearie law
"""



#--------------------------------------------------------------------------------------------------------------------
def main(
        paths_to_tissue_datasets: List[str],
        numerical_feat: List[str],
        out_dir: str, 
        features_to_rename: List[Tuple[str, str]],
        stats_from_principal_axes: Optional[bool] = True,
        remove_na: Optional[bool] = True,
        detect_outliers: Optional[bool] = True,
        num_2D_neighbors_threshold: Optional[int] = 3, 
        num_principal_comps: Optional[int] = 2,
) -> None:
    """
    Parameters:
    -----------
    paths_to_tissue_datasets: (List[str])
        A collection of one or more paths to .csv file that store the dataframes for each tissue.

    numerical_feat: (List[str])
        A list of numerical features to be extracted from the dataframes.

    out_dir: (str)
        The output directory where post-processed data are stored.

    features_to_rename: (List[Tuple[str, str]])
        A collection of pairs `(old_feature_name, new_feature_name)` for features to be renamed.

    stats_from_principal_axes: (Optional[bool] = True)
        If `True`, the statistics for Lewis/Aboav laws are computed on 2D slices along the principal axes.
    
    remove_na: (Optional[bool] = True)
        If `True`, containing any NA value are removed (usually associated to cells that needs to be filtered out).

    find_outliers: (Optional[bool] = True)
        If `True`, a new column `is_outlier` is added to the resulting dataframe of numerical features.

    num_2D_neighbors_threshold: (Optional[int] = 3)
        When extracting data for Lewis and Aboav law, all the records with a number of neighbors lower than this
        threshold are discarded.

    num_principal_comps: (Optional[int] = 2)
        The number of principal components to consider.

    NOTE: 
    1. Numerical features must be chosen from: 
        [
            'surface_area', 'volume', 'isoperimetric_ratio', 
            'num_neighbors', 'elongation',
            'contact_area_fraction', 'mean_contact_area'
        ]
    """

    # Load and merge the dataframes
    stats_df = sa.prepare_df(paths_to_dfs=paths_to_tissue_datasets)


    # Rename features
    features_to_rename.append(("area", "surface_area"))
    stats_df = sa.rename_features(
        df=stats_df,
        old_names=[names[0] for names in features_to_rename],
        new_names=[names[1] for names in features_to_rename]
    )
    
    # Outlier detection (if necessary)
    if detect_outliers:
        stats_df = sa.find_outliers(df=stats_df)

    # Extract dataframe with only ids and numerical features
    # NOTE: The function also remove NA's
    numeric_stats_df = sa.extract_numerical(
        df=stats_df,
        numeric_features=numerical_feat,
        remove_na=remove_na
    )

    # Compute PCs 
    pcs, loadings, ex_var = sa.apply_PCA(
        df=numeric_stats_df,
        numeric_features=numerical_feat,
        standardize_data=True,
        n_comps=num_principal_comps
    )

    # Add PCs to numerical dataframe
    pca_stats_df = numeric_stats_df.copy()
    pca_stats_df["PC1"] = pcs[:, 0]
    pca_stats_df["PC2"] = pcs[:, 1]

    # Create dictionary to hold additional info regarding PCA
    pca_dict = {
        "features": numerical_feat,
        "PC1_coeffs": list(loadings[0]),
        "PC2_coeffs": list(loadings[1]),
        "explained_variance": list(ex_var)
    }

    # Get a dictionary to hold Lewis Law data
    lewis_dict = sa._get_lewis_law_2D_stats(
        df=stats_df,
        principal_axis=stats_from_principal_axes,
        num_neighbors_lower_threshold=num_2D_neighbors_threshold
    )

    # Get a dictionary to hold the Aboav Law data
    aboav_dict = sa._get_aboav_law_2D_stats(
        df=stats_df,
        principal_axis=stats_from_principal_axes,
        num_neighbors_lower_threshold=num_2D_neighbors_threshold
    )

    # Save all data
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. Numerical + PC scores dataframe
    pca_stats_df.to_csv(os.path.join(out_dir, "numerical_stats_table.csv"))
    # 2. PCA additional info
    with open(os.path.join(out_dir, "pca_additional.json"), "w") as file:
        json.dump(pca_dict, file, indent=2)
    # 3. Lewis law data
    with open(os.path.join(out_dir, "lewis_law_data.json"), "w") as file:
        json.dump(lewis_dict, file, indent=2)
    # 4. Aboav-Weaire law data
    with open(os.path.join(out_dir, "aboav_law_data.json"), "w") as file:
        json.dump(aboav_dict, file, indent=2)

    return
#--------------------------------------------------------------------------------------------------------------------


if __name__ =="__main__":

    out_root_path = '/path/to/stats/collection/outputs'

    ### SPECIFY PATHS TO INPUT CSV TISSUE DATASETS ###
    df_files = [
        'output_tissue_1/cell_stats/stats_tissue_1.csv',
        'output_tissue_1/cell_stats/stats_tissue_2.csv',
        'output_tissue_1/cell_stats/stats_tissue_3.csv'
    ]
    df_paths = [os.path.join(out_root_path, df_file) for df_file in df_files]
    ###################################################

    ### SPECIFY NUMERICAL FEATURES TO CONSIDER ###
    num_features = [
            'surface_area', 'volume', 'isoperimetric_ratio', 
            'num_neighbors', 'elongation',
            'contact_area_fraction', 'mean_contact_area'
    ]
    ##############################################

    ### SPECIFY OUTPUT DIRECTORY ###
    output_dir = os.path.join(out_root_path, "data_for_plotting")
    ################################

    ### SPECIFY FEATURES TO RENAME (Tuples (old_name, new_name)) ###
    rename_feats = [] 
    ################################################################


    ### SPECIFY OTHER PARAMETERS DIRECTLY IN THE FUNCTION CALL ###


    main(
        paths_to_tissue_datasets=df_paths,
        numerical_feat=num_features,
        out_dir=output_dir,
        features_to_rename=rename_feats,
        stats_from_principal_axes=False,
        remove_na=True,
        find_outliers=True,
        num_2D_neighbors_threshold=3,
        num_principal_comps=2
    )



