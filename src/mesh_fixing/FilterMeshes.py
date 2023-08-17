import os
import sys
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Set

sys.path.append('/nas/groups/iber/Users/Federico_Carrara/Statistics_Collection/EpiStats/src/statistics_collection/')
from StatsAnalytics import prepare_df


#---------------------------------------------------------------------------------------------------------------------
def get_valid_cell_ids(
        path_to_stats_df: str
) -> Set[int]:
    """
    Load the dataframe of collected statistics and extract the ids of valid cells (i.e., not to be excluded).

    Parameters:
    -----------
    path_to_stats_df: (str)
        The path to the statistics dataframe.

    Returns:
    --------
    valid_cell_idxs: (List[int])
        A list of valid cell ids.
    """

    cell_stats_df = prepare_df([path_to_stats_df])
    valid_row_idxs = np.nonzero(~cell_stats_df['exclude_cell'])[0]
    valid_cell_idxs = np.asarray(cell_stats_df['cell_ID'])[valid_row_idxs]

    return valid_cell_idxs
#---------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------------
def copy_selected_meshes(
        source_dir: str, 
        dest_dir: str, 
        selected_ids: List[int]
) -> None:
    """
    Copy a collection of selected meshes from a source directory to a destination directory.

    Parameters:
    -----------
    source_dir: (str)
        The source directory.

    dest_dir: (str)
        The destination directory. 

    selected_ids: (Set[int])
        A list of ids associated to the mesh to select.
    """

    # Create the source directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get the list of files in the source directory
    source_files = os.listdir(source_dir)

    # Iterate through the files
    for file in tqdm(source_files, desc='Moving mesh files'):
        # Get the cell ID from the file name in the form of cell_{id}.stl
        cell_id = int(file.split('_')[1].split('.')[0])

        # If the cell ID is in the filtered_cell_ids list, copy the file to the output directory
        if cell_id in selected_ids:
            shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))
#---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    ### SET PATHS ###
    common_root_dir = "/nas/groups/iber/Users/Federico_Carrara/"
    path_to_stats_df = os.path.join(
        common_root_dir, "Statistics_Collection/outputs/outputs_v4/output_lung_pseudostratified_from_harold_s_10_e_6_d_8/cell_stats/stats_dataset_lung.csv"
    )
    path_to_meshes_source_dir = os.path.join(
        common_root_dir, "Statistics_Collection/outputs/outputs_v4/output_lung_pseudostratified_from_harold_s_10_e_6_d_8/cell_meshes/"
    )
    path_to_meshes_dest_dir = os.path.join(
        common_root_dir, "FoldingNet_project/data/CellsData/cell_meshes/lung"
    )
    ##################

    valid_cell_ids = get_valid_cell_ids(path_to_stats_df)
    copy_selected_meshes(path_to_meshes_source_dir, path_to_meshes_dest_dir, valid_cell_ids) 

