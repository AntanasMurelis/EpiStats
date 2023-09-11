import os
import sys
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Set

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.append(PROJECT_ROOT)
from statistics_collection.scripts.StatsAnalytics import prepare_df


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



#---------------------------------------------------------------------------------------------------------------------
def get_cell_neighbors(
        path_to_stats_df: str,
        cell_ids: List[int]
) -> List[List[int]]:
    """
    Load the dataframe of cell statistics and, for each cell in cell_ids, extract its neighbors' ids list.

    Parameters:
    -----------
    path_to_stats_df: (str)
        The path to the statistics dataframe.
    
    cell_ids: (List[int])
        A list of cell ids.

    Returns:
    --------
    cell_neighbors_lst: (List[List[int]])
        A list of neighbors' ids for each cell.
    """

    cell_stats_df = prepare_df([path_to_stats_df])
    id_mask = np.isin(cell_stats_df["cell_ID"], cell_ids)
    cell_neighbors_lst = cell_stats_df[id_mask]["neighbors"].tolist()

    return cell_neighbors_lst
#---------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    
    ### SET PATHS ###
    ROOT_DIR = ""
    path_to_stats_df = os.path.join(
        ROOT_DIR, "/relative/path/to/stats/df.csv"
    )
    path_to_meshes_source_dir = os.path.join(
        ROOT_DIR, "/relative/path/to/cell/meshes/source/dir/"
    )
    path_to_meshes_dest_dir = os.path.join(
        ROOT_DIR, "/relative/path/to/cell/meshes/destination/dir/"
    )
    ##################

    valid_cell_ids = get_valid_cell_ids(path_to_stats_df)
    copy_selected_meshes(path_to_meshes_source_dir, path_to_meshes_dest_dir, valid_cell_ids) 

