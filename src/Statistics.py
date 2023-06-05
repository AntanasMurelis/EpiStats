import numpy as np
import pandas as pd
import trimesh as tm
from scipy import ndimage
from ExtendedTrimesh import ExtendedTrimesh
import os
from Visualisation import generate_required_plots
from tqdm import tqdm
from VoxelProcessing import full_label_processing
import concurrent.futures
from typing import Optional




#------------------------------------------------------------------------------------------------------------
def compute_cell_areas(cell_mesh_lst: list):
    """
    Use the meshes of the cells to compute their areas (in micron squared).
    
    Parameters:
    -----------

    cell_mesh_lst: (list of trimesh object)
        The tirangular meshes of the cell in the standard trimesh format

    Returns
    area_lst: (list of float)
        List of the areas of each cells
    """

    assert cell_mesh_lst.__len__() > 0
    return list(map(lambda x: x.area, cell_mesh_lst))
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def compute_cell_volumes(cell_mesh_lst: list):
    """
    Use the meshes of the cells to compute their volumes (in micron cubes).
    
    Parameters:
    -----------

    cell_mesh_lst: (list of trimesh object)
        The tirangular meshes of the cell in the standard trimesh format

    Returns
    volume_lst: (list of float)
        List of the volumes of each cells
    """

    assert cell_mesh_lst.__len__() > 0
    return list(map(lambda x: abs(x.volume), cell_mesh_lst))
#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
def get_cell_neighbors(labeled_img: np.array, cell_id: int):
    """
    Get all the neighbors of a given cell. Two cells are considered neighborhs if 
    a subset of their surfaces are directly touching.
    
    Parameters:
    -----------

    labeled_img: (np.array, 3D)
        The tirangular meshes of the cell in the standard trimesh format

    cell_id: (int)
        The id of the cell for which we want to find the neighbors

    Returns
    -------
    neighbors_lst: (list of int)
        Return the ids of the neighbors in a list
    """

    #Get the voxels of the cell
    binary_img = labeled_img == cell_id

    #Expand the volume of the cell by 2 voxels in each direction
    expanded_cell_voxels = ndimage.binary_dilation(binary_img, iterations=2)

    #Find the voxels that are directly in contact with the surface of the cell
    cell_surface_voxels = expanded_cell_voxels ^ binary_img

    #Get the labels of the neighbors
    neighbors_lst = np.unique(labeled_img[cell_surface_voxels])

    #Remove the label of the cell itself, and the label of the background from the neighbors list
    neighbors_lst = neighbors_lst[(neighbors_lst != cell_id) & (neighbors_lst != 0)]

    return neighbors_lst.tolist()
#------------------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------------------
def compute_cell_principal_axis_and_elongation(cell_mesh):
    """
    Compute the cell principal axis by computing the eigenvectors of the inertia tensor of the cell shape.
    The major axis is the one corresponding to the smallest eigen value. The elongation is the ratio between
    the length of the major axis and the length of the minor axis.
    
    Parameters:
    -----------

    cell_mesh: (trimesh object)
        The tirangular mesh of the cell in the standard trimesh format

    Returns:
    --------

    major_axis: (np.array, 3D)
        The major axis of the cell

    elongation: (float)
        The ratio of the major axis length to the minor axis length
    """

    #Use the inertia tensor of the cell shape to compute its principal axis
    eigen_values, eigen_vectors = tm.inertia.principal_axis(cell_mesh.moment_inertia)


    #Get the index of the smallest eigen value
    smallest_eigen_value_idx = np.argmin(np.abs(eigen_values))
    greatest_eigen_value_idx = np.argmax(np.abs(eigen_values))

    #Get the corresponding eigen vector
    major_axis = eigen_vectors[smallest_eigen_value_idx]

    elongation = np.sqrt(eigen_values[greatest_eigen_value_idx] / eigen_values[smallest_eigen_value_idx])

    return major_axis, elongation
#------------------------------------------------------------------------------------------------------------





#--------------------------------------------------------------------------------------------------
def get_cell_principal_axis_and_elongation(mesh_lst, cell_id_lst):
    """
    Compute the principal axis and elongation for each cell in the mesh list.
    
    Parameters:
    -----------
        mesh_lst (list):
            List of cell meshes.

        cell_id_lst (list):
            List of cell IDs.

    Returns:
    --------
        cell_principal_axis_lst (list):
            List of principal axes for each cell.

        cell_elongation_lst (list):
            List of elongations for each cell.
    """

    # Initialize lists to store the principal axis and elongation for each cell
    cell_principal_axis_lst = []
    cell_elongation_lst = []

    # Loop over each cell ID
    for cell_id in tqdm(cell_id_lst, desc='Computing cell principal axis and elongation'):
        # Compute the principal axis and elongation for the current cell
        principal_axis, elongation = compute_cell_principal_axis_and_elongation(mesh_lst[cell_id - 1])

        # Store the results in the corresponding lists
        cell_principal_axis_lst.append(principal_axis)
        cell_elongation_lst.append(elongation)

    # Return the lists of principal axes and elongations
    return cell_principal_axis_lst, cell_elongation_lst
#--------------------------------------------------------------------------------------------------





############################################################################################################
# Implementation of the contact area fraction computation -- Old version
# Benefit: more memory efficient
# Drawback: Contact surface area fraction can go beyond 1
############################################################################################################
#------------------------------------------------------------------------------------------------------------
# def compute_cell_contact_area_fraction(cell_mesh_lst, cell_id, cell_neighbors_lst, contact_cutoff):
#     """
#     Compute the fraction of the cell surface area which is in contact with other cells.
    
#     Parameters:
#     -----------
#     cell_mesh_lst: (list, ExtendedTrimesh)
#         List of the cell meshes in the ExtendedTrimesh format
#     cell_id: (int)
#         The id of the cell for which we want to calculate the contact area fraction
#     cell_neighbors_lst: (list, int)
#         List of the ids of the neighbors of the cell
#     contact_cutoff: (float)
#         The cutoff distance in microns for two cells to be considered in contact

#     Returns
#     -------
#         contact_fraction: (float)
#         contact_area_distribution: (list of float)
#         mean_contact_area: (float)
#     """
    
#     if cell_neighbors_lst.__len__() == 0:
#         return 0, [], 0

#     cell_mesh = cell_mesh_lst[cell_id - 1]
#     cell_mesh = ExtendedTrimesh(cell_mesh.vertices, cell_mesh.faces)
    
#     contact_area_total = 0
#     contact_area_distribution = []
#     # Loop over the neighbors of the cell
#     for neighbor_id in cell_neighbors_lst:
#         # Get potential contact faces using the get_potential_contact_faces method from ExtendedTrimesh class
#         neighbour_mesh = ExtendedTrimesh(cell_mesh_lst[neighbor_id - 1].vertices, cell_mesh_lst[neighbor_id - 1].faces)
#         contact_area = cell_mesh.calculate_contact_area(neighbour_mesh, contact_cutoff)
#         contact_area_total += contact_area
#         contact_area_distribution.append(contact_area)

#     # Calculate the fraction of contact area
#     contact_fraction = contact_area_total / cell_mesh.area

#     # Calculate the mean of contact area
#     mean_contact_area = np.mean(contact_area_distribution)

#     return contact_fraction, contact_area_distribution, mean_contact_area





#------------------------------------------------------------------------------------------------------------

############################################################################################################   
# Implementation considers the area fraction on the target cell itself, not negihbors
# Benefit: Maximal contact area fraction is 1
# Drawback: The contact area fraction is not symmetric between the two cells + not as memory efficient
############################################################################################################

def compute_cell_contact_area_fraction(cell_mesh_lst, cell_id, cell_neighbors_lst, contact_cutoff):
    """
    Compute the fraction of the cell surface area which is in contact with other cells.
    
    Parameters:
    -----------
    cell_mesh_lst: (list, ExtendedTrimesh)
        List of the cell meshes in the ExtendedTrimesh format
    cell_id: (int)
        The id of the cell for which we want to calculate the contact area fraction
    cell_neighbors_lst: (list, int)
        List of the ids of the neighbors of the cell
    contact_cutoff: (float)
        The cutoff distance in microns for two cells to be considered in contact

    Returns
    -------
        contact_fraction: (float)
        contact_area_distribution: (list of float)
        mean_contact_area: (float)
    """
    
    if len(cell_neighbors_lst) == 0:
        return 0, [], 0

    cell_mesh = cell_mesh_lst[cell_id - 1]
    cell_mesh = ExtendedTrimesh(cell_mesh.vertices, cell_mesh.faces)
    
    contact_face_indices = set()
    contact_area_distribution = []
    # Loop over the neighbors of the cell
    for neighbor_id in cell_neighbors_lst:
        # Get potential contact faces using the get_potential_contact_faces method from ExtendedTrimesh class
        neighbour_mesh = cell_mesh_lst[neighbor_id - 1]
        neighbour_mesh = ExtendedTrimesh(neighbour_mesh.vertices, neighbour_mesh.faces)
        contact_faces = neighbour_mesh.get_potential_contact_faces(cell_mesh, contact_cutoff)

        # Calculate contact area for the current neighbor and add to distribution
        neighbor_contact_area = np.sum(neighbour_mesh.area_faces[contact_faces])
        contact_area_distribution.append(neighbor_contact_area)
        contact_face_indices.update(contact_faces)

    # Calculate the contact area for unique faces only
    contact_area_total = np.sum(cell_mesh.area_faces[list(contact_face_indices)])
    
    # Calculate the fraction of contact area
    contact_fraction = contact_area_total / cell_mesh.area

    return contact_fraction, contact_area_distribution, np.mean(contact_area_distribution)
#------------------------------------------------------------------------------------------------------------





#--------------------------------------------------------------------------------------------------
############################################################################################################
# Multiprocessing implementation of the contact area fraction calculation -- Old version
############################################################################################################

# def get_contact_area_fraction(mesh_lst: list, cell_id_lst: list, cell_neighbors_lst: list, contact_cutoff: float, calculate_contact_area_fraction: bool, max_workers: int):
#     """
#     Compute the contact area fraction for each cell in the mesh list.
    
#     Parameters:
#     -----------
#         mesh_lst (list):
#             List of cell meshes.

#         cell_id_lst (list):
#             List of cell IDs.

#         cell_neighbors_lst (list):
#             List of neighbors for each cell.

#         contact_cutoff (float):
#             Contact cutoff value.

#         calculate_contact_area_fraction (bool):
#             Whether to calculate the contact area fraction.

#         max_workers (int):
#             Maximum number of workers for the ThreadPoolExecutor.

#     Returns:
#     --------
#         cell_contact_area_dict (dict):
#             Dictionary where key is cell_id and value is a tuple containing contact area fraction, contact area distribution, and mean contact area for each cell.
#     """
#     cell_contact_area_dict = {}

#     # Check if contact area fraction should be calculated
#     if calculate_contact_area_fraction:
#         # Create a pool of workers
#         with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#             future_to_cell_id = {executor.submit(compute_cell_contact_area_fraction, mesh_lst, cell_id, cell_neighbors, contact_cutoff): cell_id for cell_id, cell_neighbors in zip(cell_id_lst, cell_neighbors_lst)}

#             # Collect the results as they complete
#             for future in concurrent.futures.as_completed(future_to_cell_id):
#                 cell_id = future_to_cell_id[future]
#                 try:
#                     cell_contact_area_dict[cell_id] = future.result()
#                 except Exception as exc:
#                     print(f'Cell {cell_id} generated an exception: {exc}')
#     else:
#         for cell_id in cell_id_lst:
#             cell_contact_area_dict[cell_id] = (None, [], None)

#     return cell_contact_area_dict
#------------------------------------------------------------------------------------------------------------







#------------------------------------------------------------------------------------------------------------
def get_contact_area_fraction(mesh_lst: list, cell_id_lst: list, cell_neighbors_lst: list, contact_cutoff: float, calculate_contact_area_fraction: bool, max_workers: int):
    """
    Compute the contact area fraction for each cell in the mesh list.
    
    Parameters:
    -----------
        mesh_lst (list):
            List of cell meshes.

        cell_id_lst (list):
            List of cell IDs.

        cell_neighbors_lst (list):
            List of neighbors for each cell.

        contact_cutoff (float):
            Contact cutoff value.

        calculate_contact_area_fraction (bool):
            Whether to calculate the contact area fraction.

        max_workers (int):
            Maximum number of workers for the ThreadPoolExecutor.

    Returns:
    --------
        cell_contact_area_dict (dict):
            Dictionary where key is cell_id and value is a tuple containing contact area fraction, contact area distribution, and mean contact area for each cell.
    """
    cell_contact_area_dict = {}

    # Check if contact area fraction should be calculated
    if calculate_contact_area_fraction:
        # Create a pool of workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cell_id = {executor.submit(compute_cell_contact_area_fraction, mesh_lst, cell_id, cell_neighbors, contact_cutoff): cell_id for cell_id, cell_neighbors in zip(cell_id_lst, cell_neighbors_lst)}

            # Collect the results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_cell_id), desc='Calculating contact area fraction', total=len(cell_id_lst)):
                cell_id = future_to_cell_id[future]
                try:
                    cell_contact_area_dict[cell_id] = future.result()
                except Exception as exc:
                    print(f'Cell {cell_id} generated an exception: {exc}')
    else:
        for cell_id in cell_id_lst:
            cell_contact_area_dict[cell_id] = (None, [], None)

    return cell_contact_area_dict
#------------------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------------------
def get_statistics_df(cell_id_lst: list, cell_areas: list, cell_volumes: list, cell_isoperimetric_ratio: list, cell_neighbors_lst: list, cell_nb_of_neighbors: list, cell_principal_axis_lst: list, cell_elongation_lst: list, cell_contact_area_dict: dict):
    """
    Create a DataFrame to store cell statistics.

    Parameters:
        cell_id_lst: A list of cell IDs.
        cell_areas: A list of cell areas.
        cell_volumes: A list of cell volumes.
        cell_isoperimetric_ratio: A list of cell isoperimetric ratios.
        cell_neighbors_lst: A list of lists where each sub-list is the IDs of the neighbors of a cell.
        cell_nb_of_neighbors: A list of number of neighbors for each cell.
        cell_principal_axis_lst: A list of principal axis for each cell.
        cell_elongation_lst: A list of elongation for each cell.
        cell_contact_area_dict: A dictionary containing the contact area fraction, contact area distribution, and mean contact area for each cell.

    Returns:
        cell_statistics_df: A DataFrame containing the cell statistics.
    """
    #Reformat the principal axis list and the neighbor list to be in the format of a string
    to_scientific_str = lambda x: '{:.2e}'.format(x)
    cell_principal_axis_lst = [(" ").join(map(to_scientific_str, principal_axis.tolist())) for principal_axis in cell_principal_axis_lst]
    cell_neighbors_lst = [(" ").join(map(str, neighbors)) for neighbors in cell_neighbors_lst]

    #Extract the contact area fraction, distribution, and mean from the dictionary
    cell_contact_area_fraction = [cell_contact_area_dict[cell_id][0] for cell_id in cell_id_lst]
    cell_contact_area_distribution = [cell_contact_area_dict[cell_id][1] for cell_id in cell_id_lst]
    cell_mean_contact_area = [cell_contact_area_dict[cell_id][2] for cell_id in cell_id_lst]

    #Create a pandas dataframe to store the cell statistics
    cell_statistics_df = pd.DataFrame(
        {
            'cell_id': pd.Series(cell_id_lst),
            'cell_area': pd.Series(cell_areas),
            'cell_volume': pd.Series(cell_volumes),
            'cell_isoperimetric_ratio': pd.Series(cell_isoperimetric_ratio),
            'cell_neighbors': pd.Series(cell_neighbors_lst),
            'cell_nb_of_neighbors': pd.Series(cell_nb_of_neighbors),
            'cell_principal_axis': pd.Series(cell_principal_axis_lst),
            'cell_elongation': pd.Series(cell_elongation_lst),
            'cell_contact_area_fraction': pd.Series(cell_contact_area_fraction),
            'cell_contact_area_distribution': pd.Series(cell_contact_area_distribution),
            'cell_mean_contact_area': pd.Series(cell_mean_contact_area),
        }
    )

    return cell_statistics_df
#------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def get_filtered_cell_statistics(cell_statistics_df: pd.DataFrame, filtered_cell_id_lst: list):
    """
    Get cell statistics only for cells present in the filtered cell ID list.

    Parameters:
        cell_statistics_df: A DataFrame containing the statistics of all cells.
        filtered_cell_id_lst: A list of cell IDs that passed the filtering.

    Returns:
        filtered_cell_statistics: A DataFrame containing the statistics of only the filtered cells.
    """
    filtered_cell_statistics = cell_statistics_df[cell_statistics_df['cell_id'].isin(filtered_cell_id_lst)]
    return filtered_cell_statistics
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def save_statistics(cell_statistics_df: pd.DataFrame, filtered_cell_statistics: pd.DataFrame, output_directory: str):
    """
    Save all cell statistics and filtered cell statistics to CSV files.

    Parameters:
        cell_statistics_df: A DataFrame containing the statistics of all cells.
        filtered_cell_statistics: A DataFrame containing the statistics of only the filtered cells.
        output_directory: The directory to store the CSV files.

    """
    cell_statistics_df.to_csv(os.path.join(output_directory, "all_cell_statistics.csv"), index=False)
    filtered_cell_statistics.to_csv(os.path.join(output_directory, "filtered_cell_statistics.csv"), index=False)
#--------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
def calculate_all_statistics(mesh_lst: list, cell_id_lst: list, cell_neighbors_lst: list, filtered_cell_id_lst: list, 
                            contact_cutoff: float, output_directory: str, max_workers: int = None, 
                            calculate_contact_area_fraction: bool = False, plot: str = None, plot_type: str = 'violin') -> pd.DataFrame:
    """
    This function calculates a set of statistics for a given list of cell meshes.

    Parameters:
    - mesh_lst (list): A list of cell meshes.
    - cell_id_lst (list): A list of cell IDs.
    - cell_neighbors_lst (list): A list of cell neighbors.
    - filtered_cell_id_lst (list): A list of filtered cell IDs.
    - contact_cutoff (float): The cutoff value for contact.
    - output_directory (str): The directory where output files will be saved.
    - max_workers (int, optional): The maximum number of worker threads. If None, the number of CPUs will be used. Default is None.
    - calculate_contact_area_fraction (bool, optional): If True, the contact area fraction will be calculated. If False, it will not be calculated. Default is False.
    - plot (str, optional): The plot to generate. Can be 'filtered', 'all', 'both', or None. Default is None.
    - plot_type (str, optional): The type of plot to generate. Can be 'violin' or other types. Default is 'violin'.

    Returns:
    - pd.DataFrame: A DataFrame containing the calculated statistics
    """


    cell_areas = compute_cell_areas(mesh_lst)

    cell_volumes = compute_cell_volumes(mesh_lst)

    cell_isoperimetric_ratio = [(area**3) / (volume**2) for area, volume in zip(cell_areas, cell_volumes)]

    cell_nb_of_neighbors = [len(cell_neighbors) for cell_neighbors in cell_neighbors_lst]

    cell_principal_axis_lst, cell_elongation_lst = get_cell_principal_axis_and_elongation(mesh_lst, cell_id_lst)

    cell_contact_area_fraction_dict = get_contact_area_fraction(mesh_lst, cell_id_lst, cell_neighbors_lst, contact_cutoff, calculate_contact_area_fraction, max_workers)

    cell_statistics_df = get_statistics_df(cell_id_lst, cell_areas, cell_volumes, cell_isoperimetric_ratio, cell_neighbors_lst, cell_nb_of_neighbors, cell_principal_axis_lst, cell_elongation_lst, cell_contact_area_fraction_dict)
    
    filtered_cell_statistics = get_filtered_cell_statistics(cell_statistics_df, filtered_cell_id_lst)
    
    save_statistics(cell_statistics_df, filtered_cell_statistics, output_directory)
    
    generate_required_plots(plot, output_directory, cell_statistics_df, filtered_cell_statistics, plot_type)
    
    return cell_statistics_df
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def collect_cell_morphological_statistics(labeled_img: np.ndarray, img_resolution: np.ndarray, contact_cutoff: float,
                                          smoothing_iterations: int = 5, erosion_iterations: int = 1, dilation_iterations: int = 2,
                                          output_folder: str = 'output', meshes_only: bool = False, overwrite: bool = False, 
                                          preprocess: bool = True, max_workers: Optional[int] = None, 
                                          calculate_contact_area_fraction: bool = False, plot: Optional[str] = None, 
                                          plot_type: str = 'violin', volume_lower_threshold: Optional[int] = None,
                                          volume_upper_threshold: Optional[int] = None, *args, **kwargs) -> pd.DataFrame:

    """
    Collect the following statistics about the cells:
        - cell_areas
        - cell_volumes
        - cell_neighbors_list
        - cell_nb_of_neighbors
        - cell_principal_axis
        - cell_contact_area_fraction (optional)

    And returns this statistics in a pandas dataframe.

    Parameters:
    labeled_img: (np.array, 3D)
        The results of the curation, where the background has a label of 
        0 and the cells are labeled with consecutive integers starting from 1. The 3D image 
        is assumed to be in the standard numpy format (x, y, z).

    img_resolution: (np.array, 1D)
        The resolution of the image in microns (x_res, y_res, z_res)

    contact_cutoff: (float)
        The cutoff distance in microns for two cells to be considered in contact
    
    clear_meshes_folder: (bool, optional, default=False)
        If True, delete all the current meshes in the "cell_meshes" directory before saving new meshes

    smoothing_iterations: (int, optional, default=5)
        Number of smoothing iterations applied to the mesh

    erosion_iterations: (int, optional, default=1)
        Number of iterations for erosion during label extension

    dilation_iterations: (int, optional, default=2)
        Number of iterations for dilation during label extension

    output_folder: (str, optional, default='output')
        Name of the folder where the processed labels, cell_meshes, and filtered_cell_meshes will be saved

    meshes_only: (bool, optional, default=False)
        If True, only save the meshes and do not compute any statistics

    overwrite: (bool, optional, default=False)
        If True, overwrite the preprocessed_labels.npy file if it exists

    preprocess: (bool, optional, default=True)
        If True, apply erosion and dilation to the labeled image before processing

    max_workers: (int, optional, default=None)
        The maximum number of workers to use for parallel computation. Defaults to the number of available CPU cores minus 1.

    calculate_contact_area_fraction: (bool, optional, default=True)
        If True, calculate the contact area fraction for each cell. If False, skip this computation.
    
    volume_lower_threshold: (int, optional, default=None)
        The minimum volume to include a cell in the statistics. If not specified (None), no lower threshold is applied.

    volume_upper_threshold: (int, optional, default=None)
        The maximum volume to include a cell in the statistics. If not specified (None), no upper threshold is applied.

    Returns:
    filtered_cell_statistics: (pd.DataFrame)
        A pandas dataframe containing the filtered cell statistics (cells not touching the background)
        
    Example:
    ```python
    import numpy as np
    
    # Load the labeled image (numpy array)
    labeled_img = np.load("path/to/labeled_img.npy")

    # Set the image resolution in microns
    img_resolution = np.array([0.21, 0.21, 0.39])

    # Set the contact cutoff distance for cells in microns
    contact_cutoff = 1.0

    # Compute the filtered cell statistics
    filtered_cell_statistics = collect_cell_morphological_statistics(
        labeled_img=labeled_img,
        img_resolution=img_resolution,
        contact_cutoff=contact_cutoff,
        output_folder="output",
        calculate_contact_area_fraction=True,
        volume_lower_threshold=200,
        volume_upper_threshold=500
    )

    print(filtered_cell_statistics)
    ```
    """
    
    mesh_lst, cell_id_lst, filtered_cell_id_lst, cell_neighbors_lst, output_directory = full_label_processing(
        labeled_img, img_resolution, smoothing_iterations, erosion_iterations, dilation_iterations, 
        output_folder, preprocess, overwrite, volume_lower_threshold, volume_upper_threshold)
    
    if meshes_only:
        return
    
    cell_statistics_df = calculate_all_statistics(mesh_lst, cell_id_lst, cell_neighbors_lst, filtered_cell_id_lst, 
                                                  contact_cutoff, output_directory, max_workers, calculate_contact_area_fraction, 
                                                  plot, plot_type)

    return cell_statistics_df
#-----------------------------------------------------------------------------------------------------------


