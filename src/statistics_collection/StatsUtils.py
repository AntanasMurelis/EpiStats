import numpy as np
import pandas as pd
import trimesh as tm
import concurrent.futures
from scipy import ndimage
from tqdm import tqdm
from skimage.measure import mesh_surface_area
from collections import defaultdict
from typing import Optional, List, Tuple, Iterable, Dict, Union
from ExtendedTrimesh import ExtendedTrimesh
import napari


#------------------------------------------------------------------------------------------------------------
def compute_cell_surface_areas(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        exclude_labels: Iterable[int],
    ) -> Dict[int, float]:
    """
    Use the meshes of the cells to compute their areas (in micron squared).
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The tirangular meshes of the cell in the standard trimesh format
        associated to the corresponding cell label.

    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns:
    --------
    area_dict: (Dict[int, float])
        The surface area of each cell with the corresponding cell label
    """

    assert cell_mesh_dict.__len__() > 0

    area_dict = {}

    for id, mesh in tqdm(cell_mesh_dict.items(),
                         desc="Computing cell surface area",
                         total=len(cell_mesh_dict)):
        if id not in exclude_labels:
            area_dict[id] = mesh.area
        else:
            area_dict[id] = None
    
    return area_dict
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def compute_cell_volumes(
        labeled_img: np.ndarray[int],
        exclude_labels: Iterable[int],
        voxel_size: Iterable[float]
    ) -> Dict[int, float]:
    """
    Use the meshes of the cells to compute their volumes (in micron cubes).
    
    Parameters:
    -----------
    labeled_img: (np.ndarray, 3D, dtype=int)
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.

    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from computation.

    voxel_size: (Iterable[float])
        The voxel size of the input labeled image.

    Returns:
    --------
    volume_dict: (Dict[int, float])
        List of the volume of each cell
    """

    volume_dict = {}

    label_ids, label_counts = np.unique(labeled_img, return_counts=True)

    for label_id, label_count in tqdm(zip(label_ids[1:], label_counts[1:]), 
                             desc="Computing cell volume",
                             total=len(label_ids)-1):
        if label_id not in exclude_labels:
            volume_dict[label_id] = label_count*np.prod(voxel_size)
        else:
            volume_dict[label_id] = None

    return volume_dict
#------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def compute_cell_principal_axis_and_elongation(
    cell_mesh_dict: Dict[int, tm.base.Trimesh],
    exclude_labels: Iterable[int],
) -> Tuple[Dict[int, tm.base.Trimesh], Dict[int, float]]:
    """
    Compute the principal axis and elongation for each cell in the mesh list.
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The tirangular meshes of the cell in the standard trimesh format
        associated to the corresponding cell label

    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns:
    --------
    cell_cell_elongation_axes_dict (Dict[int, Tuple[float, List[float]]]):
        Dict whose keys are cell ids and values are tuples of elongation and
        principal axes for the cell id.

    """

    # Initialize lists to store the principal axis and elongation for each cell
    cell_elongation_axes_dict = {}

    # Loop over each cell ID
    for cell_id, cell_mesh in tqdm(cell_mesh_dict.items(), 
                                   desc='Computing cell principal axis and elongation',
                                   total=len(cell_mesh_dict)):
        if cell_id not in exclude_labels:
            # Use the inertia tensor of the cell shape to compute its principal axis
            eigen_values, eigen_vectors = tm.inertia.principal_axis(cell_mesh.moment_inertia)

            # Get the index of the smallest eigen value
            smallest_eigen_value_idx = np.argmin(np.abs(eigen_values))
            greatest_eigen_value_idx = np.argmax(np.abs(eigen_values))

            # Get the corresponding eigen vector 
            principal_axes = eigen_vectors[smallest_eigen_value_idx]
            # Compute elongation
            elongation = np.sqrt(eigen_values[greatest_eigen_value_idx] / eigen_values[smallest_eigen_value_idx])
        else:
            principal_axes = None
            elongation = None

        # Store the results in the corresponding lists
        cell_elongation_axes_dict[cell_id] = elongation, principal_axes

    # Return the lists of principal axes and elongations
    return cell_elongation_axes_dict
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def compute_cell_neighbors(
        labeled_img: np.ndarray[int], 
        exclude_labels: Iterable[int],
    ) -> Dict[int, List[int]]:
    """
    Get all the neighbors of a given cell. Two cells are considered neighborhs if 
    a subset of their surfaces are directly touching.
    
    Parameters:
    -----------
    labeled_img: (np.ndarray, 3D, dtype=int)
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.
    
    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns
    -------
    neighbors_dict: (Dict[int, List[int]])
        A dict whose key is the cell label and the value is a list of the ids of the neighbors
    """

    label_ids = np.unique(labeled_img)

    neighbors_dict = {}

    for label in tqdm(label_ids[1:], desc="Computing cell neighbors"):
        if label not in exclude_labels:
            #Get the voxels of the cell
            binary_img = labeled_img == label

            #Expand the volume of the cell by 2 voxels in each direction
            expanded_cell_voxels = ndimage.binary_dilation(binary_img, iterations=2)

            #Find the voxels that are directly in contact with the surface of the cell
            cell_surface_voxels = expanded_cell_voxels ^ binary_img

            #Get the labels of the neighbors
            neighbors_lst = np.unique(labeled_img[cell_surface_voxels])

            #Remove the label of the cell itself, and the label of the background from the neighbors list
            neighbors_lst = neighbors_lst[(neighbors_lst != label) & (neighbors_lst != 0)]
        else:
            neighbors_lst = []

        # Append to the dictionary
        neighbors_dict[label] = neighbors_lst

    return neighbors_dict
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def _compute_contact_area(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        cell_id: int, 
        cell_neighbors_lst: List[int], 
        contact_cutoff: float,
        show_napari: Optional[bool] = False
    ) -> Tuple[float, List[float]]:
    """
    Compute the fraction of the cell surface area which is in contact with other cells.
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The triangular meshes of the cell in the standard trimesh format
        with the corresponding cell label as key.

    cell_id: (int)
        The id of the cell for which we want to calculate the contact area fraction.

    cell_neighbors_lst: (List[int])
        List of the ids of the neighbors of the current cell.

    contact_cutoff: (float)
        The cutoff distance in microns for two cells to be considered in contact.

    show_napari: (Optional[bool], default=False)
        For debugging purpose.
        If `True` opens a napari viewer to show the cell mesh, the neighbors meshes and the
        contact areas.

    Returns:
    --------
    contact_fraction: (float)
        The percentage of the total cell surface are in contact with neighboring cells.
    
    contact_area_distribution: (List[float])
        A list of the contact areas between the cell and each one of the neigbors.
    """

    num_cells = len(cell_mesh_dict)

    if len(cell_neighbors_lst) > 0:
        print(f'    Calculating contact area for cell {cell_id}/{num_cells} ... ')
        cell_mesh = cell_mesh_dict[cell_id]
        cell_mesh = ExtendedTrimesh(cell_mesh.vertices, cell_mesh.faces)
        contact_face_indices = set()
        contact_area_distribution = np.zeros(len(cell_neighbors_lst))
        if show_napari:
            neighbor_meshes = []
            contact_faces_lst = []
        # Loop over the neighbors of the cell
        for i, neighbor_id in enumerate(cell_neighbors_lst):
            # Get potential contact faces using the get_potential_contact_faces method from ExtendedTrimesh class
            neighbor_mesh = ExtendedTrimesh(cell_mesh_dict[neighbor_id].vertices, cell_mesh_dict[neighbor_id].faces)
            contact_faces = neighbor_mesh.get_potential_contact_faces(cell_mesh, contact_cutoff)
            
            # Calculate contact area for the current cell and add to distribution
            contact_area = np.sum(cell_mesh.area_faces[contact_faces])
            contact_area_distribution[i] = contact_area
            contact_face_indices.update(contact_faces)

            if show_napari:
                neighbor_meshes.append(neighbor_mesh)
                contact_faces_lst.append(contact_faces)

        # Calculate total and fraction of contact area
        contact_area_total = np.sum(cell_mesh.area_faces[list(contact_face_indices)])
        contact_fraction = contact_area_total / cell_mesh.area

        if show_napari:
            with napari.gui_qt():
                viewer = napari.Viewer()
                viewer.add_surface((cell_mesh.vertices, cell_mesh.faces), name='target', blending='additive', colormap='green')
                for i, neighbor_mesh in enumerate(neighbor_meshes):
                    viewer.add_surface((cell_mesh.vertices, cell_mesh.faces[contact_faces_lst[i]]), colormap='red', name='contact_faces')
                    viewer.add_surface((neighbor_mesh.vertices, neighbor_mesh.faces), name=f'neighbor_{i}', blending='additive', opacity=0.5)

    else:
        print(f'    Skipping cell {cell_id}/{num_cells} ...')
        contact_fraction, contact_area_distribution = None, []

    return contact_fraction, contact_area_distribution
#------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def compute_cell_contact_area(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        cell_neighbors_dict: Dict[int, List[int]], 
        max_workers: int,
        contact_cutoff: Optional[float] = 0.5, 
    ) -> Dict[any, Union[float, List[float]]]:
    """
    Compute the contact area fraction for each cell in the mesh list.
    
    Parameters:
    -----------
        cell_mesh_dict (Dict[int, tm.base.Trimesh]):
            Dictionary of cell meshes.

        cell_neighbors_lst (List[int]):
            List of neighbors for each cell.

        max_workers (int):
            Maximum number of workers for the ThreadPoolExecutor.
        
        contact_cutoff (Optional[float], default=0.1):
            The cutoff distance in microns for two cells to be considered in contact.

    Returns:
    --------
        cell_contact_area_dict (Dict[int, Union[float, List[float]]]):
            Dictionary where key is cell_id and value is a tuple containing contact area fraction, 
            and contact area distribution.
    """
    print('Computing cell contact area ...', end='')    
    
    cell_contact_area_dict = {}

    # Create a pool of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cell_id = {
            executor.submit(_compute_contact_area, cell_mesh_dict, cell_id, cell_neighbors, contact_cutoff): cell_id 
            for cell_id, cell_neighbors in cell_neighbors_dict.items()
        }

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(future_to_cell_id):
            cell_id = future_to_cell_id[future]
            try:
                cell_contact_area_dict[cell_id] = future.result()
            except Exception as exc:
                print(f'Cell {cell_id} generated an exception: {exc}')

    return cell_contact_area_dict
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _compute_2D_area(
        pixel_counts: np.ndarray[int],
        pixel_size: Iterable[float]
) -> np.ndarray[float]:

    # Compute areas
    pixel_area = pixel_size[0]*pixel_size[1]
    areas = pixel_counts[1:]*pixel_area

    return areas

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _compute_2D_neighbors(
        labeled_slice: np.ndarray[int], 
        exclude_labels: Iterable[int],
        label_ids: np.ndarray[int]
) -> Dict[int, List[int]]:
    
    neighbors_dict = {}
    for label in label_ids[1:]:
        if label not in exclude_labels:
            #Get the voxels of the cell
            binary_slice = labeled_slice == label

            #Expand the volume of the cell by 2 voxels in each direction
            expanded_cell_voxels = ndimage.binary_dilation(binary_slice, iterations=2)

            #Find the voxels that are directly in contact with the surface of the cell
            cell_surface_voxels = expanded_cell_voxels ^ binary_slice

            #Get the labels of the neighbors
            neighbors_lst = np.unique(labeled_slice[cell_surface_voxels])

            #Remove the label of the cell itself, and the label of the background from the neighbors list
            neighbors_lst = neighbors_lst[(neighbors_lst != label) & (neighbors_lst != 0)]
        else:
            neighbors_lst = []

        neighbors_dict[label] = list(neighbors_lst)

    return neighbors_dict

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def compute_2D_statistics(
        labeled_img: np.ndarray[int],
        slicing_dim: int,
        exclude_labels: Iterable[int],
        pixel_size: Iterable[float]
) -> Dict[int, Tuple[List[float], List[List[int]]]]:

    # Change axis order putting the slicing axis first    
    if slicing_dim == 0:
        reordering_expr = 'ijk->ijk'
    elif slicing_dim == 1:
        reordering_expr = 'ijk->jik'
    elif slicing_dim == 2: 
        reordering_expr = 'ijk->kij'
    else:
        raise ValueError('Labeled image is 3D, so slicing_dim must be either 0, 1, or 2.')

    reordered_labeled_img = np.einsum(reordering_expr, labeled_img)

    # Initialize temporary dictionaries that store statistics by slice index
    temp_neighbors_2D = []
    temp_area_2D = []
    for labeled_slice in tqdm(reordered_labeled_img, desc='Computing cell 2D statistics'):
        # Compute statistics for the current slice
        slice_ids, slice_counts = np.unique(labeled_slice, return_counts=True)
        if len(slice_ids) > 1: #check it is not only background
            curr_neighbors_dict = _compute_2D_neighbors(labeled_slice, exclude_labels, slice_ids)
            curr_areas_list = _compute_2D_area(slice_counts, pixel_size)

            # Gather statistics by label
            temp_neighbors_2D.append(curr_neighbors_dict)
            temp_area_2D.append(curr_areas_list)

    # Unwrap the temporary data structures to have dictionaries indexes by label
    neighbors_2D_dict = defaultdict(list)
    area_2D_dict = defaultdict(list)
    slices_dict = defaultdict(list)
    for slice_id in range(len(temp_area_2D)):
        for i, label in enumerate(temp_neighbors_2D[slice_id].keys()):
            neighbors_2D_dict[label].append(temp_neighbors_2D[slice_id][label])
            area_2D_dict[label].append(temp_area_2D[slice_id][i])
            slices_dict[label].append(slice_id)

    return dict(neighbors_2D_dict), dict(area_2D_dict), dict(slices_dict)

#------------------------------------------------------------------------------------------------------------
