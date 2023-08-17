import numpy as np
import pandas as pd
import trimesh as tm
import concurrent.futures
from scipy import ndimage
from tqdm import tqdm
from skimage.measure import mesh_surface_area
from collections import defaultdict
from typing import Optional, List, Tuple, Iterable, Dict, Union
import napari
from scipy.spatial.transform import Rotation
from typing import Union, Optional, Iterable, List, Tuple
from morphosamplers.sampler import (
    generate_2d_grid,
    place_sampling_grids,
    sample_volume_at_coordinates,
)
from skimage.measure import regionprops
from scripts.ExtendedTrimesh import ExtendedTrimesh
from scripts.utils import (
    _get_centroid_and_length, 
    _get_rotation, 
    _get_principal_axis, 
    _get_slices_along_direction, 
    find_closest
)


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
    """
    Given an array of pixel counts for different labels and the size of each pixel, compute the area.

    Parameters:
    -----------
        pixel_counts: (np.ndarray[int])
            An array of pixel counts for each label.

        pixel_size: Iterable[float]
            The pixel size expressed as a pair of float (in microns).
    
    Returns:
    --------
        areas: (np.ndarray[float])
            An array of areas, one value for each label.
    """
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



#------------------------------------------------------------------------------------------------------------
def _compute_2D_area_along_direction(
		labeled_slice: np.ndarray[int], 
		cell_label: np.ndarray[int],
		pixel_size: Iterable[float]
) -> float:
	
	binary_slice = (labeled_slice == cell_label).astype(np.uint16)
	if np.any(binary_slice):
		pixel_count = np.sum(binary_slice)
		pixel_area = pixel_size[0] * pixel_size[1]
		area = pixel_count * pixel_area
		return area
	else:
		return 0.0
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _compute_2D_neighbors_along_direction(
        labeled_slice: np.ndarray[int], 
        cell_label: np.ndarray[int],
        background_threshold: float = 0.1
) -> List[int]:

	#Get the pixels of the cell
	binary_slice = labeled_slice == cell_label

	# Check if cell is present in the slice
	if not np.any(binary_slice):
		return [-1]

	#Expand the volume of the cell by 2 voxels in each direction
	expanded_cell_voxels = ndimage.binary_dilation(binary_slice, iterations=2)
		
	#Find the voxels that are directly in contact with the surface of the cell
	cell_surface_voxels = expanded_cell_voxels ^ binary_slice

	#Get the labels of the neighbors
	neighbors, counts = np.unique(labeled_slice[cell_surface_voxels], return_counts=True)

	#Check if the label is touching the background above a certain threshold
	# print(f'Cell {cell_label}: {neighbors}, {counts}')
	if (0 in neighbors) and (counts[0] > np.sum(counts) * background_threshold):
		return [-1]
	else:
		#Remove the label of the cell itself, and the label of the background from the neighbors list
		neighbors = neighbors[(neighbors != cell_label) & (neighbors != 0)]
		return list(neighbors)
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _compute_neighbors_of_neighbors_along_direction(
        labeled_img: np.ndarray[int],
        neighbors: Iterable[int],
        grid_coords: np.ndarray[float],
        principal_axes: Dict[int, np.ndarray[float]],
        principal_axis_pts: Dict[int, np.ndarray[float]],
        grid_to_place: np.ndarray[float]
) -> List[int]:

    # iterate over neighbors from a slice to compute neighbors of neighbors
    neigh_num_neighbors = []
    for neighbor in neighbors:
        # get points on principal axis of neighboring cell
        neigh_principal_pts = principal_axis_pts[neighbor]
        neigh_principal_vector = principal_axes[neighbor]
        # get intersection between grid of main cell and points of neighbor principal axis
        neigh_center = find_closest(grid_coords, neigh_principal_pts, 20)
        # place grid and sample slice for neighbor
        neigh_rot = _get_rotation(neigh_principal_vector)
        neigh_placed_grid = place_sampling_grids(grid_to_place, neigh_center, neigh_rot)
        neigh_sampled_slice = sample_volume_at_coordinates(
            labeled_img,
            neigh_placed_grid,
            interpolation_order=0,
        )
        # compute number of neighbors for neighbor
        neigh_neighbors = _compute_2D_neighbors_along_direction(neigh_sampled_slice, neighbor, 0)
        # if any neighbor of main cell doesn't have complete neighborhood go to next slice
        if neigh_neighbors == [-1]:
            break
        else:
            neigh_num_neighbors.append(len(neigh_neighbors))

    if len(neigh_num_neighbors) == len(neighbors):
        return neigh_num_neighbors
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def compute_2D_statistics_along_axes(
        labeled_img: np.ndarray[int],
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        exclude_labels: Iterable[int],
        voxel_size: Iterable[float],
        number_slices: int = 10, 
        slice_size: int = 200,
        remove_empty: Optional[bool] = True
) -> Tuple[Dict[int, List[List[int]]], 
           Dict[int, List[float]], 
           Dict[int, Dict[int, List[int]]], 
           Dict[int, Tuple[List[List[float]], List[float]]]]:
    
    if np.any(slice_size > np.asarray(labeled_img.shape)):
        slice_size = np.min(labeled_img.shape)
    
    print('Computing cell 2D statistics along apical-basal axis...')

    # Iterate over the cells
    label_ids = np.unique(labeled_img)

    # Compute principal axes, centroids and lengths for all the cells
    cell_centroids, cell_lengths, cell_principal_axes = {}, {}, {}
    cell_principal_vectors = {} # array of points on the direction of the principal axes
    for label_id in tqdm(label_ids[1:], desc='Computing principal axes and centroids'):
        if label_id in exclude_labels:
            cell_principal_axes[label_id] = None
            cell_centroids[label_id] = None
            cell_lengths[label_id] = None
        else:
            # Compute principal axis, axis length and centroid coordinates 
            cell_mesh = cell_mesh_dict[label_id]
            principal_axis = _get_principal_axis(
                mesh=cell_mesh,
                scale=voxel_size
            )
            cell_principal_axes[label_id] = principal_axis

            binary_img = (labeled_img == label_id).astype(np.uint8)
            cell_centroid, cell_length = _get_centroid_and_length(binary_img)
            cell_length = int(cell_length // 2)
            cell_centroids[label_id] = cell_centroid
            cell_lengths[label_id] = cell_length
            cell_principal_vectors[label_id] = np.asarray([
                principal_axis * i + cell_centroid 
                for i in np.linspace(-cell_length, cell_length, number_slices)
            ])
    
    # Generate a grid of the desired size for sampling from the image
    grid_shape = [slice_size + 1] * 2
    grid = generate_2d_grid(grid_shape)

    neighbors_dict = {}
    areas_dict = {}
    neighbors_of_neighbors_dict = {}
    slices_dict = {}
    for label_id in tqdm(label_ids[1:], desc='Computing 2D statistics along apical-basal axis:'):
        if label_id in exclude_labels:
            neighbors_dict[label_id] = []
            areas_dict[label_id] = []
            neighbors_of_neighbors_dict[label_id] = {}
            slices_dict[label_id] = ()
        else:
            # Get slices along principal axis direction
            labeled_slices, grid_coords, slices_specs = _get_slices_along_direction(
                labeled_img=labeled_img,
                slicing_dir=cell_principal_axes[label_id],
                centroid=cell_centroids[label_id],
                height=cell_lengths[label_id],
                grid_to_place=grid,
                num_slices=number_slices
            )

            # Iterate across slices to compute the statistics
            cell_areas = []
            cell_neighbors = []
            cell_neighbors_of_neighbors = defaultdict(list)
            for i, labeled_slice in enumerate(labeled_slices):
                area_in_slice = _compute_2D_area_along_direction(
                    labeled_slice=labeled_slice,
                    cell_label=label_id,
                    pixel_size=voxel_size[:2]
                )
                cell_areas.append(area_in_slice)

                neighbors_in_slice = _compute_2D_neighbors_along_direction(
                    labeled_slice=labeled_slice,
                    cell_label=label_id
                )
                cell_neighbors.append(neighbors_in_slice)
                
                # check for incomplete neighborhood
                if (neighbors_in_slice == [-1]) or np.any(np.isin(neighbors_in_slice, exclude_labels)): 
                    continue
                neighbors_of_neighbors_in_slice = _compute_neighbors_of_neighbors_along_direction(
                    labeled_img=labeled_img,
                    neighbors=neighbors_in_slice,
                    grid_coords=grid_coords[i],
                    principal_axis_pts=cell_principal_vectors,
                    principal_axes=cell_principal_axes,
                    grid_to_place=grid
                )
                if neighbors_of_neighbors_in_slice:
                    cell_neighbors_of_neighbors[len(neighbors_in_slice)].append(neighbors_of_neighbors_in_slice)
            
            if remove_empty:
                # Post-process results to remove empty values
                to_remove = [neighs == [-1] for neighs in cell_neighbors]
                cell_neighbors = [neighs for neighs, flag in zip(cell_neighbors, to_remove) if not flag]
                cell_areas = [area for area, flag in zip(cell_areas, to_remove) if not flag]
                new_slices_specs = [item for item, flag in zip(slices_specs[0], to_remove) if not flag]
                slices_specs = (new_slices_specs, slices_specs[1])
                
            neighbors_dict[label_id] = cell_neighbors
            areas_dict[label_id] = cell_areas
            neighbors_of_neighbors_dict[label_id] = dict(cell_neighbors_of_neighbors)
            slices_dict[label_id] = slices_specs

    return neighbors_dict, areas_dict, neighbors_of_neighbors_dict, slices_dict
#------------------------------------------------------------------------------------------------------------