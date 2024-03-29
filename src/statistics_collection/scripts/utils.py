import os
import json
import numpy as np
from skimage import io
from scipy.spatial import kdtree
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
from skimage.measure import regionprops
from morphosamplers.sampler import (
    place_sampling_grids,
    sample_volume_at_coordinates,
)
import trimesh as tm
from types import SimpleNamespace
from typing import Union, Tuple, Optional, List, Dict, Iterable


#------------------------------------------------------------------------------------------------------------
def load_labeled_img(
        labeled_img: Union[str, np.ndarray]
    ) -> np.ndarray:
    """
    Load a labeled image.

    Parameters:
        labeled_img: If string, the path of the labeled image file. If numpy array, the actual labeled image.

    Returns:
        labeled_img: The loaded labeled image.
    """
    if isinstance(labeled_img, str) and os.path.isfile(labeled_img):
        labels = io.imread(labeled_img)
        labeled_img = np.einsum('kij->ijk', labels)

    return labeled_img
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def create_output_directory(
        output_folder: str, 
        input_img_path: str,
        smoothing_iterations: int, 
        erosion_iterations: int, 
        dilation_iterations: int
    ) ->  str:
    """
    Create a directory to store the output data.

    Parameters:
        output_folder: (str) 
            The root folder for storing the outputs.
        input_img_path: (str)
            The path to the image to analyze, used to produce an
            identifier for the directory.
        smoothing_iterations: (int)
            The number of smoothing iterations in preprocessing.
        erosion_iterations: (int)
            The number of erosion iterations in preprocessing.
        dilation_iterations: (int)
            The number of dilation iterations in preprocessing.

    Returns:
        output_directory: (str) 
            The full path of the created output directory.
    """
    fname = os.path.basename(input_img_path).replace('.tif', '')
    output_directory = f"{output_folder}_{fname}_s_{smoothing_iterations}_e_{erosion_iterations}_d_{dilation_iterations}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_directory
#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """
    When printing a warning suppress everything except for the message and the category of the warning.
    """
    print(f"{category.__name__}: {message}")
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def read_config(path):
    """
    Read args from config.json file.

    Parameters:
    -----------
        path: (str)
            Path to the config.json file storing parameters for running the code.
        
    Returns:
    --------
        args: 
        A dictionary of parameters, whose attributes can be accessed by `args.my_attribute`.

    """
    with open(path) as f:
        args = json.load(f, object_hook = lambda d: SimpleNamespace(**d))

    return args

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def find_closest(
        point_cloud1: np.ndarray,
        point_cloud2: np.ndarray,
) -> np.ndarray:
    """
    Given two 3D pointclouds, find the closest pair of points, each one belonging to one 
    of the two point clouds. Then, return the midpoint among the pair.
    
    Parameters:
    -----------
        points_cloud1: (np.ndarray)
            A (M, 3) array, whose rows are the coordinates of the M points in the point cloud.

        points_cloud2: (np.ndarray)
            A (N, 3) array, whose rows are the coordinates of the N points in the point cloud.

    Returns:
    --------
        (np.ndarray):
            The midpoint among the pair of closest points, each one belonging to one 
            of the two point clouds.
    """
    
    # Calculate the pairwise Euclidean distances between all points in the two clouds
    distances = np.sqrt(((point_cloud1[:, np.newaxis] - point_cloud2) ** 2).sum(axis=-1))

    # Find the minimum distance and the indices of the closest pair of points
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)

    # Get the closest pair of points
    closest_point1 = point_cloud1[min_indices[0]]
    closest_point2 = point_cloud2[min_indices[1]]
    
    return (closest_point1 + closest_point2) / 2
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def get_centroid_and_length(
        binary_img: np.ndarray[int]
) -> Tuple[np.ndarray[float], float]:
    """
    Compute the centroid and the major axis length of a binary image.

    Parameters:
    -----------
        binary_img: (np.ndarray[int])
			A binary image stored in a N-dimensional numpy array.
    
    Returns:
    --------
		centroid: (np.ndarray[int])
            An array of shape (N, ) storing the coordinates of the object in the binary image.

        length: (float)
            The length of the major axis of the object in the binary image.
    """ 

    props = regionprops(binary_img)[0]
    centroid = props.centroid
    _, _, min_z, _, _, max_z = props.bbox
    length = max_z - min_z

    return centroid, length
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def get_slices_along_direction(
    labeled_img: np.ndarray[int],
    slicing_dir: Iterable[float],
    centroid: Iterable[float],
    height: int,    
    grid_to_place: np.ndarray[int],
    original_voxel_size: Optional[float],
    num_slices: Optional[int] = 10,
) -> Tuple[List[np.ndarray[int]], Tuple[List[List[float]], np.ndarray[float], List[float]]]:
    """
    Extract 2D slice along a given direction from a 3D labeled image.

    Parameters:
    -----------

    labeled_img: (np.ndarray[int])
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.
    
    slicing_dir: (Iterable[float])
        A triplet describing a unit vector in the labeled_img 3D coordinate system.

    centroid: (Iterable[float])
        A triplet associated to the coordinates of the centroid of the object at the center
        of the slices.
    
    height: (int)
        The height of the sliced volume above and below the centroid (i.e. total volum is 2*height).
    
    slice_size: (Optional[int], default=200)
        The size of the each side (orthogonal to slicing_dir) of the grid used to extract slices.
    
    num_slices: (Optional[int], default=10)
        The number of slices to extract from the labeled image.
    
    original_voxel_size: (np.ndarray[float])
        The voxel size in the original coordinate system.

    Returns:
    --------

    labeled_slices: (List[np.ndarray[int]])
        A list of slices obtained along slicing_dir direction.

    grid_coords: (List[np.ndarray[float]])
        A list of grid coordinates used to sample slices for the original volume.
    
    new_voxel_size: (np.ndarray[float])
        The voxel size seen in the coordinate system defined by the new slicing direction.
        Note that if the voxels in the original volume are isotropic, the new voxel size is 
        the same as the original one. 
    
    grid_specs: (Tuple[List[List[float]], List[float]]])
        A tuple consisting of lists of coordinates of grid centers and slicing directions (which
        is always the same). These values are used to identify the grids that have been used
        to sample labeled_slices from labeled_img.
    """
    
    # Define the centers of the sampling grids
    slicing_dir = np.asarray(slicing_dir)
    grid_centers = [
        slicing_dir * i + centroid 
        for i in np.linspace(-height, height, num_slices)
    ]

    # Compute the rotation matrix
    rot = get_rotation(slicing_dir)
    
    # Compute the voxel sizes in the new coordinate system
    new_axes_directions = rot[0].as_matrix() # (X, Y, Z as column vectors, Z is the principal axis)
    new_voxel_size = abs(np.dot(new_axes_directions, original_voxel_size))

    # Store identifiers of the different grids (centers and direction)
    grid_specs = (
        [list(center) for center in grid_centers], 
        list(slicing_dir)
    )

    labeled_slices = []
    grid_coords = []
    for center in grid_centers:
        # Rotate and translate the grid
        grid_center_point = center
        sampling_coordinates = place_sampling_grids(
            grid_to_place, grid_center_point, rot
        )
        grid_coords.append(sampling_coordinates.reshape(-1, 3))

        # Sample values from the grid
        sampled_slice = sample_volume_at_coordinates(
            labeled_img,
            sampling_coordinates,
            interpolation_order=0,
        )

        labeled_slices.append(sampled_slice)

    return labeled_slices, grid_coords, new_voxel_size, grid_specs
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def get_principal_axis(
        mesh: tm.base.Trimesh,
		scale: Iterable[float],
) -> np.ndarray[float]:
	"""
	Compute principal axis of from a mesh object.

	Parameters:
	-----------

    mesh: (tm.base.Trimesh)
        A Trimesh object in a N-dimensional space (N=2,3). 

    scale: (np.ndarray[float])
        An array of shape (N,) containing scale of mesh coordinate system in microns.
        NOTE: in the statistics collection pipeline the labeled image is to be 
        considered in a coordinate system of scale (1, 1, 1), whereas meshes are 
        generated in a different system with scale (voxel_size). 
        Therefore for this task we need to move the principal axis computed on the mesh
        into the labeled image coordinate system. 

	Returns:
	--------

    normalized_principal_axis: (np.ndarray[float])
        An array of shape (N,) representing the components of the rescaled and normalized 
        principal axis.
	"""
	eigen_values, eigen_vectors = tm.inertia.principal_axis(mesh.moment_inertia)
	smallest_eigen_value_idx = np.argmin(np.abs(eigen_values))
	principal_axis = eigen_vectors[smallest_eigen_value_idx]
	rescaled_principal_axis = np.asarray(principal_axis) / np.asarray(scale)
	normalized_principal_axis = rescaled_principal_axis / np.linalg.norm(rescaled_principal_axis)

	return normalized_principal_axis

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def get_rotation(
        principal_vector: np.ndarray
    ) -> Rotation:
    """
    Given the direction of the principal vector, compute the rotation matrix 
    finding the pair of orthogonal unit vectors that form the new coordinate 
    system.

    Parameters:
    -----------

    principal_vector: np.ndarray
        The principal vector that determines the rotation.

    """
    random_vector = np.random.rand(3)
    normal_vector = random_vector - np.dot(random_vector, principal_vector) * principal_vector
    normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
    third_vector = np.cross(principal_vector, normal_unit_vector)
    rot_matrix = np.column_stack(
        (normal_unit_vector, third_vector, principal_vector)
    )
    return [Rotation.from_matrix(rot_matrix)]