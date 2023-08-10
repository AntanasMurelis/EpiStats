import numpy as np
from skimage import io
import os
import json
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
def find_closest(points_cloud1, points_cloud2, lower_threshold):
    # Iterate through all combinations of points
    min_dist = float("inf")
    closest_points = None, None
    found = False
    midpoint = points_cloud1.shape[0]//2
    for i in range(midpoint):
        for j in range(points_cloud2.shape[0]):
            point1, point2 = points_cloud1[midpoint+i, :], points_cloud2[j, :]
            distance = euclidean(point1, point2)
            if distance < lower_threshold:
                closest_points = point1, point2
                found = True
                break
            if distance < min_dist:
                closest_points = point1, point2
                min_dist = distance
            point1 = points_cloud1[midpoint-i, :]
            distance = euclidean(point1, point2)
            if distance < lower_threshold:
                closest_points = point1, point2
                found = True
                break
            if distance < min_dist:
                closest_points = point1, point2
                min_dist = distance
        if found: break
    
    return sum(closest_points) / 2
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _get_centroid_and_length(
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
def _get_slices_along_direction(
    labeled_img: np.ndarray[int],
    slicing_dir: Iterable[float],
    centroid: Iterable[float],
    height: int,    
    grid_to_place: np.ndarray[int],
    num_slices: Optional[int] = 10,
) -> Tuple[List[np.ndarray[int]], Tuple[List[List[float]], List[float]]]:
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

    Returns:
    --------
        labeled_slices: (List[np.ndarray[int]])
            A list of slices obtained along slicing_dir direction.
        
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
    rot = _get_rotation(slicing_dir)

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

    return labeled_slices, grid_coords, grid_specs
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def _get_principal_axis(
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
def _get_rotation(principal_vector):
    random_vector = np.random.rand(3)
    normal_vector = random_vector - np.dot(random_vector, principal_vector) * principal_vector
    normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
    third_vector = np.cross(principal_vector, normal_unit_vector)
    rot_matrix = np.column_stack(
        (normal_unit_vector, third_vector, principal_vector)
    )
    return [Rotation.from_matrix(rot_matrix)]