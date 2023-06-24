import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndimage
from skimage.measure import regionprops
import trimesh as tm
import os
from napari_process_points_and_surfaces import label_to_surface
from statistics_collection.misc import load_labeled_img, create_output_directory




#-----------------------------------------------------------------------------------------------------------------------------

#######################################################################################################################
# Removing unconnected regions:
# Naive implementation
#######################################################################################################################

def remove_unconnected_regions(labeled_img, pad_width=10):
    """
    Removes regions of labels that are not connected to the main cell body.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.

    padding: (int, optional, default=1)
        The number of pixels to pad the labeled image along each axis.

    Returns:
    --------
    filtered_labeled_img: (np.array, 3D)
        A 3D labeled image with unconnected regions removed.
    """
    
    unique_labels = np.unique(labeled_img)
    filtered_labeled_img = labeled_img.copy()

    for label in tqdm(unique_labels, desc='Removing unconnected regions'):
        if label == 0:
            continue
        binary_mask = (filtered_labeled_img == label).astype(np.uint8)

        # Label connected regions
        labeled_mask, num_features = ndimage.label(binary_mask)
        # Remove unconnected regions
        if num_features > 1:
            region_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
            largest_region_label = np.argmax(region_sizes[1:]) + 1
            filtered_region = labeled_mask == largest_region_label
            filtered_labeled_img[binary_mask != filtered_region] = 0

    return filtered_labeled_img
#-----------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------

#######################################################################################################################
# Steven's implementation:
#######################################################################################################################

# def remove_unconnected_regions(label_img_ar):
#     """
#     Some of the labels are not entirely connected in space. This scripts find the voxels that 
#     belong to different regions of the cell label and then only keep the voxels that belong to 
#     the biggest region. Could be done manually but this script saves a lot of time

#     Parameters:
#     ------------

#     label_img_ar (np.array):
#         The image of the cell labels.

#     Returns:
#     label_img_ar (np.array):
#         The image of the cell labels where the unconnected regions have been removed.
#     """
    
#     #Loop over the labels in the image
#     for label_id in np.unique(label_img_ar):
#         if label_id == 0: continue #The 0 label is for the background

#         #Only select the part of the image that corresponds to the cell label
#         masked_ar = np.ma.masked_where(label_img_ar == label_id, label_img_ar).mask.astype(int)

#         #Check if they are different regions of the cell label
#         labels_out = cc3d.connected_components(masked_ar, connectivity= 6)
#         unique_labels, label_counts = np.unique(labels_out, return_counts = True)

#         #Select the cell lable (!= 0) that has the most voxels
#         max_label = unique_labels[np.argmax(label_counts[1:]) + 1]

#         #Select all the voxels that do not belong to the biggest region of the cell label
#         voxel_to_remove = np.argwhere((labels_out != max_label) & (labels_out != 0))

#         #Remove the voxels that do not belong to the biggest region of the cell label in the orginal image
#         label_img_ar[voxel_to_remove[:,0], voxel_to_remove[:,1], voxel_to_remove[:,2]] = 0

#     return label_img_ar

#-------------------------------------------------------------------------------------------------------------





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





#-----------------------------------------------------------------------------------------------------------------------------

#######################################################################################################################
# Removing peripheral labels:
# Naive implementation: Removing all labels touching the background
# Issue: Very lossy, removes labels that are not always peripheral
# Benefit: Solid statistics obtained from the remaining labels
#######################################################################################################################

def remove_labels_touching_background(labeled_img, threshold=10):
    """
    Remove all labels that touch the background (label 0) more than the specified threshold.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The input 3D labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.

    threshold: int (default: 10)
        The minimum number of background pixels a label must touch to be removed.

    Returns:
    --------
    filtered_labeled_img: (np.array, 3D)
        The filtered 3D labeled image, where labels touching the background more than the threshold have been removed.
    """
    # Find the unique labels in the labeled image
    unique_labels = np.unique(labeled_img)

    # Pad the input labeled image with a single layer of background pixels (label 0)
    padded_labeled_img = np.pad(labeled_img, pad_width=10, mode='constant', constant_values=0)

    filtered_padded_labeled_img = np.copy(padded_labeled_img)

    # Iterate through the unique labels, excluding the background label (0)
    for label in unique_labels[1:]:
        # Create a binary image for the current label
        binary_img = padded_labeled_img == label

        # Dilate the binary image by one voxel to find the border of the label
        dilated_binary_img = ndimage.binary_dilation(binary_img)

        # Find the border by XOR operation between the dilated and original binary images
        border_binary_img = dilated_binary_img ^ binary_img

        # Count the number of background pixels (label 0) touching the border
        background_touch_count = np.sum(padded_labeled_img[border_binary_img] == 0)

        # Check if the background touch count is greater than or equal to the threshold
        if background_touch_count >= threshold:
            # If so, remove the label from the filtered padded labeled image
            filtered_padded_labeled_img[binary_img] = 0
    
    # Unpad the filtered labeled image to restore its original shape
    filtered_labeled_img = filtered_padded_labeled_img[10:-10, 10:-10, 10:-10]

    return filtered_labeled_img
#-------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------

#######################################################################################################################
# Removing peripheral labels:
# Improved implementation: Removing all labels touching the edges of the image
# Issue: Can lead to statistics that are poor due to the insufficient removal of cells
# that may be beside a cell that was not segmented properly
# Benefit: Less lossy, more cells are conserved
#######################################################################################################################

def remove_labels_touching_edges(labeled_img):
    """
    Remove all labels that touch the edge of the image.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The input 3D labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.

    Returns:
    --------
    filtered_labeled_img: (np.array, 3D)
        The filtered 3D labeled image, where labels touching the edges have been removed.
    """
    # Create a copy of the labeled image to store the filtered output
    filtered_labeled_img = labeled_img.copy()

    # Get the dimensions of the image
    image_shape = np.array(labeled_img.shape)

    # Calculate the regions and their properties
    regions = regionprops(labeled_img)

    # Iterate through the regions
    for region in regions:
        # Get the bounding box of the region
        min_slice, minr, minc, max_slice, maxr, maxc = region.bbox

        # Check if the bounding box touches the edge of the image
        if (minr == 0 or minc == 0 or maxr == image_shape[0] or maxc == image_shape[1]):
            # If so, remove the label from the filtered labeled image
            filtered_labeled_img[labeled_img == region.label] = 0

    return filtered_labeled_img
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def renumber_labels(labeled_img):
    """
    Renumber the labels in the input labeled image, such that the label values start from 1 
    and end at the number of unique labels, excluding the background (label 0).

    Parameters:
    -----------
    labeled_img: (np.array)
        The input labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.

    Returns:
    --------
    renumbered_labeled_img: (np.array)
        The renumbered labeled image, where label values start from 1 and end at the number of
        unique labels, excluding the background (label 0).
    """

    # Create a copy of the input labeled image to store the renumbered output
    renumbered_labeled_img = np.copy(labeled_img)

    # Find the unique labels in the labeled image, excluding the background label (0)
    unique_labels = np.unique(labeled_img)[1:]

    # Iterate through the unique labels, renumbering them sequentially
    for new_label, old_label in enumerate(unique_labels, start=1):
        renumbered_labeled_img[labeled_img == old_label] = new_label

    return renumbered_labeled_img
#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
def extend_labels(labeled_img, erosion_iterations=1, dilation_iterations=2, radius = 5):
    """
    Extends the labels in a 3D labeled image to ensure that they touch one another.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.

    iterations: (int, optional, default=1)
        The number of dilation iterations to perform. A higher number will result in 
        more extended labels.

    Returns:
    --------
    extended_labeled_img: (np.array, 3D)
        A 3D labeled image with extended labels.
    """
    
    unique_labels = np.unique(labeled_img)
    extended_labeled_img = labeled_img.copy()
    
    for label in unique_labels:
        if label == 0:
            continue
        
        # Get the binary mask for the current label
        binary_mask = (labeled_img == label).astype(np.uint8)

        # Erode the binary mask first
        eroded_mask = ndimage.binary_erosion(binary_mask, iterations=erosion_iterations)

        # Dilate the eroded mask
        dilated_mask = ndimage.binary_dilation(eroded_mask, iterations=dilation_iterations)
        
        # Fill only the background pixels in the dilated mask
        background_mask = (extended_labeled_img == 0)
        label_mask = np.logical_and(dilated_mask, background_mask)
        extended_labeled_img[label_mask] = label
        
    return extended_labeled_img
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def convert_cell_labels_to_meshes(
    img: np.array,
    voxel_resolution: np.array,
    smoothing_iterations: int = 1,
    output_directory='output',
    overwrite=False,
    pad_width=10
):
    """
    Convert the labels of the cells in the 3D segmented image to triangular meshes. Please make sure that the 
    image has correctly been segmented, i.e. that there is no artificially small cells, or cell labels 
    split in different parts/regions.

    Parameters:
    -----------
        img (np.ndarray, 3D):
            The 3D segmented image.

        voxel_resolution (np.ndarray):
            The voxel side lengths in microns in the order [x, y, z]

        smoothing_iterations (int):
            The number of smoothing iterations to perform on the surface meshes

        preprocess (bool):
            If True, do not add padding to the image, otherwise add padding

        output_directory (str, optional):
            Name of the folder where the cell_meshes will be saved

        overwrite (bool, optional):
            If True, overwrite existing mesh files

    Returns:
    --------
        mesh_lst (list):
            A list of triangular meshes, in the trimesh format.
    
    """
    

    img_padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    mesh_lst = []
    label_ids = np.unique(img_padded)
    
    meshes_folder = os.path.join(output_directory, 'cell_meshes')
    if not os.path.exists(meshes_folder):
        os.makedirs(meshes_folder)

    for label_id in tqdm(label_ids, desc="Converting labels to meshes"):
        if label_id == 0: continue

        cell_mesh_file = os.path.join(meshes_folder, f"cell_{label_id - 1}.stl")

        if not os.path.isfile(cell_mesh_file) or overwrite:
            surface = label_to_surface(img_padded == label_id)
            points, faces = surface[0], surface[1]
            points = (points - np.array([pad_width, pad_width, pad_width])) * voxel_resolution

            cell_surface_mesh = tm.Trimesh(points, faces)
            cell_surface_mesh = tm.smoothing.filter_mut_dif_laplacian(cell_surface_mesh, iterations=smoothing_iterations, lamb=0.5)
            cell_surface_mesh.export(cell_mesh_file)  # save the mesh as an STL file

        else:
            cell_surface_mesh = tm.load_mesh(cell_mesh_file)
        mesh_lst.append(cell_surface_mesh)

    return mesh_lst
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def process_labels(labeled_img, erosion_iterations=1, dilation_iterations=2, output_directory='output', overwrite=False):
    """
    Process the labels by removing unconnected regions, renumbering the labels, and extending them.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The results of the segmentation, where the background has a label of 
        0 and the cells are labeled with consecutive integers starting from 1. The 3D image 
        is assumed to be in the standard numpy format (x, y, z).

    erosion_iterations: (int, optional, default=1)
        Number of iterations for erosion during label extension

    dilation_iterations: (int, optional, default=2)
        Number of iterations for dilation during label extension

    output_directory: (str, optional, default='output')
        Name of the folder where the processed labels will be saved

    overwrite: (bool, optional, default=False)
        If True, overwrite the existing processed_labels.npy file

    Returns:
    --------
    preprocessed_labels: (np.array, 3D)
        The processed labels
    """

    # Check if processed_labels file exists
    processed_labels_file = os.path.join(output_directory, 'processed_labels.npy')

    if os.path.exists(processed_labels_file) and not overwrite:
        # Load the existing processed_labels
        preprocessed_labels = np.load(processed_labels_file)
    else:
        # Generate processed_labels
        unconnected_labels = remove_unconnected_regions(labeled_img)
        renumbered_labeled_img = renumber_labels(unconnected_labels)
        preprocessed_labels = extend_labels(renumbered_labeled_img, erosion_iterations=erosion_iterations, dilation_iterations=dilation_iterations)

        # Save the processed_labels
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        np.save(processed_labels_file, preprocessed_labels)

    return preprocessed_labels
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def load_or_create_filtered_labels(preprocessed_labels: np.ndarray, cell_volumes: list, output_directory: str, overwrite: bool, volume_lower_threshold: int, volume_upper_threshold: int):
    """
    Load or create labels filtered by volume threshold.

    Parameters:
        preprocessed_labels: The preprocessed labels.
        cell_volumes: A list of cell volumes.
        output_directory: The directory to store the filtered labels.
        overwrite: If True, overwrite the filtered labels if they exist.
        volume_lower_threshold: The minimum volume to include a cell in the filtered labels.
        volume_upper_threshold: The maximum volume to include a cell in the filtered labels.

    Returns:
        filtered_labels: The labels after filtering.
    """
    
    filtered_labels_path = os.path.join(output_directory, 'filtered_labels.npy')
    
    if os.path.isfile(filtered_labels_path) and not overwrite:
        filtered_labels = np.load(filtered_labels_path)
    else:
        # Remove the cells that are touching the background, leaving all the "inner" cells
        # filtered_labeled_img = remove_labels_touching_background(preprocessed_labels, threshold=1)
        filtered_label_image = remove_labels_touching_edges(preprocessed_labels)
        filtered_labels = np.unique(filtered_label_image)[1:]
        
        # Filter based on volume thresholds
    if volume_lower_threshold is not None or volume_upper_threshold is not None:
        filtered_labels = np.array([
            label for label in filtered_labels 
            if (volume_lower_threshold is None or (volume_lower_threshold is not None and cell_volumes[label - 1] >= volume_lower_threshold)) 
            and (volume_upper_threshold is None or (volume_upper_threshold is not None and cell_volumes[label - 1] <= volume_upper_threshold))
    ])
        np.save(filtered_labels_path, filtered_labels)
    
    return filtered_labels
#------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
def get_preprocessed_labels(labeled_img: np.ndarray, preprocess: bool, erosion_iterations: int, dilation_iterations: int, output_directory: str, overwrite: bool):
    """
    Get preprocessed labels from labeled image.

    Parameters:
        labeled_img: The labeled image to be preprocessed.
        preprocess: Boolean indicating whether to perform preprocessing.
        erosion_iterations: The number of erosion iterations in preprocessing.
        dilation_iterations: The number of dilation iterations in preprocessing.
        output_directory: The directory to store the preprocessed labels.
        overwrite: If True, overwrite the preprocessed labels if they exist.

    Returns:
        preprocessed_labels: The preprocessed labels.
    """
    if preprocess:
        preprocessed_labels = process_labels(labeled_img, erosion_iterations=erosion_iterations, 
                                             dilation_iterations=dilation_iterations, 
                                             output_directory=output_directory,
                                             overwrite=overwrite)
    else:
        preprocessed_labels = labeled_img
    return preprocessed_labels
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def full_label_processing(labeled_img, img_resolution, smoothing_iterations=5, erosion_iterations=1, dilation_iterations=2, 
                          output_folder='output', preprocess=True, overwrite=False, volume_lower_threshold=200, volume_upper_threshold=10000) -> tuple:
    """
    This function performs preprocessing of labeled image and returns a list of cell meshes, cell IDs, filtered cell IDs, cell neighbors, and the output directory.

    Parameters:
    - labeled_img: The input image, can be a string (file path) or a numpy array.
    - img_resolution: The resolution of the input image.
    - smoothing_iterations: The number of iterations for smoothing operation. Default is 5.
    - erosion_iterations: The number of iterations for erosion operation. Default is 1.
    - dilation_iterations: The number of iterations for dilation operation. Default is 2.
    - output_folder: The name of the output folder. Default is 'output'.
    - preprocess: If True, the input image will be preprocessed. If False, the image will be used as is. Default is True.
    - overwrite: If True, existing files will be overwritten. If False, existing files will not be overwritten. Default is False.
    - volume_threshold: The volume threshold to filter out smaller cells. Default is 200.

    Returns:
    - tuple: A tuple containing a list of cell meshes, cell IDs, filtered cell IDs, cell neighbors, and the output directory.
    """
    
    labeled_img = load_labeled_img(labeled_img)
    
    output_directory = create_output_directory(output_folder, '', smoothing_iterations, erosion_iterations,
                                               dilation_iterations)
    
    preprocessed_labels = get_preprocessed_labels(labeled_img, preprocess, erosion_iterations, dilation_iterations, output_directory, overwrite)
    
    cell_id_lst = np.unique(preprocessed_labels)[1:]

    mesh_lst = convert_cell_labels_to_meshes(preprocessed_labels, img_resolution, output_directory=output_directory, 
                                            smoothing_iterations=smoothing_iterations, overwrite=overwrite)

    cell_volumes = compute_cell_volumes(mesh_lst)

    filtered_cell_id_lst = load_or_create_filtered_labels(preprocessed_labels, cell_volumes, output_directory, overwrite, volume_lower_threshold, volume_upper_threshold=volume_upper_threshold)
    
    cell_neighbors_lst = [get_cell_neighbors(preprocessed_labels, cell_id) for cell_id in cell_id_lst]

    return mesh_lst, cell_id_lst, filtered_cell_id_lst, cell_neighbors_lst, output_directory
#--------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
def test_remove_unconnected_regions():
    # Create a 3D image with labeled regions
    labeled_img = np.zeros((20, 20, 20))
    
    # Add labeled regions
    labeled_img[5:10, 5:10, 5:10] = 1  # main cell body
    labeled_img[15:17, 15:17, 15:17] = 1  # unconnected region

    labeled_img[5:10, 5:10, 10:15] = 2  # main cell body
    labeled_img[0:2, 0:2, 0:2] = 2  # unconnected region

    # Run the function
    result = remove_unconnected_regions(labeled_img)

    # Check that the output has the expected properties
    assert np.unique(result).tolist() == [0, 1, 2], "All expected labels are not present in the output"
    assert np.sum(result == 1) == 5*5*5, "The size of the connected region with label 1 is not as expected"
    assert np.sum(result == 2) == 5*5*5, "The size of the connected region with label 2 is not as expected"

    # Check that unconnected regions have been removed
    assert not np.any(result[15:17, 15:17, 15:17]), "Unconnected region with label 1 was not removed"
    assert not np.any(result[0:2, 0:2, 0:2]), "Unconnected region with label 2 was not removed"
#--------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    test_remove_unconnected_regions()