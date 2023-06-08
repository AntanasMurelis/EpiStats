from typing import List, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm
import os
import warnings
from skimage.io import imread, imsave
import scipy.ndimage as ndimage
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
import pyclesperanto_prototype as cle
from napari_process_points_and_surfaces import label_to_surface
from misc import load_labeled_img, create_output_directory, custom_showwarning


#-----------------------------------------------------------------------------------------------------------------------------
def remove_unconnected_regions(
        labeled_img: np.ndarray[int], 
        warning_size_threshold: Optional[float] = 0.1
    ) -> np.ndarray[int]:
    """
    Removes regions of labels that are not connected to the main cell body.

    Parameters:
    -----------
    labeled_img: (np.array, 3D, dtype=int)
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.

    warning_size_threshold: (float, optional, default=0.1)
        Return a warning if the relative size of a removed region is greter than this threshold.

    Returns:
    --------
    filtered_labeled_img: (np.array, 3D, dtype=int)
        A 3D labeled image with unconnected regions removed.
    """

    warnings.showwarning = custom_showwarning
    
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
            relative_region_sizes = region_sizes / np.max(region_sizes)
            num_regions_over_threshold = np.sum(relative_region_sizes > warning_size_threshold) - 1
            if num_regions_over_threshold: 
                warnings.warn(f'Removing {num_regions_over_threshold} large regions with label {label} with threshold set at {warning_size_threshold}.')
            largest_region_label = np.argmax(region_sizes[1:]) + 1
            filtered_region = (labeled_mask == largest_region_label).astype(np.int8) * largest_region_label
            filtered_labeled_img[labeled_mask != filtered_region] = 0


    return filtered_labeled_img
#-----------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------------
def get_labels_touching_background(
        labeled_img: np.ndarray[int], 
        output_directory: Optional[str] = None,
        threshold: Optional[int] = 0
    ) -> Tuple[np.ndarray[int], Dict[int, int]]:
    """
    Remove all labels that touch the background (label 0) more than the specified threshold.

    Parameters:
    -----------
    labeled_img: (np.array, 3D, dtype=int)
        The input 3D labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.

    output_directory: (str)
        Name of the folder where the labels indexes will be saved.
        If None nothing will be saved.

    threshold: (int, optional, default=0)
        The minimum number of background pixels a label must touch to be removed.

    Returns:
    --------
    labels_touching_background: (np.ndarray, 1D, dtype=int)
        The label ids of the cells in touch with the background for at least 'threshold' voxels.
    
    background_touch_counts: (dict)
        A dictionary that associates to each label in 'labeled_img' the number of voxels in touch with the background.
    """
    # Find the unique labels in the labeled image
    unique_labels = np.unique(labeled_img)

    # Pad the input labeled image with a single layer of background pixels (label 0)
    padded_labeled_img = np.pad(labeled_img, pad_width=10, mode='constant', constant_values=0)

    # Initialize lists to store labels touching background and counts of background touching voxels
    labels_touching_background = []
    background_touch_counts = {} 

    # Iterate through the unique labels, excluding the background label (0)
    for label in tqdm(unique_labels[1:], desc="Checking labels touching background: "):
        # Create a binary image for the current label
        binary_img = padded_labeled_img == label

        # Dilate the binary image by one voxel to find the border of the label
        dilated_binary_img = ndimage.binary_dilation(binary_img)

        # Find the border by XOR operation between the dilated and original binary images
        border_binary_img = dilated_binary_img ^ binary_img

        # Count the number of background pixels (label 0) touching the border
        background_touch_count = np.sum(padded_labeled_img[border_binary_img] == 0)
        background_touch_counts[label] = background_touch_count

        # Check if the background touch count is greater than the threshold
        if background_touch_count > threshold:
            # Add label to the list of labels touching background
            labels_touching_background.append(label)
    
    # Convert labels list into numpy array
    labels_touching_background = np.asarray(labels_touching_background, dtype=np.uint16)
    
    if output_directory:
        if not os.path.exists(output_directory): os.makedirs(output_directory)
        np.savetxt(os.path.join(output_directory, "background_touching_labels.txt"), labels_touching_background)

    return labels_touching_background, background_touch_counts
#-------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------
def get_labels_touching_edges(
        labeled_img: np.ndarray[int], 
        output_directory: Optional[str] = None,
    ) -> np.ndarray[int]:
    """
    Remove all labels that touch the edge of the image.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The input 3D labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.

    output_directory: (str)
        Name of the folder where the labels indexes will be saved.
        If None nothing will be saved.

    Returns:
    --------
    labels_touching_edges: (np.ndarray, 1D)
        An array of labels of cells touching edges
    """
    # Get the dimensions of the image
    image_shape = np.array(labeled_img.shape)

    # Calculate the regions and their properties
    regions = regionprops(labeled_img)

    # List to store labels touching the edges
    labels_touching_edges = []

    # Iterate through the regions
    for region in tqdm(regions, total=len(regions), desc="Checking labels touching edges: "):
        # Get the bounding box of the region
        min_slice, minr, minc, max_slice, maxr, maxc = region.bbox

        # Check if the bounding box touches the edge of the image
        if (min_slice == 0 or minr == 0 or minc == 0 or max_slice == image_shape[0] or maxr == image_shape[1] or maxc == image_shape[2]):
            # Add label to the list of cut labels
            labels_touching_edges.append(region.label)
    
    # Convert labels list into numpy array
    labels_touching_edges = np.asarray(labels_touching_edges, dtype=np.uint16)
    
    if output_directory:
        if not os.path.exists(output_directory): os.makedirs(output_directory)
        np.savetxt(os.path.join(output_directory, "cut_cells_labels.txt"), labels_touching_edges)

    return labels_touching_edges
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def filter_labels(labeled_img: np.ndarray, labels_to_remove: list) -> np.ndarray:
    """
    Remove the selected indexes from a labeled image.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The input 3D labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.
    
    labels_to_remove: (list)
        The list of label indexes to remove from the input image.

    Returns:
    --------
    filtered_labeled_img: (np.array, 3D)
        A 3D labeled image with selected labels removed.
    """

    filtered_labeled_img = labeled_img.copy()

    for label in tqdm(labels_to_remove, desc="Removing labels: "):
        binary_mask = filtered_labeled_img == label
        filtered_labeled_img[binary_mask] = 0

    return filtered_labeled_img
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def extend_labels(
        labeled_img: np.ndarray, 
        erosion_iterations: Optional[int] = 1, 
        dilation_iterations: Optional[int] = 2
    ) -> np.ndarray:
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
    
    for label in tqdm(unique_labels, desc="Applying erosion and dilation"):
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
def smooth_labels(
        labeled_image: np.ndarray, 
        smoothing_radius: int
    ) -> np.ndarray:
    """
    Perform smoothing of a labeled image applying a morphological opening operation and afterwards
    fills gaps between the labels using voronoi-labeling.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The results of the segmentation, where the background has a label of 
        0 and the cells are labeled with consecutive integers starting from 1.

    smoothing_radius: (int)
        The radius of the morphological filter applied to smooth the labels.

    Returns:
    --------
    smoothed_labeled_img: (np.array, 3D)
        The labeled image after applying smoothing.
    """

    smoothed_labeled_img = cle.smooth_labels(labeled_image, radius=smoothing_radius)

    return smoothed_labeled_img

#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def process_labels(
        labeled_img: np.ndarray, 
        erosion_iterations: Optional[int] = None, 
        dilation_iterations: Optional[int] = None,
        smoothing_radius: Optional[int] = None, 
        output_directory: Optional[str] = 'output',
        overwrite: Optional[bool] = False
    ) -> np.ndarray:
    """
    Process the labels by removing unconnected regions, renumbering the labels, and extending via dilation/erosion steps.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The results of the segmentation, where the background has a label of 
        0 and the cells are labeled with consecutive integers starting from 1. The 3D image 
        is assumed to be in the standard numpy format (x, y, z).

    erosion_iterations: (int, optional, default=None)
        Number of iterations for erosion during label extension

    dilation_iterations: (int, optional, default=None)
        Number of iterations for dilation during label extension

    smoothing_radius: (int, optional, default=None)
        The radius of the morphological filter applied to smooth the labels.

    output_directory: (str, optional, default='output')
        Name of the folder where the processed labels will be saved

    overwrite: (bool, optional, default=False)
        If True, overwrite the existing processed_labels.npy file

    Returns:
    --------
    preprocessed_labeled_img: (np.array, 3D)
        The processed labels
    """

    # Check if processed_labels file exists
    processed_labels_file = os.path.join(output_directory, 'processed_labels.tif')

    if os.path.exists(processed_labels_file) and not overwrite:
        # Load the existing processed_labels
        print("Loading previously preprocessed labels...")
        preprocessed_labeled_img = imread(processed_labels_file)
    else:
        # Generate processed_labels
        print("Preprocessing labels...")
        preprocessed_labeled_img = remove_unconnected_regions(labeled_img)
        preprocessed_labeled_img, _, _ = relabel_sequential(preprocessed_labeled_img)
        if smoothing_radius:
            preprocessed_labeled_img = smooth_labels(
                preprocessed_labeled_img, 
                smoothing_radius
            )
        if erosion_iterations and dilation_iterations:
            preprocessed_labeled_img = extend_labels(
                preprocessed_labeled_img, 
                erosion_iterations, 
                dilation_iterations
            )
        
        # Save the processed_labels
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        imsave(processed_labels_file, preprocessed_labeled_img)

    return preprocessed_labeled_img
#------------------------------------------------------------------------------------------------------------

