import numpy as np
import pandas as pd
import trimesh as tm
import trimesh
from scipy import ndimage
from napari_process_points_and_surfaces import label_to_surface
from os import path, _exit, getcwd, mkdir
from skimage import io
from tqdm import tqdm
from tests.CubeLatticeTest import *
from tests.SphereTest import *
import os
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import pyclesperanto_prototype as cle
from typing import Union, Optional
from pykdtree.kdtree import KDTree


"""
    The methods in this script gather statistics from the 3D segmented labeled images. 
"""


class ExtendedTrimesh(trimesh.Trimesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_kdtree = KDTree(self.vertices)


    def get_potential_contact_faces(self, other_mesh, distance):
        potential_contact_faces = []
        for face_index in range(len(other_mesh.faces)):
            face = other_mesh.vertices[other_mesh.faces[face_index]]
            centroid = np.mean(face, axis=0).reshape(1, -1)
            dist, idx = self.my_kdtree.query(centroid)
            if dist < distance:
                potential_contact_faces.append(face_index)
        return potential_contact_faces

    def calculate_contact_area(self, other_mesh, distance):
        contact_faces_indices = self.get_potential_contact_faces(other_mesh, distance)
        # Assuming contact area is just the sum of the areas of the contact faces
        contact_area = np.sum(other_mesh.area_faces[contact_faces_indices])
        return contact_area

# Centroid implementation:
# class ExtendedTrimesh(trimesh.Trimesh):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # calculate the centroid of each face and use them to build the KDTree
#         self.my_kdtree = KDTree(self.triangles_center)

#     def get_potential_contact_faces(self, other_mesh, distance):
#         potential_contact_faces = []
#         for face_index in range(len(other_mesh.faces)):
#             face_centroid = other_mesh.triangles_center[face_index].reshape(1, -1)
#             dist, idx = self.my_kdtree.query(face_centroid)
#             if dist < distance:
#                 potential_contact_faces.append(face_index)
#         return potential_contact_faces

#     def calculate_contact_area(self, other_mesh, distance):
#         contact_faces_indices = self.get_potential_contact_faces(other_mesh, distance)
#         contact_area = np.sum(other_mesh.area_faces[contact_faces_indices])
#         return contact_area



    # def calculate_contact_area(self, other_mesh, distance):
    #     contact_faces_indices = self.get_potential_contact_faces(other_mesh, distance)
    #     contact_faces = other_mesh.faces[contact_faces_indices]
        
    #     # Create vertices, faces and values lists for napari
    #     vertices = other_mesh.vertices
    #     faces = contact_faces
    #     values = np.arange(len(faces))
        
    #     # Assuming contact area is just the sum of the areas of the contact faces
    #     contact_area = np.sum(other_mesh.area_faces[contact_faces_indices])
        
    #     # Visualize the contact faces using Napari
    #     with napari.gui_qt():
    #         viewer = napari.Viewer()
    #         viewer.add_surface((self.vertices, self.faces), name='target')
    #         viewer.add_surface((vertices, faces), colormap='red', name='contact_faces')
    #         viewer.add_surface((other_mesh.vertices, other_mesh.faces), name='neighbour')
        
    #     return contact_area
    
#------------------------------------------------------------------------------------------------------------
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
            filtered_labeled_img[labeled_mask != filtered_region] = 0


    return filtered_labeled_img

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
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
# def remove_labels_touching_background(labeled_img, threshold=10):
#     """
#     Remove all labels that touch the background (label 0) more than the specified threshold.

#     Parameters:
#     -----------
#     labeled_img: (np.array, 3D)
#         The input 3D labeled image, where the background has a label of 0 and other objects have 
#         positive integer labels.

#     threshold: int (default: 10)
#         The minimum number of background pixels a label must touch to be removed.

#     Returns:
#     --------
#     filtered_labeled_img: (np.array, 3D)
#         The filtered 3D labeled image, where labels touching the background more than the threshold have been removed.
#     """
#     # Find the unique labels in the labeled image
#     unique_labels = np.unique(labeled_img)

#     # Pad the input labeled image with a single layer of background pixels (label 0)
#     padded_labeled_img = np.pad(labeled_img, pad_width=10, mode='constant', constant_values=0)

#     # Create a copy of the padded labeled image to store the filtered output
#     filtered_padded_labeled_img = np.copy(padded_labeled_img)

#     # Iterate through the unique labels, excluding the background label (0)
#     for label in unique_labels[1:]:
#         # Create a binary image for the current label
#         binary_img = padded_labeled_img == label

#         # Dilate the binary image by one voxel to find the border of the label
#         dilated_binary_img = ndimage.binary_dilation(binary_img)

#         # Find the border by XOR operation between the dilated and original binary images
#         border_binary_img = dilated_binary_img ^ binary_img

#         # Count the number of background pixels (label 0) touching the border
#         background_touch_count = np.sum(padded_labeled_img[border_binary_img] == 0)

#         # Check if the background touch count is greater than or equal to the threshold
#         if background_touch_count >= threshold:
#             # If so, remove the label from the filtered padded labeled image
#             filtered_padded_labeled_img[binary_img] = 0
    
#     # Unpad the filtered labeled image to restore its original shape
#     filtered_labeled_img = filtered_padded_labeled_img[10:-10, 10:-10, 10:-10]

#     return filtered_labeled_img

from skimage.measure import label, regionprops

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
    
    # Flood fill the gaps between labels
    extended_labeled_img = np.asarray(cle.closing_labels(extended_labeled_img, radius=radius))
    
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






#------------------------------------------------------------------------------------------------------------
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
    
    if cell_neighbors_lst.__len__() == 0:
        return 0, [], 0

    cell_mesh = cell_mesh_lst[cell_id - 1]
    cell_mesh = ExtendedTrimesh(cell_mesh.vertices, cell_mesh.faces)
    
    contact_area_total = 0
    contact_area_distribution = []
    # Loop over the neighbors of the cell
    for neighbor_id in cell_neighbors_lst:
        # Get potential contact faces using the get_potential_contact_faces method from ExtendedTrimesh class
        neighbour_mesh = ExtendedTrimesh(cell_mesh_lst[neighbor_id - 1].vertices, cell_mesh_lst[neighbor_id - 1].faces)
        contact_area = cell_mesh.calculate_contact_area(neighbour_mesh, contact_cutoff)
        contact_area_total += contact_area
        contact_area_distribution.append(contact_area)

    # Calculate the fraction of contact area
    contact_fraction = contact_area_total / cell_mesh.area

    # Calculate the mean of contact area
    mean_contact_area = np.mean(contact_area_distribution)

    return contact_fraction, contact_area_distribution, mean_contact_area
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
        #labeled_img = np.pad(labeled_img, 10, mode='constant', constant_values=0)
        unconnected_labels = remove_unconnected_regions(labeled_img)
        renumbered_labeled_img = renumber_labels(unconnected_labels)
        preprocessed_labels = extend_labels(renumbered_labeled_img, erosion_iterations=erosion_iterations, dilation_iterations=dilation_iterations)
        #  Remove the padding:
        #preprocessed_labels = preprocessed_labels[10:-10, 10:-10, 10:-10]
        
        # Save the processed_labels
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        np.save(processed_labels_file, preprocessed_labels)

    return preprocessed_labels
#------------------------------------------------------------------------------------------------------------







#------------------------------------------------------------------------------------------------------------
def generate_plots(input_data, columns_to_plot=None, plot_type='violin', output_folder: str = None, exclude_columns=None):
    """
    Generate plots for the provided columns using the exported CSV file or the pandas DataFrame.

    Parameters:
    -----------
    input_data: (str or pd.DataFrame)
        Path to the exported CSV file containing cell statistics or a pandas DataFrame containing the cell statistics.

    columns_to_plot: (list of str, optional)
        List of columns to plot. If None, all numeric columns from the input data will be plotted.

    plot_type: (str, optional, default='violin')
        Type of plot to generate. Supported types are 'violin', 'box', and 'histogram'.

    output_folder: (str, optional)
        Path to the folder where the plots will be saved. If None, the plots will be displayed but not saved.

    exclude_columns: (list of str, optional)
        List of columns to exclude from plotting. If None, all columns containing 'id' in their titles will be excluded.

    Returns:
    --------
    None
    """

    # Load the cell statistics DataFrame from a CSV file or use the input DataFrame
    if isinstance(input_data, str):
        data_df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        data_df = input_data
    else:
        raise ValueError("input_data should be either a str (path to CSV file) or a pandas DataFrame")

    # If columns_to_plot is None, plot all numeric columns except the excluded ones
    if columns_to_plot is None:
        columns_to_plot = data_df.select_dtypes(include=['number']).columns

    # If exclude_columns is None, exclude all columns containing 'id' in their titles
    if exclude_columns is None:
        exclude_columns = [col for col in columns_to_plot if 'id' in col]

    # Exclude specified columns and filter out columns with None values
    columns_to_plot = [col for col in columns_to_plot if col not in exclude_columns and data_df[col].notna().any()]

    sns.set(style="whitegrid")

    # Create subplots for each column
    num_plots = len(columns_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 10), sharey=False)
    fig.tight_layout(pad=6)

    colors = sns.color_palette("hls", num_plots)

    for i, col in enumerate(columns_to_plot):
        data = data_df[[col]]

        if plot_type == 'violin':
            sns.violinplot(data=data, orient="v", cut=0, inner="quartile", ax=axes[i], color=colors[i])
            sns.stripplot(data=data, color=".3", size=4, jitter=True, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(data=data, orient="v", ax=axes[i], color=colors[i])
        elif plot_type == 'histogram':
            sns.histplot(data=data, kde=True, ax=axes[i], color=colors[i])
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}. Supported types are 'violin', 'box', and 'histogram'.")

        axes[i].xaxis.set_tick_params(labelbottom=False)
        axes[i].set(xlabel=col)
        axes[i].set_title("", pad=-15)

    sns.despine(left=True)

    # Save the plot to a file or display it
    if output_folder is not None:
        plt.savefig(output_folder)
    else:
        plt.show()

    # Close all figures to release memory
    plt.close('all')
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def load_labeled_img(labeled_img: Union[str, np.ndarray]):
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
def create_output_directory(output_folder: str, smoothing_iterations: int, erosion_iterations: int, dilation_iterations: int):
    """
    Create a directory to store the output data.

    Parameters:
        output_folder: The root folder for storing the outputs.
        smoothing_iterations: The number of smoothing iterations in preprocessing.
        erosion_iterations: The number of erosion iterations in preprocessing.
        dilation_iterations: The number of dilation iterations in preprocessing.

    Returns:
        output_directory: The full path of the created output directory.
    """
    output_directory = f"{output_folder}_s_{smoothing_iterations}_e_{erosion_iterations}_d_{dilation_iterations}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory
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
def load_or_create_filtered_labels(preprocessed_labels: np.ndarray, cell_volumes: list, output_directory: str, overwrite: bool, volume_threshold: int):
    """
    Load or create labels filtered by volume threshold.

    Parameters:
        preprocessed_labels: The preprocessed labels.
        cell_volumes: A list of cell volumes.
        output_directory: The directory to store the filtered labels.
        overwrite: If True, overwrite the filtered labels if they exist.
        volume_threshold: The minimum volume to include a cell in the filtered labels.

    Returns:
        filtered_labels: The labels after filtering.
    """
    
    filtered_labels_path = os.path.join(output_directory, 'filtered_labels.npy')
    
    if os.path.isfile(filtered_labels_path) and not overwrite:
        filtered_labels = np.load(filtered_labels_path)
    else:
        # Remove the cells that are touching the background, leaving all the "inner" cells
        # filtered_labeled_img = remove_labels_touching_background(preprocessed_labels, threshold=1)
        filtered_labeled_img = remove_labels_touching_edges(preprocessed_labels)
        # Exclude cells with volumes below the volume threshold
        if volume_threshold is not None:
            unique_labels = np.unique(filtered_labeled_img)[1:]  # Exclude the zero label

            # Create a dictionary mapping labels to their volumes
            label_to_volume = dict(zip(unique_labels, cell_volumes))

            # Find labels of cells above the volume threshold
            cells_above_threshold = [label for label, volume in label_to_volume.items() if volume >= volume_threshold]

            filtered_labels = np.array(cells_above_threshold)
        
        np.save(filtered_labels_path, filtered_labels)
    
    return filtered_labels
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
#--------------------------------------------------------------------------------------------------





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
def generate_required_plots(plot: str, output_directory: str, cell_statistics_df: pd.DataFrame, filtered_cell_statistics: pd.DataFrame, plot_type: str):
    """
    This function generates the requested plots based on the specified plot type and saves them in the provided output directory.

    Parameters:
    - plot: A string indicating which plots to generate. Can be 'filtered', 'all', 'both' or None.
    - output_directory: The directory where the generated plots will be saved.
    - cell_statistics_df: A DataFrame containing statistics for all cells.
    - filtered_cell_statistics: A DataFrame containing statistics for only the filtered cells.
    - plot_type: The type of plot to generate. 

    No return value. The function saves the generated plots directly to the specified output directory.
    """
    
    if plot in ('filtered', 'both'):
        generate_plots(input_data = filtered_cell_statistics, plot_type = plot_type, output_folder = os.path.join(output_directory, 'filtered_cell_plots.png'))
    
    if plot in ('all', 'both'):
        generate_plots(input_data = cell_statistics_df, plot_type = plot_type,  output_folder = os.path.join(output_directory,'all_cell_plots.png'))
#--------------------------------------------------------------------------------------------------






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






#--------------------------------------------------------------------------------------------------
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
            Maximum number of workers for the ProcessPoolExecutor.

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
            cell_contact_area_futures = {
                executor.submit(
                    compute_cell_contact_area_fraction, mesh_lst, cell_id, cell_neighbors, contact_cutoff
                ): cell_id
                for cell_id, cell_neighbors in tqdm(
                    enumerate(cell_neighbors_lst, start=1),
                    desc="Computing contact area fraction",
                    total=len(cell_neighbors_lst),
                )
            }

            # Collect the results as they complete
            for future in tqdm(concurrent.futures.as_completed(cell_contact_area_futures), desc="Collecting results", total=len(cell_contact_area_futures)):
                cell_id = cell_contact_area_futures[future]
                try:
                    cell_contact_area_dict[cell_id] = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
    else:
        for cell_id in cell_id_lst:
            cell_contact_area_dict[cell_id] = (None, [], None)

    # Return the dictionary containing contact area fractions, contact area distributions, and mean contact areas
    return cell_contact_area_dict
#--------------------------------------------------------------------------------------------------





#--------------------------------------------------------------------------------------------------
def full_label_processing(labeled_img, img_resolution, smoothing_iterations=5, erosion_iterations=1, dilation_iterations=2, 
                          output_folder='output', preprocess=True, overwrite=False, volume_threshold=200) -> tuple:
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
    
    output_directory = create_output_directory(output_folder, smoothing_iterations, erosion_iterations, dilation_iterations)
    
    preprocessed_labels = get_preprocessed_labels(labeled_img, preprocess, erosion_iterations, dilation_iterations, output_directory, overwrite)
    
    cell_id_lst = np.unique(preprocessed_labels)[1:]

    mesh_lst = convert_cell_labels_to_meshes(preprocessed_labels, img_resolution, output_directory=output_directory, 
                                            smoothing_iterations=smoothing_iterations, overwrite=overwrite)

    cell_volumes = compute_cell_volumes(mesh_lst)

    filtered_cell_id_lst = load_or_create_filtered_labels(preprocessed_labels, cell_volumes, output_directory, overwrite, volume_threshold)
    
    cell_neighbors_lst = [get_cell_neighbors(preprocessed_labels, cell_id) for cell_id in cell_id_lst]

    return mesh_lst, cell_id_lst, filtered_cell_id_lst, cell_neighbors_lst, output_directory
#--------------------------------------------------------------------------------------------------





#--------------------------------------------------------------------------------------------------
def calculate_all_statistics(mesh_lst, cell_id_lst, cell_neighbors_lst, filtered_cell_id_lst, contact_cutoff, output_directory, 
                            max_workers=None, calculate_contact_area_fraction=False, plot=None, plot_type='violin') -> pd.DataFrame:
    """
    This function calculates a set of statistics for a given list of cell meshes.

    Parameters:
    - mesh_lst: A list of cell meshes.
    - cell_id_lst: A list of cell IDs.
    - cell_neighbors_lst: A list of cell neighbors.
    - filtered_cell_id_lst: A list of filtered cell IDs.
    - contact_cutoff: The cutoff value for contact.
    - output_directory: The directory where output files will be saved.
    - max_workers: The maximum number of worker threads. If None, the number of CPUs will be used.
    - calculate_contact_area_fraction: If True, the contact area fraction will be calculated. If False, it will not be calculated.
    - plot: The plot to generate. Can be 'filtered', 'all', 'both' or None.
    - plot_type: The type of plot to generate. Can be 'violin' or other types. Default is 'violin'.

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
                                          plot_type: str = 'violin', volume_threshold: int = 200, 
                                          *args, **kwargs) -> pd.DataFrame:

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
    -----------
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

    Returns:
    --------
    filtered_cell_statistics: (pd.DataFrame)
        A pandas dataframe containing the filtered cell statistics (cells not touching the background)
        
    Example:
    --------
    
    
    ```python
    import numpy as np
    
    #Load the labeled image (numpy array)
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
        calculate_contact_area_fraction=True
    )

    print(filtered_cell_statistics)
    ```
    """
    
    mesh_lst, cell_id_lst, filtered_cell_id_lst, cell_neighbors_lst, output_directory = full_label_processing(
        labeled_img, img_resolution, smoothing_iterations, erosion_iterations, dilation_iterations, 
        output_folder, preprocess, overwrite, volume_threshold)
    
    if meshes_only:
        return
    
    cell_statistics_df = calculate_all_statistics(mesh_lst, cell_id_lst, cell_neighbors_lst, filtered_cell_id_lst, 
                                                  contact_cutoff, output_directory, max_workers, calculate_contact_area_fraction, 
                                                  plot, plot_type)

    return cell_statistics_df
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Test the mesh generation part by meshing of the test lattice cube images. 
    img = generate_cube_lattice_image(
        nb_x_voxels=200,
        nb_y_voxels=200,
        nb_z_voxels=200,
        cube_side_length=5,
        nb_cubes_x=5,
        nb_cubes_y=5,
        nb_cubes_z=5,
        interstitial_space=-1,
    )

    
    # path = "//Users/antanas/BC_Project/Control_Segmentation_final/Total_meshes/Validated_labels_final_copy"
    
    # Import tiff file
    # labels = io.imread('/Users/antanas/BC_Project/Control_Segmentation/Validated_labels_final.tif')
    # img = np.einsum('kij->ijk', labels)
    
    #Collect the statistics of the cells
    
    cell_statistics_df = collect_cell_morphological_statistics(img, np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.8, output_folder="/Users/antanas/BC_Project/Control_Segmentation/Test", meshes_only=False, overwrite=True,
                                                               smoothing_iterations=5, erosion_iterations=2, dilation_iterations=3, calculate_contact_area_fraction=True, plot = 'all', plot_type = 'violin', preprocess = True, max_workers=None)
    
    # cell_statistics_df = collect_cell_morphological_statistics(labeled_img=img, img_resolution = np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.5, clear_meshes_folder=False, output_folder="./Test_1000", preprocess = False, meshes_only=False, overwrite=False,
    #                                                           smoothing_iterations=5, erosion_iterations=2, dilation_iterations=5, max_workers=4, calculate_contact_area_fraction=True, plot = 'all', plot_type = 'violin')
    # print(cell_statistics_df)
    # generate_plots(input_data = '/Users/antanas/GitRepo/EpiStats/all_cell_statistics.csv', plot_type='violin')
    
    # cell_statistics_df = collect_cell_morphological_statistics(labeled_img = 'path to .tif or 3D array of your image', img_resolution = np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.2, clear_meshes_folder=True, 
    #                                                             output_folder="Cube_test", preprocess = False, meshes_only=False, overwrite=True, 
    #                                                             max_workers=4, calculate_contact_area_fraction=True, plot = 'all', plot_type = 'violin')
    
    # generate_plots(input_data = '/Users/antanas/BC_Project/Control_Segmentation_final/BC_control_2_s_8_e_3_d_4/filtered_cell_statistics.csv', plot_type='violin')