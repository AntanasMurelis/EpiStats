import numpy as np
import pandas as pd
import trimesh as tm
from scipy import ndimage
from napari_process_points_and_surfaces import label_to_surface
from os import path, _exit, getcwd, mkdir
from skimage import io
from tqdm import tqdm
from tests.CubeLatticeTest import *
from tests.SphereTest import *
import os
from multiprocessing import Pool, cpu_count
import concurrent.futures
import gc
import matplotlib.pyplot as plt
import seaborn as sns


"""
    The methods in this script gather statistics from the 3D segmented labeled images. 
"""

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
    
    # Pad the labeled image
    padded_labeled_img = np.pad(labeled_img, pad_width=pad_width, mode='constant', constant_values=0)
    filtered_labeled_img = padded_labeled_img.copy()

    for label in tqdm(unique_labels, desc='Removing unconnected regions'):
        if label == 0:
            continue
        binary_mask = (padded_labeled_img == label).astype(np.uint8)

        # Label connected regions
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        # Remove unconnected regions
        if num_features > 1:
            region_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
            largest_region_label = np.argmax(region_sizes[1:]) + 1
            filtered_region = labeled_mask == largest_region_label
            filtered_labeled_img[labeled_mask != filtered_region] = 0


    return filtered_labeled_img
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------

def remove_labels_touching_background(labeled_img):
    """
    Remove all labels that are touching the label 0 (background) from the input labeled image.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The input 3D labeled image, where the background has a label of 0 and other objects have 
        positive integer labels.

    Returns:
    --------
    filtered_labeled_img: (np.array, 3D)
        The filtered 3D labeled image, where labels that were touching the background have been removed.
    """
    # Find the unique labels in the labeled image
    unique_labels = np.unique(labeled_img)

    # Pad the input labeled image with a single layer of background pixels (label 0)
    padded_labeled_img = np.pad(labeled_img, pad_width=10, mode='constant', constant_values=0)

    # Create a copy of the padded labeled image to store the filtered output
    filtered_padded_labeled_img = np.copy(padded_labeled_img)

    # Iterate through the unique labels, excluding the background label (0)
    for label in unique_labels[1:]:
        # Create a binary image for the current label
        binary_img = padded_labeled_img == label

        # Dilate the binary image by one voxel to find the border of the label
        dilated_binary_img = ndimage.binary_dilation(binary_img)

        # Find the border by XOR operation between the dilated and original binary images
        border_binary_img = dilated_binary_img ^ binary_img

        # Check if the border is touching the background (label 0)
        if np.any(padded_labeled_img[border_binary_img] == 0):
            # If it's touching the background, remove the label from the filtered padded labeled image
            filtered_padded_labeled_img[binary_img] = 0
    
    # Unpad the filtered labeled image to restore its original shape
    filtered_labeled_img = filtered_padded_labeled_img[10:-10, 10:-10, 10:-10]

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
def extend_labels(labeled_img, erosion_iterations=1, dilation_iterations=2):
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

    for label in tqdm(unique_labels, desc='Extending labels'):
        if label == 0:
            continue
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
    img,
    voxel_resolution,
    smoothing_iterations=1,
    preprocess=False,
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
    label_ids = np.unique(img)

    meshes_folder = os.path.join(output_directory, 'cell_meshes')
    if not os.path.exists(meshes_folder):
        os.makedirs(meshes_folder)

    for label_id in tqdm(label_ids, 
                         desc="Converting labels to meshes"):
        if label_id == 0: continue

        cell_mesh_file = os.path.join(meshes_folder, f"cell_{label_id - 1}.stl")

        if not os.path.isfile(cell_mesh_file) or overwrite:
            surface = label_to_surface(img_padded == label_id)
            points, faces = surface[0], surface[1]
            points = (points - np.array([pad_width, pad_width, pad_width])) * voxel_resolution

            cell_surface_mesh = tm.Trimesh(points, faces)
            cell_surface_mesh = tm.smoothing.filter_laplacian(cell_surface_mesh, iterations=smoothing_iterations)
        else:
            cell_surface_mesh = tm.load_mesh(cell_mesh_file)

        mesh_lst.append(cell_surface_mesh)

    return mesh_lst
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def compute_cell_areas(cell_mesh_lst):
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
def compute_cell_volumes(cell_mesh_lst):
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
def get_cell_neighbors(labeled_img, cell_id):
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

    cell_mesh_lst: (list, tm.Trimesh)
        List of the cell meshes in the standard trimesh format

    cell_id: (int)
        The id of the cell for which we want to calculate the contact area fraction

    cell_neighbors_lst: (list, int)
        List of the ids of the neighbors of the cell

    contact_cutoff: (float)
        The cutoff distance in microns for two cells to be considered in contact

    Returns
    -------
        contact_fraction: (float)
    """


    # Keep track of all the faces of the given cell that are in contact with other cells
    contact_faces = []

    cell_mesh = cell_mesh_lst[cell_id - 1]
    assert (isinstance(cell_mesh, tm.Trimesh))

    # Loop over the neighbors of the cell
    for neighbor_id in cell_neighbors_lst:
        # Find the closest distance between the center of the given cell's triangles and the triangles of the neighbor cells
        closest_point, distance, triangle_id = tm.proximity.closest_point(cell_mesh_lst[neighbor_id - 1],
                                                                          cell_mesh.triangles_center)

        # Get the indices of the triangles whose centroids are located at distances shorter than the cutoff distance from the neighbor cell surface
        lateral_triangle_id = np.where(distance <= contact_cutoff)[0]
        contact_faces.append(lateral_triangle_id)

    # Create a unique list of faces that are in contact with other cells
    contact_faces = np.unique(np.concatenate(contact_faces))

    # sum the area of all the faces that are in contact with other cells
    contact_area = np.sum(cell_mesh_lst[cell_id - 1].area_faces[contact_faces])

    # store all the contact faces in a mesh
    #contact_mesh = tm.Trimesh(cell_mesh_lst[cell_id - 1].vertices, cell_mesh_lst[cell_id - 1].faces[contact_faces])
    #contact_mesh.export('contact_mesh.stl')

    # Calculate the fraction of contact area
    contact_fraction = contact_area / cell_mesh_lst[cell_id - 1].area

    return contact_fraction
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def process_labels(labeled_img, erosion_iterations=1, dilation_iterations=2, output_directory='output', overwrite=False, renumber=False):
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

    renumber: (bool, optional, default=False)
        If True, renumber the labels after removal of unconnected regions

    Returns:
    --------
    preprocessed_labels: (np.array, 3D)
        The processed labels
    """

    # Check if processed_labels file exists
    file_name = 'processed_labels_er{}_dil{}.npy'.format(erosion_iterations, dilation_iterations)
    processed_labels_file = os.path.join(output_directory, file_name)

    if os.path.exists(processed_labels_file) and not overwrite:
        # Load the existing processed_labels
        preprocessed_labels = np.load(processed_labels_file)
    else:
        # Generate processed_labels
        unconnected_labels = remove_unconnected_regions(labeled_img, pad_width=10)
        if renumber:
            updated_labels = renumber_labels(unconnected_labels)
        else:
            updated_labels = unconnected_labels     
        preprocessed_labels = extend_labels(updated_labels, 
                                            erosion_iterations=erosion_iterations, 
                                            dilation_iterations=dilation_iterations)
        #  Remove the padding:
        preprocessed_labels = preprocessed_labels[10:-10, 10:-10, 10:-10]
        
        # Save the processed_labels
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        np.save(processed_labels_file, preprocessed_labels)

    return preprocessed_labels
#------------------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------------------
def save_meshes(mesh_lst, filtered_cell_id_lst, output_directory='output', clear_meshes_folder=False, overwrite=False):
    """
    Save the cell meshes to the specified folders.

    Parameters:
    -----------
    mesh_lst: (list)
        A list of cell meshes

    filtered_cell_id_lst: (list)
        A list of cell IDs that are not touching the background

    output_directory: (str, optional, default='output')
        Name of the folder where the cell_meshes and filtered_cell_meshes will be saved

    clear_meshes_folder: (bool, optional, default=False)
        If True, delete all the current meshes in the "cell_meshes" and "filtered_cell_meshes" directories before saving new meshes

    overwrite: (bool, optional, default=False)
        If True, overwrite existing mesh files

    Returns: None
    --------
    """

    # Create subfolders for cell_meshes and filtered_cell_meshes
    meshes_folder = os.path.join(output_directory, 'cell_meshes')
    filtered_meshes_folder = os.path.join(output_directory, 'filtered_cell_meshes')

    for folder_name in [meshes_folder, filtered_meshes_folder]:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        elif clear_meshes_folder:
            # Delete all files in the directory
            for filename in os.listdir(folder_name):
                file_path = os.path.join(folder_name, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    # Save the meshes of the cells to the folder
    for cell_id, mesh in enumerate(mesh_lst):
        mesh_file = os.path.join(meshes_folder, f"cell_{cell_id}.stl")
        filtered_mesh_file = os.path.join(filtered_meshes_folder, f"filtered_cell_{cell_id}.stl")

        if not os.path.isfile(mesh_file) or overwrite:
            mesh.export(mesh_file)
        
        if (cell_id + 1) in filtered_cell_id_lst and (not os.path.isfile(filtered_mesh_file) or overwrite):
            mesh.export(filtered_mesh_file)
#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def compute_cell_contact_area_fraction_helper(args):
    mesh_lst, cell_id, cell_neighbors, contact_cutoff = args
    return compute_cell_contact_area_fraction(mesh_lst, cell_id, cell_neighbors, contact_cutoff)
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
def collect_cell_morphological_statistics(labeled_img, img_resolution, contact_cutoff, clear_meshes_folder=False,
                                          smoothing_iterations=5, erosion_iterations=1, dilation_iterations=2,
                                          output_folder='output', meshes_only=False, overwrite=False, preprocess=True, 
                                          max_workers=None, calculate_contact_area_fraction = False, plot=None, plot_type='violin',
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
    
    # Load labeled image if it's a string
    if isinstance(labeled_img, str) and os.path.isfile(labeled_img):
        labels = io.imread(labeled_img)
        labeled_img = np.einsum('kij->ijk', labels)

    # Create a folder to store the processed labels, cell_meshes, and filtered_cell_meshes
    output_directory = f"{output_folder}_s_{smoothing_iterations}_e_{erosion_iterations}_d_{dilation_iterations}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Obtain preprocessed labels if preprocess is True
    if preprocess:
        preprocessed_labels = process_labels(labeled_img, erosion_iterations=erosion_iterations, 
                                             dilation_iterations=dilation_iterations, 
                                             output_directory=output_directory,
                                             overwrite=overwrite)
    else:
        preprocessed_labels = labeled_img

    del labeled_img
    # Get the list of the ids of the cells in the image
    cell_id_lst = np.unique(preprocessed_labels)[1:]

    # Make sure the cell ids are consecutive integers starting from 1
    assert(np.all(cell_id_lst == np.arange(1, len(cell_id_lst) + 1)))

    # Convert the cell labels to meshes
    mesh_lst = convert_cell_labels_to_meshes(preprocessed_labels, img_resolution, output_directory=output_directory, smoothing_iterations=smoothing_iterations,
                                             preprocess=preprocess)

    # Check if filtered_labeled_img exists, and load it if it does
    filtered_labeled_img_path = os.path.join(output_directory, 'filtered_labeled_img.npy')
    if os.path.isfile(filtered_labeled_img_path):
        filtered_labeled_img = np.load(filtered_labeled_img_path)
    else:
        # Remove the cells that are touching the background, leaving all the "inner" cells
        filtered_labeled_img = remove_labels_touching_background(preprocessed_labels)
        np.save(filtered_labeled_img_path, filtered_labeled_img)
    
    
    filtered_cell_id_lst = np.unique(filtered_labeled_img)[1:]

    # Save the meshes
    save_meshes(mesh_lst, filtered_cell_id_lst, output_directory, clear_meshes_folder=clear_meshes_folder)
    
    if meshes_only:
        return
    
    # Get the cell areas
    cell_areas = compute_cell_areas(mesh_lst)

    # Get the cell volumes
    cell_volumes = compute_cell_volumes(mesh_lst)

    # Compute the isoperimetric ratio of each cell
    cell_isoperimetric_ratio = [(area**3) / (volume**2) for area, volume in zip(cell_areas, cell_volumes)]

    # Get the list of the neighbors of each cell using the new cell_id_list, but obtain neighbors from the original labeled_img
    cell_neighbors_lst = [get_cell_neighbors(preprocessed_labels, cell_id) for cell_id in cell_id_lst]

    # Delete the preprocessed labels to free up memory
    del preprocessed_labels
    gc.collect()

    # Get the number of neighbors of each cell
    cell_nb_of_neighbors = [len(cell_neighbors) for cell_neighbors in cell_neighbors_lst]

    # Get the principal axis and the elongation of each cell
    cell_principal_axis_lst = []
    cell_elongation_lst = []

    for cell_id in tqdm(cell_id_lst, desc='Computing cell principal axis and elongation'):
        principal_axis, elongation = compute_cell_principal_axis_and_elongation(mesh_lst[cell_id - 1])
        cell_principal_axis_lst.append(principal_axis)
        cell_elongation_lst.append(elongation)


# NON-PARALLELIZED VERSION
#----------------------------------------------------------------------------------------------------------------
    # Get the contact area fraction of each cell
    # cell_contact_area_fraction_lst = []
    # for cell_id in tqdm(cell_id_lst, desc='Computing contact area fraction'):
    #     if mesh_lst[cell_id - 1].volume > 0.8 * max(cell_volumes):
    #         contact_area_fraction = 0
    #     else:
    #         contact_area_fraction = compute_cell_contact_area_fraction(mesh_lst, cell_id, cell_neighbors_lst[cell_id - 1], contact_cutoff)
        
    #     cell_contact_area_fraction_lst.append(contact_area_fraction)
#----------------------------------------------------------------------------------------------------------------

# ALTERNATIVE PARALLELIZED VERSION
#----------------------------------------------------------------------------------------------------------------
    # Get the contact area fraction of each cell
    # if max_workers is None:
    #     max_workers = cpu_count()

    # with Pool(processes=max_workers-1) as pool:
    #     cell_contact_area_fraction_lst = pool.map(compute_cell_contact_area_fraction_helper, [(mesh_lst, cell_id, cell_neighbors, contact_cutoff) for cell_id, cell_neighbors in enumerate(cell_neighbors_lst, start=1)])  
#----------------------------------------------------------------------------------------------------------------   

    # Get the contact area fraction of each cell, if requested
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
        cell_contact_area_fraction_lst = []
        for future in tqdm(concurrent.futures.as_completed(cell_contact_area_futures), desc="Collecting results", total=len(cell_contact_area_futures)):
            cell_contact_area_fraction_lst.append(future.result())
    else:
        cell_contact_area_fraction_lst = [None] * len(cell_id_lst)
    
    #Reformat the principal axis list and the neighbor list to be in the format of a string
    to_scientific_str = lambda x: '{:.2e}'.format(x)
    cell_principal_axis_lst = [(" ").join(map(to_scientific_str, principal_axis.tolist())) for principal_axis in cell_principal_axis_lst]
    cell_neighbors_lst = [(" ").join(map(str, neighbors)) for neighbors in cell_neighbors_lst]

    #Create a pandas dataframe to store the cell statistics
    cell_statistics_df = pd.DataFrame(
        {
            'cell_id': cell_id_lst,
            'cell_area': cell_areas,
            'cell_volume': cell_volumes,
            'cell_isoperimetric_ratio': cell_isoperimetric_ratio,
            'cell_neighbors': cell_neighbors_lst,
            'cell_nb_of_neighbors': cell_nb_of_neighbors,
            'cell_principal_axis': cell_principal_axis_lst,
            'cell_elongation': cell_elongation_lst,
            'cell_contact_area_fraction': cell_contact_area_fraction_lst,
        }
    )
    
    # Filter the cell statistics to only include the cells that are not touching the background
    filtered_cell_statistics = cell_statistics_df[cell_statistics_df['cell_id'].isin(filtered_cell_id_lst)]
    
    # Save all cell statistics to a CSV file
    cell_statistics_df.to_csv(os.path.join(output_directory, "all_cell_statistics.csv"), index=False)

    # Save filtered cell statistics to a separate CSV file
    filtered_cell_statistics.to_csv(os.path.join(output_directory, "filtered_cell_statistics.csv"), index=False)
    
    if plot in ('filtered', 'both'):
        generate_plots(input_data = filtered_cell_statistics, plot_type = plot_type, output_folder = os.path.join(output_directory, 'filtered_cell_plots.png'))
    
    if plot in ('all', 'both'):
        generate_plots(input_data = cell_statistics_df, plot_type = plot_type,  output_folder = os.path.join(output_directory,'all_cell_plots.png'))
    
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
        nb_cubes_x=2,
        nb_cubes_y=2,
        nb_cubes_z=2,
        interstitial_space=-1,
    )

    
    # path = "/Users/antanas/BC_Project/Control_Segmentation"
    
    # # Import tiff file
    # labels = io.imread('/Users/antanas/BC_Project/Control_Segmentation/Validated_labels_final.tif')
    # img = np.einsum('kij->ijk', labels)
    
    #Collect the statistics of the cells
    
    # cell_statistics_df = collect_cell_morphological_statistics(img, np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.8, clear_meshes_folder=False, output_folder="/Users/antanas/BC_Project/Control_Segmentation/BC_control", meshes_only=False, overwrite=False,
    #                                                            smoothing_iterations=5, erosion_iterations=2, dilation_iterations=3)
    
    # cell_statistics_df = collect_cell_morphological_statistics(img, np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.2, clear_meshes_folder=True, output_folder="Cube_test", preprocess = False, meshes_only=False, overwrite=True,
                                                            #   smoothing_iterations=5, erosion_iterations=2, dilation_iterations=3, max_workers=4, calculate_contact_area_fraction=True, plot = 'all', plot_type = 'violin')
    # print(cell_statistics_df)
    # generate_plots(input_data = '/Users/antanas/GitRepo/EpiStats/all_cell_statistics.csv', plot_type='violin')
    
    cell_statistics_df = collect_cell_morphological_statistics(
        labeled_img = 'path to .tif or 3D array of your image', 
        img_resolution = np.array([0.21, 0.21, 0.39]), 
        contact_cutoff = 0.2, 
        clear_meshes_folder=True, 
        output_folder="Cube_test", 
        preprocess = False, 
        meshes_only=False, 
        overwrite=True,
        max_workers=4, 
        calculate_contact_area_fraction=True, 
        plot = 'all', 
        plot_type = 'violin')

