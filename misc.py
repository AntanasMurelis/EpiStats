import numpy as np
import pandas as pd
import trimesh as tm
import numpy as np
from os import getcwd, path, _exit, mkdir
import pandas as pd
from skimage import io
import cc3d
from skimage.morphology import binary_erosion
from skimage.segmentation import expand_labels
from napari_process_points_and_surfaces import label_to_surface

def add_padding(image, b_width):
    return np.pad(image, pad_width=b_width)

def cube_filter():
    return lambda x, y, z, value: np.ones([x, y, z], dtype=np.uint16) * value

def print_frame(df):
    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    
    # All dataframes hereafter reflect these changes.
    display(df)
    
    print('**RESET_OPTIONS**')
    
    # Resets the options
    pd.reset_option('all')


def get_label_vol_distribution(img):
    """
    Returns a list of all the cell volumes


    Parameters:
    ----------

    img (np.array):
        The image on which the filter is going to be applied

    Returns:
    -------

    label_vol_distri_df (pd.DataFrame):
        The first column corresponds to the label id, the second column is
        the volume of the label
    """

    #Some checks on the input
    assert isinstance(img, np.ndarray)
    assert np.issubdtype(img.dtype, np.integer)
    assert img.shape.__len__() == 3

    #Count the number of time each label appears in the image
    img_labels, nb_counts = np.unique(img, return_counts = True)

    #Zip together the labels and the counts
    zip_data = [[label, count] for label, count in zip(img_labels, nb_counts)]

    #Sort the labels based on the number of counts they have
    sorted_labels = sorted(zip_data, key = lambda x: x[1])

    sorted_label_ar = np.array(sorted_labels)
    sorted_label_df = pd.DataFrame(data = sorted_label_ar, columns = ["label_id", "label_vol"])
    return sorted_label_df

def remove_unconnected_regions_of_labels(label_img_ar):
    """
    Some of the labels are not entirely connected in space. This scripts find the voxels that 
    belong to different regions of the cell label and then only keep the voxels that belong to 
    the biggest region. Could be done manually but this script saves a lot of time

    Parameters:
    ------------

    label_img_ar (np.array):
        The image of the cell labels.

    Returns:
    label_img_ar (np.array):
        The image of the cell labels where the unconnected regions have been removed.
    """
    #Loop over the labels in the image
    for label_id in np.unique(label_img_ar):
        if label_id == 0: continue #The 0 label is for the background

        #Only select the part of the image that corresponds to the cell label
        masked_ar = np.ma.masked_where(label_img_ar == label_id, label_img_ar).mask.astype(int)

        #Check if they are different regions of the cell label
        labels_out = cc3d.connected_components(masked_ar, connectivity= 6)
        unique_labels, label_counts = np.unique(labels_out, return_counts = True)

        #Select the cell lable (!= 0) that has the most voxels
        max_label = unique_labels[np.argmax(label_counts[1:]) + 1]

        #Select all the voxels that do not belong to the biggest region of the cell label
        voxel_to_remove = np.argwhere((labels_out != max_label) & (labels_out != 0))

        #Remove the voxels that do not belong to the biggest region of the cell label in the orginal image
        label_img_ar[voxel_to_remove[:,0], voxel_to_remove[:,1], voxel_to_remove[:,2]] = 0

    return label_img_ar

def perform_cell_label_erosion(img_label_ar, nb_iterations = 1):
    """
    Errod the labels of the cells in the image. This is done by removing the voxels that are
    at the boundary of the cell.

    Parameters:
    -----------

    img_label_ar (np.array):
        The image of the cell labels

    nb_iterations (int):
        The number of times the erosion should be performed
    """


    #Loop over the labels in the image
    for label_id in np.unique(img_label_ar):
        if label_id == 0: continue #The 0 label is for the background
  
        #Only select the part of the image that corresponds to the cell label
        cell_img_before = np.ma.masked_where(img_label_ar == label_id, img_label_ar).mask.astype(bool)

        #Erode the cell image
        for i in range(nb_iterations): cell_img_after = binary_erosion(cell_img_before)

        #Find all the voxels that have been removed by the erosion
        voxels_to_remove = np.argwhere(cell_img_before != cell_img_after)

        #Replace all these voxels by 0 in the original image
        img_label_ar[voxels_to_remove[:, 0], voxels_to_remove[:, 1], voxels_to_remove[:, 2]] = 0

    return img_label_ar

def generate_cell_surface_meshes(label_img_ar, voxel_resolution, smoothing_iterations = 10, export = False):
    """
        Convert the label image into a surface meshes.

        Parameters:
        -----------
        label_img_ar (np.array):
            The image of the cell labels

        voxel_resolution (list):
            The resolution of the voxels in the image in the order [z, x, y]

        smoothing_iterations (int):
            The number of smoothing iterations to perform on the surface mesh
    
    """

    #Loop over the labels in the image
    meshes = {}
    
    for label_id in np.unique(label_img_ar):
        if label_id == 0: continue # The 0 label is for the background

        #Create the triangulated surface
        surface = label_to_surface(label_img_ar == label_id)

        #Extract the points and faces
        points, faces = surface[0], surface[1]
        points = points * voxel_resolution

        #Create the surface mesh
        cell_surface_mesh  = tm.Trimesh(points, faces)

        #Smooth the surface mesh
        cell_surface_mesh = tm.smoothing.filter_laplacian(cell_surface_mesh, iterations = smoothing_iterations)
        meshes[label_id] = cell_surface_mesh
        #Save the surface mesh in the folder ./cell_surface_meshes
        if export: 
            cell_surface_mesh.export("./cell_surface_meshes/cell_surface_{}.ply".format(label_id))
            continue
        meshes[label_id] = cell_surface_mesh
        
    return meshes