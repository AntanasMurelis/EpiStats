import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
import skimage
from tqdm import tqdm
from misc import get_label_vol_distribution, remove_unconnected_regions_of_labels, perform_cell_label_erosion, generate_cell_surface_meshes
from skimage.morphology import binary_erosion
from skimage.segmentation import expand_labels

def obtain_statistics(labels, voxel_resolution = [0.236e-6, 0.236e-6, 0.487e-6], min_label_vol = 10000):
    """Obtains statistics of the cells in the label image.

    Args:
        labels (array): 3D array of the cell labels.
        voxel_resolution (list, optional): Voxel resolution. Defaults to [0.236e-6, 0.236e-6, 0.487e-6].
        min_label_vol (int, optional): Minimum voxel volume to keep a cell body in the label. Defaults to 100000.

    Returns:
        DataFrame: Statistics of the cells in the label image.
    """
    
    ############ Preprocessing ############
    
    labels = np.einsum('kij->ijk', labels)
    labels = np.pad(labels, pad_width=10, mode='constant', constant_values=0)
    
    label_vol_df = get_label_vol_distribution(labels)
    
    #Remove the labels which are below a certain threshold
    label_to_remove_df = label_vol_df.loc[label_vol_df["label_vol"] < min_label_vol]
    label_to_remove_lst = label_to_remove_df["label_id"].to_list()
    labels[np.isin(labels, label_to_remove_lst)] = 0
    
    #Remove the unconnected regions of the labels
    labels = remove_unconnected_regions_of_labels(labels)

    # Erode the labels of the cells in the image. 
    labels = perform_cell_label_erosion(labels, nb_iterations = 1)
    
    # #Expand the labels to make sure that the cell surfaces are touching
    labels = expand_labels(labels, distance=2)

    # print("Saving cell surface meshes...")
    #Generate the surface meshes
    cell_meshes = generate_cell_surface_meshes(labels, voxel_resolution)

    #######################################
    
    id = np.unique(labels)[1:]
    cell_statistics = {}
    cell_statistics['ID'] = id
    ######### Elementary Statistics ########
    
    regionprops = skimage.measure.regionprops(labels)
    elongation = []
    volume = []
    
    for i in range(np.size(id)):
        elongation.append(
            regionprops[i].axis_major_length/regionprops[i].axis_minor_length)
        volume.append(regionprops[i].area)
        
    cell_statistics['Elongation'] = elongation
    cell_statistics['Volume'] = np.array(volume)*voxel_resolution[0]*voxel_resolution[1]*voxel_resolution[2]


    # Surface Area - Based on meshing - used Steve's implementation for meshing.
    s_area = []
    for _ in tqdm(id, desc='Calculating surface area'):
        area = skimage.measure.mesh_surface_area(cell_meshes[_].vertices, cell_meshes[_].faces)
        s_area.append(area)
    cell_statistics['Surface area'] = s_area
    
    
    # Sphericity and compactness
    cell_statistics['Compactness'] = (np.power(cell_statistics['Volume'], 2)) /(np.power(cell_statistics['Surface area'], 3))
    cell_statistics['Sphericity'] = np.power(cell_statistics['Compactness'], 1/3)
    
    # Neighbouring cells
    touch_matrix = np.array(cle.generate_touch_matrix(labels))
    touch_array = np.array([touch_matrix[i][1:].sum() for i in id])
    cell_statistics['Neighbours'] = touch_array
    
    # Portion of total surface in contact:
    
    touch_count_matrix = cle.generate_touch_count_matrix(labels)
    
    touch_count_matrix = np.array(cle.generate_touch_count_matrix(labels))
    touch_count_array = np.array([touch_count_matrix[i][1:].sum() for i in id])
    touch_portion = np.asarray(touch_count_array) / np.array(volume)
    cell_statistics['Contact portion'] = touch_portion
    
    return pd.DataFrame(cell_statistics)