import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
import skimage
from tqdm import tqdm

def obtain_cube_statistics(labels):
    """Generates statistics of the image labels

    Args:
        image (array): LxWxB Image
        labels (array): LxWxB Label

    Returns:
        DataFrame: Statistics Dataframe 
    """
    
    cell_statistics = {}
    # Statistics:
    statistics = cle.statistics_of_labelled_pixels(None, labels)
    cell_statistics['Volume'] = statistics['area']
    
    
    id = np.unique(labels)[1:]
    # Surface Area - Based on meshing - use Steve's implementation...
    s_area = []
    for _ in (pbar := tqdm(id)):
        pbar.set_description(f"Meshing label: {_}")
        mesh = skimage.measure.marching_cubes(labels*(labels==_),)
        area = skimage.measure.mesh_surface_area(mesh[0], mesh[1])
        s_area.append(area)
    cell_statistics['Surface area'] = s_area
    
    ############# Membrane statistics #############
    
    # membranes = skimage.segmentation.find_boundaries(labels, mode='inner')
    # l_membranes = membranes * labels
    # m_statistics = cle.statistics_of_labelled_pixels(None, l_membranes)
    
    ################################################
    
    # Sphericity and compactness
    cell_statistics['Compactness'] = (np.power(cell_statistics['Volume'], 2)) /(np.power(cell_statistics['Surface area'], 3))
    cell_statistics['Sphericity'] = np.power(cell_statistics['Compactness'], 1/3)
    
    
    # TODO: Elongation, Flatness, Sparsness
    regionprops = skimage.measure.regionprops(labels)
    elongation = []
    for i in range(np.size(id)):
        elongation.append(
            regionprops[i].axis_major_length/regionprops[i].axis_minor_length)
    cell_statistics['Elongation'] = elongation
    
    # Neighbours:
    touch_matrix = cle.generate_touch_matrix(labels)
    neighbours_arr = np.asarray(touch_matrix).sum(axis=1)[1:]
    cell_statistics['Neighbours'] = neighbours_arr
    
    # Portion of total surface in contact:
    touch_count_matrix = cle.generate_touch_count_matrix(labels)
    neighbour_touch = touch_count_matrix[1:, 1:].sum(axis=1)[0]
    touch_portion = np.asarray(neighbour_touch) / cell_statistics['Volume']
    cell_statistics['Contact portion'] = touch_portion
        
    # for i in cell_statistics.keys():
    #     print(f'{i }: {np.size(cell_statistics[i])}')
    
    return pd.DataFrame(cell_statistics)
