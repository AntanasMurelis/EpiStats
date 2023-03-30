import numpy as np
import pandas as pd
import trimesh as tm
from scipy import ndimage
from napari_process_points_and_surfaces import label_to_surface
from os import path, _exit, getcwd, mkdir
from tests.CubeLatticeTest import *
from tests.SphereTest import *



"""
    The methods in this script gather statistics from the 3D segmented labeled images. 
"""


#------------------------------------------------------------------------------------------------------------
#def obtain_cube_statistics(labels):
#    """Generates statistics of the image labels
#
#    Args:
#        image (array): LxWxB Image
#        labels (array): LxWxB Label
#
#    Returns:
#        DataFrame: Statistics Dataframe 
#    """
#    
#    cell_statistics = {}
#    # Statistics:
#    statistics = cle.statistics_of_labelled_pixels(None, labels)
#    cell_statistics['Volume'] = statistics['area']
#    
#    
#    id = np.unique(labels)[1:]
#    # Surface Area - Based on meshing - use Steve's implementation...
#    s_area = []
#    for _ in (pbar := tqdm(id)):
#        pbar.set_description(f"Meshing label: {_}")
#        mesh = skimage.measure.marching_cubes(labels*(labels==_),)
#        area = skimage.measure.mesh_surface_area(mesh[0], mesh[1])
#        s_area.append(area)
#    cell_statistics['Surface area'] = s_area
#    
#    ############# Membrane statistics #############
#    
#    # membranes = skimage.segmentation.find_boundaries(labels, mode='inner')
#    # l_membranes = membranes * labels
#    # m_statistics = cle.statistics_of_labelled_pixels(None, l_membranes)
#    
#    ################################################
#    
#    # Sphericity and compactness
#    cell_statistics['Compactness'] = (np.power(cell_statistics['Volume'], 2)) /(np.power(cell_statistics['Surface area'], 3))
#    cell_statistics['Sphericity'] = np.power(cell_statistics['Compactness'], 1/3)
#    
#    
#    # TODO: Elongation, Flatness, Sparsness
#    regionprops = skimage.measure.regionprops(labels)
#    elongation = []
#    for i in range(np.size(id)):
#        elongation.append(
#            regionprops[i].axis_major_length/regionprops[i].axis_minor_length)
#    cell_statistics['Elongation'] = elongation
#    
#    # Neighbours:
#    touch_matrix = cle.generate_touch_matrix(labels)
#    neighbours_arr = np.asarray(touch_matrix).sum(axis=1)[1:]
#    cell_statistics['Neighbours'] = neighbours_arr
#    
#    # Portion of total surface in contact:
#    touch_count_matrix = cle.generate_touch_count_matrix(labels)
#    neighbour_touch = touch_count_matrix[1:, 1:].sum(axis=1)[0]
#    touch_portion = np.asarray(neighbour_touch) / cell_statistics['Volume']
#    cell_statistics['Contact portion'] = touch_portion
#        
#    # for i in cell_statistics.keys():
#    #     print(f'{i }: {np.size(cell_statistics[i])}')
#    
#    return pd.DataFrame(cell_statistics)
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def convert_cell_labels_to_meshes(
    img,
    voxel_resolution,
    smoothing_iterations = 5
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

    Returns:
    --------
        mesh_lst (list):
            A list of triangular meshes, in the trimesh format.
    
    """
    assert isinstance(voxel_resolution, np.ndarray)
    assert voxel_resolution.ndim == 1 
    assert voxel_resolution.shape[0] == 3 

    #Always add some extra padding to the image to make sure the cell surfaces are not 
    #in direct contact with the boundaries of the image. This would cause the meshes to be 
    #only partially triangulated
    img_padded = np.pad(img, pad_width=10, mode='constant', constant_values=0)

    #Store the meshes in this list
    mesh_lst = []

    #Loop over the labels in the image
    for label_id in np.unique(img):
        if label_id == 0: continue #The 0 label is for the background

        #Create the triangulated surface
        surface = label_to_surface(img_padded == label_id)

        #Extract the points and faces
        points, faces = surface[0], surface[1]
        points = points * voxel_resolution

        #Create the surface mesh
        cell_surface_mesh  = tm.Trimesh(points, faces)

        #Smooth the surface mesh
        cell_surface_mesh = tm.smoothing.filter_laplacian(cell_surface_mesh, iterations = smoothing_iterations)

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
def compute_cell_principal_axis(cell_mesh):

    #Use the inertia tensor of the cell shape to compute its principal axis
    eigen_values, eigen_vectors = tm.inertia.principal_axis(cell_mesh.moment_inertia)





#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
def compute_cell_contact_fraction(labeled_img, cell_id):
    """
    Compute the fraction of the cell surface which is in contact with other cells.
    
    Parameters:
    -----------

    labeled_img: (np.array, 3D)
        The tirangular meshes of the cell in the standard trimesh format

    cell_id: (int)
        The id of the cell for which we want to find the neighbors

    Returns
    -------
    contact_fraction: (float)
    """

    #Add some padding to the image
    img_padded = np.pad(img, pad_width=10, mode='constant', constant_values=0)


    #Get the voxels of the cell
    binary_img = img_padded == cell_id

    #Expand the volume of the cell by 2 voxels in each direction
    expanded_cell_voxels = ndimage.binary_dilation(binary_img, iterations=2)

    #Find the voxels that are directly in contact with the surface of the cell
    cell_surface_voxels = expanded_cell_voxels ^ binary_img

    #Count the number of voxels of the cell surface which are in contact with the background
    nb_voxels_in_contact_with_bg = np.sum(img_padded[cell_surface_voxels] == 0)

    #Count the number of voxels of tthe cell surface which are in contact with other cells
    nb_voxels_in_contact_with_cells = np.sum((img_padded[cell_surface_voxels] != 0) & (img_padded[cell_surface_voxels] != cell_id))

    #TO CONTINUE!! 
    #This method needs to take into account the side length of the voxels, if the resolution is not equal in the 3 dimensions
    #the contact fraction will be wrong

    contact_fraction = float(nb_voxels_in_contact_with_cells) / float(nb_voxels_in_contact_with_bg + nb_voxels_in_contact_with_cells)

    print(nb_voxels_in_contact_with_bg, np.sum(cell_surface_voxels), nb_voxels_in_contact_with_cells, contact_fraction)

    return contact_fraction

#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
def compute_cell_morphological_statistics(labeled_img):
    pass






#------------------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------------------
#def obtain_cube_statistics(labels):
#    """Generates statistics of the image labels
#
#    Args:
#        image (array): LxWxB Image
#        labels (array): LxWxB Label
#
#    Returns:
#        DataFrame: Statistics Dataframe 
#    """
#    
#    cell_statistics = {}
#    # Statistics:
#    statistics = cle.statistics_of_labelled_pixels(None, labels)
#    cell_statistics['Volume'] = statistics['area']
#    
#    
#    id = np.unique(labels)[1:]
#    # Surface Area - Based on meshing - use Steve's implementation...
#    s_area = []
#    for _ in (pbar := tqdm(id)):
#        pbar.set_description(f"Meshing label: {_}")
#        mesh = skimage.measure.marching_cubes(labels*(labels==_),)
#        area = skimage.measure.mesh_surface_area(mesh[0], mesh[1])
#        s_area.append(area)
#    cell_statistics['Surface area'] = s_area
#    
#    ############# Membrane statistics #############
#    
#    # membranes = skimage.segmentation.find_boundaries(labels, mode='inner')
#    # l_membranes = membranes * labels
#    # m_statistics = cle.statistics_of_labelled_pixels(None, l_membranes)
#    
#    ################################################
#    
#    # Sphericity and compactness
#    cell_statistics['Compactness'] = (np.power(cell_statistics['Volume'], 2)) /(np.power(cell_statistics['Surface area'], 3))
#    cell_statistics['Sphericity'] = np.power(cell_statistics['Compactness'], 1/3)
#    
#    
#    # TODO: Elongation, Flatness, Sparsness
#    regionprops = skimage.measure.regionprops(labels)
#    elongation = []
#    for i in range(np.size(id)):
#        elongation.append(
#            regionprops[i].axis_major_length/regionprops[i].axis_minor_length)
#    cell_statistics['Elongation'] = elongation
#    
#    # Neighbours:
#    touch_matrix = cle.generate_touch_matrix(labels)
#    neighbours_arr = np.asarray(touch_matrix).sum(axis=1)[1:]
#    cell_statistics['Neighbours'] = neighbours_arr
#    
#    # Portion of total surface in contact:
#    touch_count_matrix = cle.generate_touch_count_matrix(labels)
#    neighbour_touch = touch_count_matrix[1:, 1:].sum(axis=1)[0]
#    touch_portion = np.asarray(neighbour_touch) / cell_statistics['Volume']
#    cell_statistics['Contact portion'] = touch_portion
#        
#    # for i in cell_statistics.keys():
#    #     print(f'{i }: {np.size(cell_statistics[i])}')
#    
#    return pd.DataFrame(cell_statistics)
#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    #Test the mesh generation part by meshing of the test lattice cube images. 
    img = generate_cube_lattice_image(
        nb_x_voxels=100,
        nb_y_voxels=100,
        nb_z_voxels=100,
        cube_side_length=20,
        nb_cubes_x=2,
        nb_cubes_y=1,
        nb_cubes_z=1,
        interstitial_space=0,
    )

    #img = generate_image_of_sphere(sphere_radius=100, voxel_size=1)

    #Get the meshes of the cubes
    #mesh_lst = convert_cell_labels_to_meshes(img, np.array([1, 1, 2]))


    for cell_id in np.unique(img)[1:]:
        compute_cell_contact_fraction(img, cell_id)