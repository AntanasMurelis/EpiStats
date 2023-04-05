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
def convert_cell_labels_to_meshes(
    img,
    voxel_resolution,
    smoothing_iterations = 0
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


    #Keep track of all the faces of the given cell that are in contact with other cells
    contact_faces = []

    cell_mesh = cell_mesh_lst[cell_id - 1]
    assert(isinstance(cell_mesh, tm.Trimesh))

    #Loop over the neighbors of the cell
    for neighbor_id in cell_neighbors_lst:

        #Find the closest distance between the center of the given cell's triangles and the triangles of the neighbor cells
        closest_point, distance, triangle_id = tm.proximity.closest_point(cell_mesh_lst[neighbor_id - 1], cell_mesh.triangles_center)

        #Get the indices of the triangles whose centroids are located at distances shorter than the cutoff distance from the neighbor cell surface 
        lateral_triangle_id = np.where(distance <= contact_cutoff)[0]
        contact_faces.append(lateral_triangle_id)

    #Create a unique list of faces that are in contact with other cells
    contact_faces = np.unique(np.concatenate(contact_faces))

    #sum the area of all the faces that are in contact with other cells
    contact_area = np.sum(cell_mesh_lst[cell_id - 1].area_faces[contact_faces])

    #store all the contact faces in a mesh 
    contact_mesh = tm.Trimesh(cell_mesh_lst[cell_id - 1].vertices, cell_mesh_lst[cell_id - 1].faces[contact_faces])
    contact_mesh.export('contact_mesh.stl')

    #Calculate the fraction of contact area
    contact_fraction = contact_area / cell_mesh_lst[cell_id - 1].area

    return contact_fraction
#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
def collect_cell_morphological_statistics(labeled_img, img_resolution, contact_cutoff):
    """
    Collect the following statistics about the cells:
        - cell_areas
        - cell_volumes
        - cell_neighbors_list
        - cell_nb_of_neighbors
        - cell_principal_axis
        - cell_contact_area_fraction

    And returns this statistics in a pandas dataframe.

    Parameters:
    -----------
    labeled_img: (np.array, 3D)
        The results of the segmentation, where the background as a label of 
        0 and the cells are labeled with consecutive integers starting from 1. The 3D image 
        is assumed to be in the standard numpy format (x, y, z).

    img_resolution: (np.array, 1D)
        The resolution of the image in microns (x_res, y_res, z_res)

    contact_cutoff: (float)
        The cutoff distance in microns for two cells to be considered in contact
    
    Returns:
    --------
    cell_statistics_df: (pd.DataFrame)
        A pandas dataframe containing the cell statistics
    """


    #Starts by converting the cell labels to meshes
    mesh_lst = convert_cell_labels_to_meshes(labeled_img, img_resolution)

    #Create a folder if doesn't already exists to store the meshes of the cells
    if not path.exists('cell_meshes'): mkdir('cell_meshes')

    #Save the meshes of the cells to this folder
    for cell_id, mesh in enumerate(mesh_lst): mesh.export(path.join(getcwd(), "cell_meshes", f"cell_{cell_id}.stl"))

    #Get the list of the ids of the cells
    cell_id_lst = np.unique(labeled_img)[1:]

    #Make sure the cell ids are consecutive integers starting from 1
    assert(np.all(cell_id_lst == np.arange(1, len(cell_id_lst) + 1)))

    #Get the cell areas
    cell_areas = compute_cell_areas(mesh_lst)

    #Get the cell volumes
    cell_volumes = compute_cell_volumes(mesh_lst)

    #Compute the isoperimetric ratio of each cell
    cell_isoperimetric_ratio = [(area**3) / (volume**2) for area, volume in zip(cell_areas, cell_volumes)]

    #Get the list of the neighbors of each cell
    cell_neighbors_lst = [get_cell_neighbors(labeled_img, cell_id) for cell_id in cell_id_lst]

    #Get the number of neighbors of each cell
    cell_nb_of_neighbors = [len(cell_neighbors) for cell_neighbors in cell_neighbors_lst]

    #Get the principal axis and the elongation of each cell
    cell_principal_axis_lst = []
    cell_elongation_lst = []
    for cell_id in cell_id_lst:
        principal_axis, elongation = compute_cell_principal_axis_and_elongation(mesh_lst[cell_id - 1])
        cell_principal_axis_lst.append(principal_axis)
        cell_elongation_lst.append(elongation)

    #Get the contact area fraction of each cell
    cell_contact_area_fraction_lst = []
    for cell_id in cell_id_lst:
        contact_area_fraction = compute_cell_contact_area_fraction(mesh_lst, cell_id, cell_neighbors_lst[cell_id - 1], contact_cutoff)
        cell_contact_area_fraction_lst.append(contact_area_fraction)

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
    
    return cell_statistics_df



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
        nb_cubes_y=2,
        nb_cubes_z=1,
        interstitial_space=0,
    )

    #Collect the statistics of the cells
    cell_statistics_df = collect_cell_morphological_statistics(img, np.array([1, 1, 2]), 1)


    #Save the cell statistics to a csv file
    cell_statistics_df.to_csv('cell_statistics.csv', sep = ",", index=False)
