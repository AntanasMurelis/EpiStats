import os
import numpy as np
import trimesh as tm
from skimage.io import imread, imsave
from tqdm import tqdm
import argparse

from convert_to_cell_mesh import get_polydata_from_file
from convert_to_cell_mesh import get_cell_ids
from convert_to_cell_mesh import separate_cell_points_and_faces
from convert_to_cell_mesh import write_unstructured_grid
from SegmentationStatisticsCollector import convert_cell_labels_to_meshes 
from SegmentationStatisticsCollector import process_labels

#------------------------------------------------------------------------------------------------------------
def create_combined_mesh(
        cell_labels, 
        labels_path, 
        make_meshes=True, 
        make_combined=True, 
        smoothing_iters=10,
        voxel_size=(0.1, 0.1, 0.1)):
    """
    Function to convert a preprocessed array of cell labels into meshes.

    Parameters:
    -----------
    cell_labels: (iterable)
        an iterable containing the ids of the cells to be meshed, or "all" if all the labels 
        in the array need to be converted.
    
    labels_path: (str)
        the path to the array.

    make_meshes: (bool, optional, default=True)
        a boolean telling if the meshes for the single cells are required.
    
    make_combined: (bool, optional, default=True)
        a boolean telling if the combined mesh of cell_labels is required.
    
    smoothing_iters: (int, optional, default=10)
        an int representing the number of smoothing iterations to be done when creating the mesh.
    
    voxel_size: (iterable, optional, default=(0.1, 0.1, 0.1))
        an iterable of len (3,) containing the voxel size in microns in (x, y, z) ordering.

    Returns:
    --------
    meshes: 
        a list whose elements are the meshes associated to each label.
    
    combined_mesh: 
        the mesh associated to the combined labels.
    
    cell_labels: 
        a list containing the indexes of the processed labels. 
    """
    
    # Read the image
    print("Loading labels...")
    labels = np.load(labels_path)

    # Set list of label ids
    print("Setting up things for conversion...")
    if cell_labels == "all":
        cell_labels = np.unique(labels)

    # Make a combined mask for all the labels in cell_labels
    combined_mask = np.isin(labels, cell_labels).astype(int)
    filtered_labels = labels * combined_mask

    # Transform voxel size in meters and convert to np.array
    voxel_size = [x*1e-6 for x in voxel_size]
    voxel_size = np.array(voxel_size)
    
    # Create meshese from the filtered_labels array using trimesh
    meshes = None
    if make_meshes:
        print("Converting single cells into meshes...")
        meshes = convert_cell_labels_to_meshes(filtered_labels, 
                                               voxel_resolution=voxel_size, 
                                               smoothing_iterations=smoothing_iters)
    
    # Get the combined mesh from combined_mask array
    combined_mesh = None
    if make_combined:
        print("Converting combined cells into meshes...")
        combined_mesh = convert_cell_labels_to_meshes(combined_mask, 
                                                      voxel_resolution=voxel_size, 
                                                      smoothing_iterations=smoothing_iters)
    
    return meshes, combined_mesh, cell_labels
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def export_mesh(
        mesh, 
        id, 
        save_dir, 
        file_type=".stl", 
        overwrite=False) -> None:
    """
    Export a trimesh object into .stl, .ply or .obj format

    Parameters:
    -----------
    mesh: (trimesh obj) 
        the trimesh object to export
    
    id: (int or str)
        either an int corresponding to the cell label, or "combined" if the combined mesh needs 
        to be saved
    
    save_dir: (str)
        path to the location in which the mesh object will be exported. Single meshes will
        be saved in 'save_dir/single_cells', while combined ones in 'save_dir/combined'.
    
    file_type: (str, optional, default=".stl")
        the format of the file to be saved, chosen from [.ply, .stl, .obj]

    overwrite: (bool, optional, default=False)
        if True, overwrite the file, if it already exists 
    
    Returns:
    --------
    PATH TO SAVED FILE
    """
    # check that the file_type is among the allowed ones
    if file_type not in [".stl", ".ply", ".obj"]:
        raise ValueError("file_type is not correct. It must be among [.stl, .ply, .obj].")


    if id == "combined":
        # set the saving dir
        combined_save_dir = os.path.join(save_dir, "combined")
        if not os.path.exists(combined_save_dir):
            os.makedirs(combined_save_dir)       
        # save the mesh
        file_name = 'combined_cell' + file_type
        # check if the file already exists and if the user wants to overwrite
        if os.path.exists(os.path.join(combined_save_dir, file_name)) and not overwrite:
            print('Aborting export: The file already exists, and overwrite is set to False.')
            return
        else:
            mesh.export(os.path.join(combined_save_dir, file_name))
            return os.path.join(combined_save_dir, file_name)
    elif isinstance(id, (int, np.int32, np.int64)):
        # set the saving dir
        single_cells_save_dir = os.path.join(save_dir, "single_cells")
        if not os.path.exists(single_cells_save_dir):
            os.makedirs(single_cells_save_dir)
        # save the mesh
        file_name = 'cell_{}'.format(id) + file_type
        if os.path.exists(os.path.join(single_cells_save_dir, file_name)) and not overwrite:
            print('Aborting export: the file already exists, and overwrite is set to False.')
            return
        else:
            mesh.export(os.path.join(single_cells_save_dir, file_name))
            return os.path.join(single_cells_save_dir, file_name)
    else:
        raise ValueError("id must be either an int, np.int32, np.int64, or the string ""combined"".")
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def convert_to_vtk(
        file_path, 
        output_dir
):
    """
    Convert a mesh object in .stl, .ply or .obj format to .vtk format

    Parameters:
    -----------
    file_path: (str)
        the path to the mesh file to convert
    
    output_dir: (str) 
        the path to the directory to save the converted mesh
    
    Returns:
    --------
    PATH TO SAVED FILE
    """
    #Read the mesh
    polydata = get_polydata_from_file(file_path)

    #Set the output file name
    file_name = os.path.basename(file_path)
    file_extension = file_path[-4:]
    output_file_name = file_name.replace(file_extension, ".vtk")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_file_name)

    #Get the cell to which each face belongs
    face_cell_id, point_cell_id = get_cell_ids(polydata)

    #Separate the faces and points of each cell
    face_dic, point_dic = separate_cell_points_and_faces(polydata, face_cell_id, point_cell_id)

    #Get the cells unstructured grids
    write_unstructured_grid(polydata, face_dic, point_dic, output_path)

    return output_path
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def main(
    img_path,
    output_dir,
    dilation_iters,
    erosion_iters,
    voxel_size,
    smoothing_iters=10,
    combined_mesh=True,
    labels_to_convert="all",
):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #------------------------------------
    # 1. Read and preprocess the image (skipped in case the preprocessed image already exists)
    print("1. Labels preprocessing...")
    img = imread(img_path)
    img = np.einsum('kij->ijk', img)
    process_labels(
        img, 
        dilation_iterations=dilation_iters, 
        erosion_iterations=erosion_iters, 
        output_directory=os.path.join(output_dir, "processed_labels"), 
        overwrite=False, 
        renumber=False)
    # set the path to the processed labels file
    processed_file_name = 'processed_labels_er{}_dil{}.npy'.format(erosion_iters, dilation_iters)
    processed_img_path =  os.path.join(output_dir, "processed_labels", processed_file_name)
    #-----------------------------------

    #-----------------------------------
    # 2. Create meshes and combined mesh
    print("2. Creating meshes...")
    meshes, combined_mesh, labels_list = create_combined_mesh(
        cell_labels=labels_to_convert, 
        labels_path=processed_img_path,
        make_meshes=True, 
        make_combined=combined_mesh, 
        smoothing_iters=smoothing_iters,
        voxel_size=voxel_size
    )
    #-----------------------------------

    #-----------------------------------
    # 3. Export mesh files
    print("3. Exporting mesh files...")
    output_dir = os.path.join(output_dir, "_smooth{}".format(smoothing_iters))
    mesh_paths = []
    for mesh, id in tqdm(zip(meshes, labels_list), desc="Exporting meshes: ", total=len(labels_list)):
        curr_path = export_mesh(
            mesh=mesh,
            id=id,
            save_dir=os.path.join(output_dir, 'stl_files'),
            file_type='.stl',
            overwrite=False
        )
        mesh_paths.append(curr_path)

    # export big_mesh
    if combined_mesh:
        combined_mesh = combined_mesh[0]
        combined_mesh_path = export_mesh(
            mesh=combined_mesh,
            id="combined",
            save_dir=os.path.join(output_dir, 'stl_files'),
            file_type='.stl',
            overwrite=False
        )
    #-----------------------------------

    #-----------------------------------
    # 4. Convert meshes into .vtk format
    print("4. Converting into .vtk format...")
    vtk_dir = os.path.join(output_dir, "vtk_files")
    if combined_mesh:
        convert_to_vtk(combined_mesh_path, os.path.join(vtk_dir, "combined"))

    for mp in tqdm(mesh_paths):
        convert_to_vtk(mp, os.path.join(vtk_dir, "single_cells"))
#------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    path_to_dir = "/nas/groups/iber/Users/Federico_Carrara/create_meshes/data/curated_labels_clean/"
    image_name = "lung_new_sample_b_curated_segmentation_central_crop_relabeled_filled.tif"
    img_path = os.path.join(path_to_dir, image_name)

    cells_ids = [2, 7, 22, 24, 29, 31, 35, 40, 45, 46, 48, 51, 65, 80, 120, 121, 122, 124]

    main(
        img_path=img_path,
        output_dir='./try',
        erosion_iters=2,
        dilation_iters=4,
        smoothing_iters=10,
        voxel_size=(0.1625, 0.1625, 0.25),
        combined_mesh=True,
        labels_to_convert=cells_ids
    )



