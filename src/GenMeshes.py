import numpy as np
import trimesh as tm
import os
from typing import Optional, Iterable, Dict, Union
from tqdm import tqdm
from napari_process_points_and_surfaces import label_to_surface


#------------------------------------------------------------------------------------------------------------
def convert_labels_to_meshes(
    img: np.ndarray[int], 
    voxel_resolution: Iterable[float],
    smoothing_iterations: Optional[int] = 1,
    output_directory: Optional[str] = 'output',
    overwrite: Optional[bool] = False,
    pad_width: Optional[int] = 0,
    mesh_file_format: Optional[str] = 'stl'
) -> Dict[int, Dict[str, tm.base.Trimesh]]:
    """
    Convert the labels of the cells in the 3D segmented image to triangular meshes. Please make sure that the 
    image has correctly been segmented, i.e. that there is no artificially small cells, or cell labels 
    split in different parts/regions.

    Parameters:
    -----------
        img (np.ndarray, 3D, dtype=int):
            The 3D segmented image.

        voxel_resolution (Iterable[float]):
            The voxel side lengths in microns in the order [x, y, z]

        smoothing_iterations (int, optional, default=1):
            The number of smoothing iterations to perform on the surface meshes

        output_directory (str, optional, default='output):
            Name of the folder where the cell_meshes will be saved

        overwrite (bool, optional, default=False):
            If True, overwrite existing mesh files

        pad_width (int, optional, default=0):
            The width of the padding to be applied to the input image

        mesh_file_format (str, optional, default='stl')
            The file format chosen for storing the mesh files.

    Returns:
    --------
        mesh_lst (Dict[int, Dict[str, tm.base.Trimesh or bool]):
            A dictionary whose keys are cell indexes and values are pairs of triangular meshes, 
            in the trimesh format and boolean indicating whether the cell should be excluded 
            from later computations.
    
    """
    
    img_padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    label_ids = np.unique(img_padded)
    
    meshes_folder = os.path.join(output_directory, 'cell_meshes')
    if not os.path.exists(meshes_folder):
        os.makedirs(meshes_folder)

    mesh_dict = {}

    for label_id in tqdm(label_ids, desc="Generating meshes"):
        if label_id == 0: 
            continue

        cell_mesh_file = os.path.join(meshes_folder, f"cell_{label_id}.{mesh_file_format}")

        if (not os.path.isfile(cell_mesh_file)) or overwrite:
            surface = label_to_surface(img_padded == label_id)
            points, faces = surface[0], surface[1]
            points = (points - np.array([pad_width, pad_width, pad_width])) * np.asarray(voxel_resolution)

            cell_surface_mesh = tm.Trimesh(points, faces)
            cell_surface_mesh = tm.smoothing.filter_mut_dif_laplacian(cell_surface_mesh, iterations=smoothing_iterations, lamb=0.5)
            cell_surface_mesh.export(cell_mesh_file)  # save the mesh
        else:
            cell_surface_mesh = tm.load_mesh(cell_mesh_file)

        mesh_dict[label_id] = cell_surface_mesh

    return mesh_dict

#------------------------------------------------------------------------------------------------------------


### MAKE COMBINED MESH, CONVERT TO VTK