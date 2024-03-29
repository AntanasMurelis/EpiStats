import trimesh
import shutil
import numpy as np
import pandas as pd
import pymeshfix
import os
import sys
import pymeshlab
import vtk 
from napari_process_points_and_surfaces import label_to_surface
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from typing import List, Optional, Literal
from skimage import io
from scipy.ndimage import binary_dilation, binary_closing
from OuterShell import ExtendedTrimesh, OuterShell
from misc import get_cell_neighbors

"""Script for mesh refinement and preparation for the SimuCell3D."""

#---------------------------------------------------------------------------------------------------------------
def get_edges_with_more_than_two_faces(mesh: trimesh.Trimesh):
    """
    This function finds all the edges in a mesh that are shared by more than two faces.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        edges_with_more_than_two_faces (list): A list of edges that are shared by more than two faces.
    """
    # Compute the edges from the faces
    edges = np.sort(mesh.edges_sorted.reshape(-1, 2), axis=1)

    # Count the occurrence of each edge
    edge_count = Counter(map(tuple, edges))

    # Get the edges with more than two faces attached
    edges_with_more_than_two_faces = [edge for edge, count in edge_count.items() if count > 2]

    return edges_with_more_than_two_faces
#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def fill_holes_pymeshfix(mesh: trimesh.Trimesh):
    """
    This function fills the holes in a mesh using the PyMeshFix library.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        mesh (trimesh.Trimesh): The input mesh with filled holes.
    """
    # Create TMesh object
    tin = pymeshfix.PyTMesh()

    # Load vertices and faces from Trimesh object
    tin.load_array(mesh.vertices, mesh.faces)

    # Remove small components
    tin.remove_smallest_components()
    
    # Fill holes
    tin.fill_small_boundaries()

    # Clean (removes self intersections)
    tin.clean(max_iters=10, inner_loops=3)

    # Fill holes
    tin.fill_small_boundaries()

    # Retrieve the cleaned mesh as numpy arrays
    vclean, fclean = tin.return_arrays()

    # Update the original Trimesh object
    mesh.vertices = vclean
    mesh.faces = fclean

    return mesh
#---------------------------------------------------------------------------------------------------------------

 
 
 
 
#---------------------------------------------------------------------------------------------------------------   
def remove_non_manifold_faces_and_fill_holes(
        mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
    """
    This function removes the non-manifold faces from a mesh and fills the holes.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        mesh (trimesh.Trimesh): The input mesh with non-manifold faces removed and holes filled.
    """
    # Get the edges with more than two faces attached
    edges_with_more_than_two_faces = get_edges_with_more_than_two_faces(mesh)

    # Find all the faces that include at least one of the non-manifold edges
    non_manifold_faces = [tuple(face) for face in mesh.faces 
                          if any(tuple(sorted(edge)) in edges_with_more_than_two_faces 
                          for edge in combinations(face, 2))]

    # Remove the non-manifold faces from the mesh
    mesh.update_faces([i for i, face in enumerate(mesh.faces) if tuple(face) not in non_manifold_faces])
    print("Removed {} non-manifold faces".format(len(non_manifold_faces)))
    # Fill the holes
    mesh = fill_holes_pymeshfix(mesh)
    
    return mesh
#---------------------------------------------------------------------------------------------------------------   





#---------------------------------------------------------------------------------------------------------------
def remesh(
        input_dir: str, 
        output_dir: str, 
        min_edge_length: float = 0.1
    ) -> str:
    """
    This function remeshes .stl or .ply files in a directory.
    
    Parameters:
    - input_dir (str): The directory containing the .stl or .ply files to be remeshed.
    - output_dir (str): The directory where the remeshed files will be saved.
    - min_edge_length (float, optional): The minimum edge length to be used when remeshing. Defaults to 0.1.
    
    Returns:
    - str: The path to the directory where the remeshed files were saved.
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir), desc='Applying pymeshlab remeshing'):
        if filename.endswith('.stl') or filename.endswith('.ply'):
            input_file = os.path.join(input_dir, filename)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(input_file)

            # Set the edge length parameter
            try:
                min_edge_length = float(min_edge_length)
            except ValueError:
                print(f"Invalid minimum edge length: {min_edge_length}")
                sys.exit(1)

            # Set parameters for the isotropic explicit remeshing filter
            targetlen = pymeshlab.AbsoluteValue(min_edge_length)
            remesh_par = dict(targetlen=targetlen, iterations=10)   
            
            # Remesh
            ms.apply_filter('generate_surface_reconstruction_ball_pivoting',ballradius = pymeshlab.Percentage(1))
            ms.apply_filter('apply_coord_taubin_smoothing', stepsmoothnum = 20, lambda_ = 0.5)

            # Clean up the mesh
            ms.apply_filter('meshing_remove_duplicate_vertices')
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_null_faces')

            # Make mesh manifold
            ms.apply_filter('meshing_repair_non_manifold_edges')
            ms.apply_filter('meshing_repair_non_manifold_vertices')
            ms.apply_filter('meshing_close_holes')
            ms.apply_filter('meshing_re_orient_faces_coherentely')

            # Apply the isotropic explicit remeshing filter
            ms.apply_filter('meshing_isotropic_explicit_remeshing', **remesh_par)

            # Make mesh manifold
            ms.apply_filter('meshing_repair_non_manifold_edges')
            ms.apply_filter('meshing_repair_non_manifold_vertices')
            ms.apply_filter('meshing_close_holes')
            ms.apply_filter('meshing_re_orient_faces_coherentely')

            # Save the remeshed file
            output_file = os.path.join(output_dir, 'remeshed_' + filename)
            ms.save_current_mesh(output_file)
            
    return output_dir
#---------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------------   
def clean_meshes_in_directory(
        input_directory: str, 
        output_directory: str
    ) -> str:
    """
    This function cleans .ply or .stl mesh files in a directory by removing non-manifold faces and filling holes.
    
    Parameters:
    - input_directory (str): The directory containing the .ply or .stl files to be cleaned.
    - output_directory (str): The directory where the cleaned files will be saved.
    
    Returns:
    - str: The path to the directory where the cleaned files were saved.
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for filename in tqdm(os.listdir(input_directory), desc='Cleaning non-mainfold meshes'):
        if filename.endswith(".ply") or filename.endswith(".stl"):
            mesh_path = os.path.join(input_directory, filename)
            mesh = trimesh.load_mesh(mesh_path)
            
            # Clean the mesh
            cleaned_mesh = remove_non_manifold_faces_and_fill_holes(mesh)
            cleaned_filename = 'cleaned_' + filename
            
            # Save the cleaned mesh to a new file
            cleaned_mesh_path = os.path.join(output_directory, cleaned_filename)
            cleaned_mesh.export(cleaned_mesh_path)  
             
    return output_directory
  
#---------------------------------------------------------------------------------------------------------------   




#---------------------------------------------------------------------------------------------------------------
def get_polydata_from_file(filename):
    """
    Read a mesh file and return a vtkPolyData object

    Parameters:
    -----------

    filename: str
        Path to the mesh file

    Returns:
    --------

    polydata: vtkPolyData
        The mesh as a vtkPolyData object

    """

    extension = filename.split(".")[-1]

    if extension == "vtk":
        reader = vtk.vtkUnstructuredGridReader()
    elif extension == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == "stl":
        reader = vtk.vtkSTLReader()
    elif extension == "ply":
        reader = vtk.vtkPLYReader()
    elif extension == "obj":
        reader = vtk.vtkOBJReader()
    elif extension == "vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else: 
        raise ValueError("File extension not recognized")

    reader.SetFileName(filename)
    reader.Update()

    #The output might be at this stage an unstructured grid or a polydata
    reader_output =  reader.GetOutput()

    #We make sure to convert the mesh to polydata
    if isinstance(reader_output, vtk.vtkUnstructuredGrid):
        geom_filter = vtk.vtkGeometryFilter()
        geom_filter.SetInputData(reader_output)
        geom_filter.Update()
        polydata = geom_filter.GetOutput()
        return polydata
    elif isinstance(reader_output, vtk.vtkPolyData):
        return reader_output
    else:
        raise ValueError("The mesh could not be converted to polydata")
#---------------------------------------------------------------------------------------------------------------






#---------------------------------------------------------------------------------------------------------------
def get_cell_ids(polydata):
    """
    Get the cell to which each face and node belongs

    Parameters:
    -----------

    polydata: vtkPolyData
        The mesh as a vtkPolyData object

    Returns:
    --------

    face_cell_id: np.array()
        The id cell to which each face belongs
    
    point_cell_id: np.array()
        The id cell to which each node belongs
    
    """

    #Apply the connectivity filter to the mesh
    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputData(polydata)
    connectivity_filter.SetExtractionModeToAllRegions()
    connectivity_filter.ColorRegionsOn()
    connectivity_filter.Update()
    polydata = connectivity_filter.GetOutput()

    #Get the cell to which each face belongs and same for the nodes
    face_cell_id = np.array(polydata.GetCellData().GetArray("RegionId"))
    point_cell_id = np.array(polydata.GetPointData().GetArray("RegionId"))
    return face_cell_id, point_cell_id
#---------------------------------------------------------------------------------------------------------------






#---------------------------------------------------------------------------------------------------------------
def separate_cell_points_and_faces(polydata, face_cell_id, point_cell_id):
    """
    This method separates in dictionaries the points and faces of each cell. 
    Then this dictionaries can be used to recreate the individual cells

    Parameters:
    -----------

    polydata: vtkPolyData
        The mesh as a vtkPolyData object

    face_cell_id: np.array()
        The id cell to which each face belongs

    point_cell_id: np.array()
        The id cell to which each node belongs

    Returns:
    --------
    face_dic: dict
        A dictionary where the keys are the cell ids and the values are the faces belonging to that cell

    point_dic: dict
        A dictionary where the keys are the cell ids and the values are the points belonging to that cell

    """
    
    #Get the number of cells
    num_cells =  np.unique(face_cell_id).shape[0]  

    #Create a dictionnary where for each cell will be stored in lists all the triangles that
    #belong to that cell, the key is the cell id
    face_dic = {k: [] for k in range(num_cells)}

    #Same than the dictionnary above, here we store all the point IDs of the points belonging 
    #to a given cell
    point_dic = {k: [] for k in range(num_cells)}

    #Loop over the points Ids and store in which cell they belong
    for pointId in range(polydata.GetNumberOfPoints()):
        try:
            point_dic[point_cell_id[pointId]].append(pointId)
        except IndexError:
            continue

    #Loop over all the triangles and store them in the cell dic based on their cellIDs
    for faceId in range(polydata.GetNumberOfCells()):
        #Extract the points of the face
        face_points = [polydata.GetCell(faceId).GetPointId(i) for i in range(polydata.GetCell(faceId).GetNumberOfPoints())]
        face_dic[face_cell_id[faceId]].append(face_points)

    return face_dic, point_dic
#---------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------
def write_unstructured_grid(
        polydata, 
        face_dic, 
        point_dic, 
        output_file_path, 
        label_of_the_cell = None
    ):
    """
    This method directly writes a file at the legacy unstructured grid vtk format. 

    Parameters:
    -----------

    polydata: vtkPolyData
        The mesh as a vtkPolyData object

    face_dic: dict
        A dictionary where the keys are the cell ids and the values are the faces belonging to that cell

    point_dic: dict
        A dictionary where the keys are the cell ids and the values are the points belonging to that cell

    output_file_path:
        The path where the unstructured grid will be written


    Returns:
    --------

    None
    """

    #Create the file
    f = open(output_file_path, "w")

    #Write the header 
    f.write("# vtk DataFile Version 4.2\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {} float\n".format(polydata.GetNumberOfPoints()))

    #Loop over the points of the polydata
    vtk_points = polydata.GetPoints()
    
    #Convert the point coordinates to a list
    point_lst = [list(vtk_points.GetPoint(i)) for i in range(vtk_points.GetNumberOfPoints())]

    #Write the points in the list
    for i in range(point_lst.__len__()):
        for point_coord in point_lst[i]:
            f.write("{:.5E} ".format(point_coord))
        if (i + 1) % 3 == 0:  f.write("\n")

    #Compute the number of integer needed to save each cell
    nb_integers = face_dic.__len__() * 2 
    for cell_id in face_dic.keys():
        nb_integers += sum([len(face_dic[cell_id][i]) + 1 for i in range(face_dic[cell_id].__len__())])

    #Write the cell data
    f.write("\nCELLS {} {}\n".format(face_dic.__len__(), nb_integers))

    #Loop over the cells
    for cell_id in face_dic.keys():

        #Get the number of faces of the cell
        nb_faces = face_dic[cell_id].__len__()

        #Get the number of integer needed to save the cell
        nb_integers_cell = sum([len(face_dic[cell_id][i]) + 1 for i in range(face_dic[cell_id].__len__())]) + 1

        f.write("{} {} ".format(nb_integers_cell, nb_faces))

        #Loop over the faces of the cell
        for face in face_dic[cell_id]:

            #Write the number of points of the face
            f.write("{} ".format(face.__len__()))

            #Write the points of the face
            for point in face:
                f.write("{} ".format(point))

        #Write a new line
        f.write("\n")
 

    #Indicate the type of cell
    f.write("\nCELL_TYPES {}\n".format(face_dic.__len__()))
    for cell_id in face_dic.keys():
        f.write("42\n")
    
    # Add CELL_DATA section with metadata, if a label is provided
    if label_of_the_cell is not None:
        f.write("CELL_DATA {}\n".format(face_dic.__len__()))
        f.write("SCALARS cell_ids int\nLOOKUP_TABLE default\n")
        for _ in range(len(face_dic)):
            f.write("{}\n".format(label_of_the_cell))

    f.close()

#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def get_cell_id_from_path(path):
    """
    Extract the cell_id from a path of the form 'path/cell_{cell_id}.stl'.

    Parameters:
    -----------
    path: str
        The path string to extract the cell_id from.

    Returns:
    --------
    cell_id: int
        The extracted cell_id or None if the name doesn't contain numbers.
    """
    filename = path.split('/')[-1]  # Get the last part of the path, which is the filename
    cell_id_str = filename.split('_')[-1]  # Get the last part of the filename after '_'
    try:
        cell_id = int(cell_id_str.split('.')[0])  # Remove the file extension and convert to int
        return cell_id
    except:
        return None
#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def convert_to_vtk(input_dir: str, output_dir: str):
    """
    This function converts .stl or .ply files in a directory into .vtk format.
    
    Parameters:
    - input_dir (str): The directory containing the .stl or .ply files to be converted.
    - output_dir (str): The directory where the converted .vtk files will be saved.
    
    Returns:
    - str: The path to the directory where the .vtk files were saved.
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(input_dir), desc='Converting files to .vtk'):
        if filename.endswith('.stl') or filename.endswith('.ply'):
            input_file = os.path.join(input_dir, filename)
            
            # Load the input mesh file into a vtkPolyData object
            polydata = get_polydata_from_file(input_file)

            # Get the cell to which each face belongs
            face_cell_id, point_cell_id = get_cell_ids(polydata)

            # Separate the faces and points of each cell
            face_dic, point_dic = separate_cell_points_and_faces(polydata, face_cell_id, point_cell_id)

            # Prepare the output file path
            output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.vtk')  # Change the extension to .vtk

            #Get the cell id from the path
            cell_id = get_cell_id_from_path(input_file)

            #Get the cells unstructured grids
            write_unstructured_grid(polydata, face_dic, point_dic, output_file, label_of_the_cell = cell_id)

            
    return output_dir

#---------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------
def merge_vtk_files(input_dir: str, scale_factor: float = 1e-6):
    """
    This function merges multiple .vtk files in a directory into a single .vtk file.
    
    Parameters:
    - input_dir (str): The directory containing the .vtk files to be merged.
    - scale_factor (float, optional): The scale factor to be applied to the data. Defaults to 1e-6.
    
    Returns:
    - str: The path to the merged .vtk file.
    - Dict[int, int]: A dictionary mapping sequence indices to labels.
    """
    
    # Create an append filter to combine multiple data sets
    appendFilter = vtk.vtkAppendFilter()
    cell_labels = {}
    index = 0

    for filename in tqdm(os.listdir(input_dir), desc='Merging .vtk files'):
        if filename.endswith('.vtk'):
            input_file = os.path.join(input_dir, filename)
            
            # Read the .vtk file
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(input_file)
            reader.Update()

            # Apply scaling
            transform = vtk.vtkTransform()
            transform.Scale(scale_factor, scale_factor, scale_factor)

            transformFilter = vtk.vtkTransformFilter()
            transformFilter.SetInputConnection(reader.GetOutputPort())
            transformFilter.SetTransform(transform)
            transformFilter.Update()

            # If the filename contains "large_mesh", assign label 4, else assign 0
            # TODO: Make this more general
            label = 1 if 'large_mesh' in filename else 0

            # Add the label to cell_labels dictionary with sequence index as the key
            cell_labels[index] = label
            index += 1

            # Add the data to the append filter
            appendFilter.AddInputData(transformFilter.GetOutput())

    # Combine the data sets
    appendFilter.Update()

    # Write the combined data to a .vtk file
    output_file = os.path.join(input_dir, 'merged.vtk')
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(appendFilter.GetOutput())
    writer.SetFileTypeToASCII()  # To write ASCII file. Remove this line to write binary file.
    writer.SetFileVersion(42)  # Setting file version
    writer.Write()
    
    return output_file, cell_labels
#---------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------
def add_cell_data_to_vtk(vtk_file, cell_labels):
    num_cells = len(cell_labels)
    with open(vtk_file, 'a') as f:
        # Write the cell data
        f.write("CELL_DATA {}\n".format(num_cells))
        f.write("FIELD FieldData 1\n")
        f.write("cell_type_id 1 {} int\n".format(num_cells))

        # Write the cell_type_id for each cell
        for i in range(num_cells):
            # If the cell is in the dictionary, write its label, else write 0
            f.write("{} ".format(cell_labels.get(i, 0)))
            if (i + 1) % 9 == 0:  # Start a new line every 9 cells
                f.write("\n")
#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def convert_cell_labels_to_meshes(
    img: np.ndarray[float],
    voxel_resolution: np.ndarray[float],
    smoothing_iterations: Optional[int] = 1,
    output_directory: Optional[str] = 'output',
    pad_width: Optional[int] = 10,
) -> List[trimesh.Trimesh]:
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
        mesh_lst (List[trimesh.Trimesh]):
            A list of triangular meshes, in the trimesh format.
    
    """
    # Add padding to the image
    img_padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

    mesh_lst = []
    label_ids = np.unique(img_padded)
    
    meshes_folder = os.path.join(output_directory, 'cell_meshes')
    if not os.path.exists(meshes_folder):
        os.makedirs(meshes_folder)

    for label_id in label_ids:
        if label_id == 0: continue
        
        # Initial mesh - Marching Cubes algorithm
        surface = label_to_surface(img_padded == label_id)
        points, faces = surface[0], surface[1]
        points = (points - np.array([pad_width, pad_width, pad_width])) * voxel_resolution

        # Mesh smoothing
        cell_surface_mesh = trimesh.Trimesh(points, faces)
        cell_surface_mesh = trimesh.smoothing.filter_laplacian(cell_surface_mesh, iterations=smoothing_iterations)

        mesh_lst.append(cell_surface_mesh)

    return mesh_lst
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
def create_shell_from_image(
    labeled_image: np.ndarray[int],
    cell_idxs: List[int],
    voxel_size: np.ndarray,
    smoothing_iters: Optional[int] = 10,
    dilation_iters: Optional[int] = 3, 
    closing_iters: Optional[int] = 2
) -> trimesh.Trimesh:
    """
    Create shell mesh starting from a labeled 3D image.

    Parameters:
    -----------

    labeled_image: (np.ndarray[int])
        A 3D image of segmented cells.
    cell_idxs: (List[int])
        List of cell ids to envelope in the shell.
    voxel_size: (np.ndarray[float])
        The size of the voxels in the image along x,y,z dimensions.
    smoothing_iters (Optional[int] = 10):
        The number of smoothing iterations to apply to created meshes. Defaults to 10.
    dilation_iters (Optional[int] = 3): 
        The number of iterations for the dilation operation. Defaults to 3.
    closing_iters (Optional[int] = 2): 
        The number of iterations for the closing operation. Defaults to 2.
    
    Returns:
    --------

    (trimesh.Trimesh):
        The mesh associated to the outer shell.
    """
    # Create a shell mask with all labels in the list
    shell_mask = np.isin(labeled_image, cell_idxs)

    # Apply dilation and closing to the big mask
    shell_mask = binary_dilation(shell_mask, iterations=dilation_iters)
    shell_mask = binary_closing(shell_mask, iterations=closing_iters)

    # Create mesh from voxelized image
    shell_mesh = convert_cell_labels_to_meshes(
        shell_mask, voxel_resolution=voxel_size, smoothing_iterations=smoothing_iters
    )

    return shell_mesh[0] 
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
def create_shell_from_meshes(
    meshes: List[trimesh.Trimesh],
    cell_idxs: List[int],
    cell_stats_data_path: str,
    w_displacement: Optional[bool] = True
) -> trimesh.Trimesh:
    """
    Create shell mesh starting from a meshes of the single cells.

    Parameters:
    -----------

    meshes: (List[trimesh.Trimesh])
        The list of meshes to envelope in the shell.
    cell_idxs: (List[int])
        List of cell ids to envelope in the shell.
    cell_stats_data_path: (str)
        The path to the cell statistics dataframe, needed to extract cell neighbors information.
    w_displacement: (Optional[bool] = True)
        If `True`, the shell mesh will be created after displacing the relative point cloud
        along the directions defined by the normals to the points, to be sure that all the 
        cell meshes are well contained in the envelope.

    Returns:
    --------

    (trimesh.Trimesh):
        The mesh associated to the outer shell.
    """

    assert len(meshes) == len(cell_idxs), f"The number of meshes {len(meshes)} differs from the number of cell ids {len(cell_idxs)}."

    # Get list of cell neighbors
    neighbors_lst = get_cell_neighbors(
        path_to_stats_df=cell_stats_data_path,
        cell_ids=cell_idxs
    )

    # NOTE: neighbor ids are now referred to the whole sample. Hence we need to: 
    # - filter out neighbor ids that are not considered for the patch
    # - map these ids to the ones relative to the current patch of cells
    idxs_map = dict(zip(cell_idxs, np.arange(len(meshes))))
    mapped_neighbors_lst = []
    for cell_neighbors in neighbors_lst:
        mapped_neighbors_lst.append([idxs_map[idx] for idx in cell_neighbors if idx in cell_idxs])
    
    # Convert meshes in ExtendedTrimesh format
    extended_meshes = []
    for i, mesh in enumerate(meshes):
        extended_meshes.append(
            ExtendedTrimesh(neighbors=neighbors_lst[i], vertices=mesh.vertices, faces=mesh.faces)
        )

    # Initialize an `OuterShell` object
    outer_shell = OuterShell(
        meshes=extended_meshes, 
        neighbors_lst=mapped_neighbors_lst
    )

    # Build outer shell mesh
    outer_shell.generate_outer_shell(displace_points=w_displacement)

    return outer_shell.mesh

#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
def create_and_export_meshes(
        cell_labels: list, 
        image_path: str, 
        output_dir: str,
        voxel_resolution: np.ndarray, 
        path_to_cell_stats_df: str,
        make_shell: bool = True, 
        shell_type: Literal["from_image", "from_mesh"] = "from_mesh",
        displace_shell_pts: Optional[bool] = True,
        smoothing_iterations: int = 10,
        dilation_iterations: int = 3, 
        closing_iterations: int = 2
    ) -> str:
    """
    This function creates and exports meshes for each label in a given image.
    
    Parameters:
    cell_labels (list): 
        A list of labels to be processed.
    image_path (str): 
        The path to the image file.
    output_dir (str): 
        The directory where the output will be saved.
    voxel_resolution (np.ndarray): 
        The resolution of the voxels in the image.
    path_to_cell_stats_df: (str)
        The path to the cell statistics dataframe, needed to extract cell neighbors information
        to generate the shell mesh.
    make_shell (bool, optional): 
        Whether to create a shell when creating the meshes. Defaults to True.
    shell_type(Literal["from_image", "from_mesh"], optional): 
        Whether to create the outer shell starting from voxelized image or from cell meshes. 
        The first approach is definitely faster but the second allows to get an outer shell that
        adhere more tightly to the cell meshes.
    displace_shell_pts: (bool, optional)
        If `True`, the shell mesh will be created after displacing the relative point cloud
        along the directions defined by the normals to the points, to be sure that all the 
        cell meshes are well contained in the envelope.
    smoothing_iterations (int, optional):
        The number of smoothing iterations to apply to created meshes. Defaults to 10.
    dilation_iterations (int, optional): 
        The number of iterations for the dilation operation. Defaults to 3.
    closing_iterations (int, optional): 
        The number of iterations for the closing operation. Defaults to 2.
    
    Returns:
    (str): 
        The path to the directory where the meshes were saved.
    """
    
    print('-------------------------------------------')
    print('Creating meshes from labeled img...')

    # Read the image
    if image_path.endswith('.npy'):
        labeled_img = np.load(image_path)
    else:
        labeled_img = io.imread(image_path)
    
    # Initialize an empty mesh list
    mesh_lst = []
    
    for label in tqdm(cell_labels, desc='Converting labels to meshes'):
        # Create a boolean mask for the current label
        mask = np.array(labeled_img == label).astype(int)

        # Create a mesh from the mask using trimesh
        mesh = convert_cell_labels_to_meshes(
            mask, voxel_resolution=voxel_resolution, smoothing_iterations=smoothing_iterations
        )
        mesh_lst.append(mesh[0])

    # Make a combined mesh
    print(f"Generating shell mesh {shell_type.replace('_', ' ')}...")
    if make_shell:
        if shell_type == "from_image":
            big_mesh = create_shell_from_image(
                labeled_image=labeled_img,
                cell_ids=cell_labels,
                voxel_size=voxel_resolution,
                smoothing_iters=smoothing_iterations,
                dilation_iters=dilation_iterations,
                closing_iters=closing_iterations
            )
        elif shell_type == "from_mesh":
            big_mesh = create_shell_from_meshes(
                meshes=mesh_lst,
                cell_idxs=cell_labels,
                cell_stats_data_path=path_to_cell_stats_df,
                w_displacement=displace_shell_pts
            )

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export all the meshes
    for i, cell in enumerate(cell_labels):
        mesh_file_path = os.path.join(output_dir, f'cell_{cell}.stl')
        mesh_lst[i].export(mesh_file_path)
        
    if make_shell:
        big_mesh_file_path = os.path.join(output_dir, 'large_mesh.stl')
        big_mesh.export(big_mesh_file_path)

    return output_dir
#---------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------
def mesh_process_clean(
    label_path: str, 
    output_dir: str, 
    label_list: list, 
    voxel_resolution: np.ndarray, 
    path_to_cell_stats_df: str,
    scale_factor: float = 1e-6, 
    min_edge_length: float = 0.9, 
    make_shell: bool = True,
    shell_type: Literal["from_image", "from_mesh"] = "from_mesh",
    displace_shell_pts: Optional[bool] = True,    
    inter_meshes: bool = False
):
    """
    This function processes a mesh by cleaning, remeshing, converting to vtk format, merging, and adding cell data for SimuCell3D.
    
    Parameters:
    label_path (str): 
        The path to the label data.
    output_dir (str): 
        The directory where the output will be saved.
    label_list (list): 
        A list of labels to be processed.
    voxel_resolution (np.ndarray): 
        The resolution of the voxels in the image.
    path_to_cell_stats_df: (str)
        The path to the cell statistics dataframe, needed to extract cell neighbors information
    scale_factor (float, optional): 
        The scale factor to be used when merging vtk files. Defaults to 1e-6.
    min_edge_length (float, optional): 
        The minimum edge length to be used when remeshing (in micrometers). Defaults to 0.9.
    make_shell (bool, optional): 
        Whether to create a shell when creating the meshes. Defaults to True.
    shell_type(Literal["from_image", "from_mesh"], optional): 
        Whether to create the outer shell starting from voxelized image or from cell meshes. 
        The first approach is definitely faster but the second allows to get an outer shell that
        adhere more tightly to the cell meshes.
    displace_shell_pts: (bool, optional)
        If `True`, the shell mesh will be created after displacing the relative point cloud
        along the directions defined by the normals to the points, to be sure that all the 
        cell meshes are well contained in the envelope.
    inter_meshes (bool, optional): 
        Whether to keep the intermediate mesh files. If False, these files will be deleted. Defaults to True.
    
    Returns:
    - str: The path to the final merged vtk file.
    """
    
    # Create the meshes
    unclean_mesh_dir = create_and_export_meshes(
        cell_labels=label_list, 
        image_path=label_path, 
        output_dir=output_dir, 
        path_to_cell_stats_df=path_to_cell_stats_df,
        voxel_resolution=voxel_resolution, 
        make_shell=make_shell,
        shell_type=shell_type,
        displace_shell_pts=displace_shell_pts
    )

    # Clean the meshes for the first time
    print('-------------------------------------------')
    print('First mesh cleaning...')
    first_clean_mesh_dir = clean_meshes_in_directory(
        unclean_mesh_dir, 
        output_directory=os.path.join(unclean_mesh_dir, 'first_cleaned_meshes')
    )

    # Remesh the cleaned meshes
    print('-------------------------------------------')
    print('Remeshing...')
    remeshed_directory = remesh(
        first_clean_mesh_dir, 
        output_dir=os.path.join(unclean_mesh_dir, 'remeshed_meshes'), 
        min_edge_length=min_edge_length
    )

    # Clean the remeshed meshes
    print('-------------------------------------------')
    print('Second mesh cleaning...')
    second_clean_mesh_dir = clean_meshes_in_directory(
        remeshed_directory, 
        output_directory=os.path.join(unclean_mesh_dir, 'second_cleaned_meshes')
    )
    
    # Create the vtk files
    print('-------------------------------------------')
    print('Getting `.vtk` files...')
    vtk_directory = convert_to_vtk(
        second_clean_mesh_dir, 
        output_dir=os.path.join(unclean_mesh_dir, 'vtk_files')
    )
    
    # Scale and merge the vtk files
    print('-------------------------------------------')
    print('Preparing file for simulation...')
    merged_file, label_dict = merge_vtk_files(vtk_directory, scale_factor=scale_factor)
    
    # Add cell data to the merged vtk file
    add_cell_data_to_vtk(merged_file, label_dict)

    # Copy merged file to output directory
    shutil.copy(merged_file, output_dir)

    # Remove intermediate mesh directories if inter_meshes is False
    if not inter_meshes:
        shutil.rmtree(first_clean_mesh_dir)
        shutil.rmtree(remeshed_directory)
        shutil.rmtree(second_clean_mesh_dir)

    return merged_file
#---------------------------------------------------------------------------------------------------------------
    


#---------------------------------------------------------------------------------------------------------------
def string_to_array(input_string):
    # Split the string into substrings
    substrings = input_string.split()

    # Convert each substring to an integer and store in a new list
    numbers = [int(s) for s in substrings]

    # Return the list of numbers
    return numbers
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
def read_cell_ids(input_data):
    # Load the data into a DataFrame
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("input_data must be a path (str) or a DataFrame (pd.DataFrame)")

    return df['cell_id'].tolist()
#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def convert_filtered_meshes(
        input_dir: str, 
        output_dir: str, 
        filtered_cell_ids: list
) -> None:
    """
    Copy meshes file from source (input_dir) to destination (output_dir) directory.
    After being copied the mesh files are converted from their original formal (e.g., 'stl')
    to the 'vtk' format.

    Parameters:
    -----------
    input_dir: (str)
        The path to the source directory.
    
    output_dir: (str)
        The path to the destination directory.

    filtered_cell_ids: (list)
        A list containing the indices of "filtered" cells. That is, all the cells whose id 
        is not in this list are discarded.
    """

    # Create the output directory
    copy_dir = os.path.join(output_dir, 'stl_files')
    os.makedirs(copy_dir, exist_ok=True)

    # Get the list of files in the input directory
    files = os.listdir(input_dir)

    # Iterate through the files
    for file in tqdm(files, desc='Moving mesh files'):
        # Get the cell ID from the file name in the form of cell_{id}.stl
        cell_id = int(file.split('_')[1].split('.')[0])

        # If the cell ID is in the filtered_cell_ids list, copy the file to the output directory
        if cell_id in filtered_cell_ids:
            shutil.copy(os.path.join(input_dir, file), os.path.join(copy_dir, file))
    
    convert_to_vtk(input_dir=copy_dir, output_dir=os.path.join(output_dir, "vtk_cells"))
#---------------------------------------------------------------------------------------------------------------



def main():
    """
    This is the main function that calls the mesh_process_clean function to process and clean mesh data.
    
    Please ensure that the path to label is preprocessed (perform erosion/dilation/removal of detached regions etc...)
    
    Obtain the labels for a geometry using paraview: see ___ for a tutorial. If you do not have meshes generated 
    (such as from statistics algorithms, you can)

    Example usage:
    ```python
    voxel_resolution = np.array([0.21, 0.21, 0.39])
    label_path = '/path/to/processed_labels.npy'
    label_list = [111, 112, 114, 125, 126, 129, 134, 137, 139, 140, 143] # Use string_to_array if you have a space separated string
    output_dir = '/path/to/output/Experiment_20_condensed'
    mesh_process_clean(
        label_path=label_path, 
        output_dir=output_dir, 
        label_list=label_list, 
        voxel_resolution=voxel_resolution, 
        scale_factor=1e-6, 
        make_shell=True
    )
    ```
    """

    voxel_resolution = np.array([1., 1., 1.])
    label_list = np.array([1, 2, 3, 4])
    label_path = 'path/to/processed/labels.tif'
    output_dir = 'path/to/output'
    
    
    # Call the mesh_process_clean function
    mesh_process_clean(
        label_path=label_path, 
        output_dir=output_dir, 
        label_list=label_list, 
        voxel_resolution=voxel_resolution, 
        scale_factor=1, 
        make_shell=True
    )

if __name__ == "__main__":
    main()
