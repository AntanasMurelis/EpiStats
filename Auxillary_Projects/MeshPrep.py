import trimesh
import numpy as np
from collections import Counter
from itertools import combinations
import pymeshfix
import os
import sys
import pymeshlab
from napari_process_points_and_surfaces import label_to_surface
from tqdm import tqdm
import vtk 
from sys import argv
import numpy as np 
from os import path


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
def remove_non_manifold_faces_and_fill_holes(mesh: trimesh.Trimesh):
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
def remesh(input_dir, output_dir, min_edge_length=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
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
            ms.apply_filter('generate_surface_reconstruction_ball_pivoting',ballradius = pymeshlab.Percentage(5))
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
def clean_meshes_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith(".ply") or filename.endswith(".stl"):
            mesh_path = os.path.join(input_directory, filename)
            mesh = trimesh.load_mesh(mesh_path)
            cleaned_mesh = remove_non_manifold_faces_and_fill_holes(mesh)
            cleaned_filename = 'cleaned_' + filename
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)
            
            cleaned_mesh_path = os.path.join(output_directory, cleaned_filename)
            cleaned_mesh.export(cleaned_mesh_path)  # Save the cleaned mesh to a new file
            
            
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
        point_dic[point_cell_id[pointId]].append(pointId)


    #Loop over all the triangles and store them in the cell dic based on their cellIDs
    for faceId in range(polydata.GetNumberOfCells()):

        #Extract the points of the face
        face_points = [polydata.GetCell(faceId).GetPointId(i) for i in range(polydata.GetCell(faceId).GetNumberOfPoints())]
        face_dic[face_cell_id[faceId]].append(face_points)

    return face_dic, point_dic
#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def write_unstructured_grid(polydata, face_dic, point_dic, output_file_path):
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
#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
def convert_to_vtk(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.stl') or filename.endswith('.ply'):
            input_file = os.path.join(input_dir, filename)
            
            # Load the input mesh file into a vtkPolyData object
            polydata = get_polydata_from_file(input_file)

            # Get the cell to which each face belongs
            face_cell_id, point_cell_id = get_cell_ids(polydata)

            # Separate the faces and points of each cell
            face_dic, point_dic = separate_cell_points_and_faces(polydata, face_cell_id, point_cell_id)

            # Prepare the output file path
            output_file = os.path.splitext(input_file)[0] + '.vtk'  # Change the extension to .vtk

            # Write the vtkUnstructuredGrid to the output file
            write_unstructured_grid(polydata, face_dic, point_dic, output_file)
            
    return input_dir
#---------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------------
def write_legacy_vtk_file(unstructured_grid, output_file_path):
    with open(output_file_path, 'w') as f:
        # Write the header
        f.write("# vtk DataFile Version 4.2\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Get the points from the unstructured grid
        points = unstructured_grid.GetPoints()
        num_points = points.GetNumberOfPoints()
        f.write("POINTS {} float\n".format(num_points))

        # Write the points to the file
        for i in range(num_points):
            point = points.GetPoint(i)
            f.write("{:.5E} {:.5E} {:.5E} ".format(point[0], point[1], point[2]))
            if (i + 1) % 3 == 0:  # Start a new line every 3 points
                f.write("\n")

        # Write the cells to the file
        num_cells = unstructured_grid.GetNumberOfCells()
        total_num_points = 0
        for i in range(num_cells):
            cell = unstructured_grid.GetCell(i)
            cell_points = cell.GetPointIds()
            total_num_points += cell_points.GetNumberOfIds()

        f.write("CELLS {} {}\n".format(num_cells, total_num_points + num_cells))  # One extra for each cell for the number of points in that cell

        for i in range(num_cells):
            cell = unstructured_grid.GetCell(i)
            cell_points = cell.GetPointIds()
            num_cell_points = cell_points.GetNumberOfIds()
            f.write("{} ".format(num_cell_points))
            for j in range(num_cell_points):
                f.write("{} ".format(cell_points.GetId(j)))
            f.write("\n")


        # Write the cell types to the file
        f.write("CELL_TYPES {}\n".format(num_cells))
        for _ in range(num_cells):
            f.write("42\n")  # 42 is the VTK type for polygons

        # Write the cell data
        f.write("CELL_DATA {}\n".format(num_cells))
        f.write("FIELD FieldData {}\n".format(num_cells))
        f.write("cell_type_id 1 {} int\n".format(num_cells))

        # Write the cell_type_id for each cell
        for i in range(num_cells):
            f.write("0 ")
            if (i + 1) % 9 == 0:  # Start a new line every 9 cells
                f.write("\n")
#---------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------------
def merge_vtk_files(input_dir, scale_factor=1e-6):
    # Create an append filter to combine multiple data sets
    appendFilter = vtk.vtkAppendFilter()
    cell_labels = {}
    index = 0

    for filename in os.listdir(input_dir):
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
            label = 4 if 'large_mesh' in filename else 0

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
    img,
    voxel_resolution,
    smoothing_iterations=1,
    output_directory='output',
    pad_width=10,
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
        
        surface = label_to_surface(img_padded == label_id)
        points, faces = surface[0], surface[1]
        points = (points - np.array([pad_width, pad_width, pad_width])) * voxel_resolution

        cell_surface_mesh = trimesh.Trimesh(points, faces)
        cell_surface_mesh = trimesh.smoothing.filter_laplacian(cell_surface_mesh, iterations=smoothing_iterations)

        mesh_lst.append(cell_surface_mesh)

    return mesh_lst
#---------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------
def create_and_export_meshes(cell_labels: list, image_path: str, output_dir:str, voxel_resolution: np.ndarray) -> str:
    # Read the image
    labels = np.load(image_path)
    
    # Initialize an empty mesh list
    labels_list = []
    for label in cell_labels:
        # Create a boolean mask for the current label
        mask = np.array(labels == label).astype(int)

        # Create a mesh from the mask using trimesh
        mesh = convert_cell_labels_to_meshes(mask, voxel_resolution=voxel_resolution, smoothing_iterations=10)
        labels_list.append(mesh)

    # Make a combined mesh
    big_mask = np.isin(labels, cell_labels)
    big_mesh = convert_cell_labels_to_meshes(big_mask, voxel_resolution=voxel_resolution, smoothing_iterations=10)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export all the meshes
    for i, cell in enumerate(cell_labels):
        labels_list[i][0].export(os.path.join(output_dir, f'cell_{cell-1}.stl'))
    big_mesh[0].export(os.path.join(output_dir, 'large_mesh.stl'))

    # Return the directory where the meshes were saved
    return output_dir
#---------------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------------
def mesh_process_clean(label_path:str, output_dir:str, label_list:list, voxel_resolution:np.ndarray, scale_factor:float = 1e-6) -> str:
    
    
    # Create the meshes
    #unclean_mesh_dir = create_and_export_meshes(cell_labels=label_list, image_path=label_path, output_dir=output_dir, voxel_resolution=voxel_resolution)
    
    
    # Clean the meshes
    unclean_mesh_dir = '/Users/antanas/BC_Project/Experiment_11'
    
    
    #first_clean_mesh_dir = clean_meshes_in_directory(unclean_mesh_dir, output_directory=os.path.join(unclean_mesh_dir, 'first_cleaned_meshes'))
    #first_clean_mesh_dir = unclean_mesh_dir
    
    
    remeshed_directory = remesh(unclean_mesh_dir, output_dir=os.path.join(unclean_mesh_dir, 'remeshed_meshes'), min_edge_length=0.6)
    second_clean_mesh_dir = clean_meshes_in_directory(remeshed_directory, output_directory=os.path.join(remeshed_directory, 'second_cleaned_meshes'))
    
    
    # Create the vtk files
    vtk_directory = convert_to_vtk(second_clean_mesh_dir)
    # Scale and vtk merge
    merged_file, label_dict = merge_vtk_files(vtk_directory)
    # Add cell data
    add_cell_data_to_vtk(merged_file, label_dict)

    return merged_file
#---------------------------------------------------------------------------------------------------------------
    
    

if __name__ == "__main__":

    voxel_resolution = np.array([0.21, 0.21, 0.39])
    output_dir = '/Users/antanas/BC_Project/Experiment_11'
    label_path = '/Users/antanas/BC_Project/Control_Segmentation_final/BC_control_s_5_e_2_d_5/processed_labels.npy'
    label_list = [530, 163, 110, 146, 594, 109, 138, 115, 157, 533, 94, 145, 155, 164, 200, 129, 178, 201, 241, 522]
    
    mesh_process_clean(label_path=label_path, output_dir=output_dir, label_list=label_list, voxel_resolution=voxel_resolution, scale_factor=1e-6)    
    

