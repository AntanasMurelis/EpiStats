import os
import pandas as pd
import numpy as np
import trimesh
import napari
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from pykdtree.kdtree import KDTree


class ExtendedTrimesh(trimesh.Trimesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_kdtree = KDTree(self.vertices)


    def get_potential_contact_faces(self, other_mesh, distance):
        potential_contact_faces = []
        for face_index in range(len(other_mesh.faces)):
            face = other_mesh.vertices[other_mesh.faces[face_index]]
            centroid = np.mean(face, axis=0).reshape(1, -1)
            dist, idx = self.my_kdtree.query(centroid)
            if dist < distance:
                potential_contact_faces.append(face_index)
                
        # with napari.gui_qt():
        #     viewer = napari.Viewer()
        #     viewer.add_surface((self.vertices, self.faces), name='target')
        #     viewer.add_surface((other_mesh.vertices, other_mesh.faces[potential_contact_faces]), colormap='red', name='contact_faces')
        #     viewer.add_surface((other_mesh.vertices, other_mesh.faces), name='neighbour')
        
        return potential_contact_faces

    def calculate_contact_area(self, other_mesh, distance):
        contact_faces_indices = self.get_potential_contact_faces(other_mesh, distance)
        # Assuming contact area is just the sum of the areas of the contact faces
        
        contact_area = np.sum(other_mesh.area_faces[contact_faces_indices])
        return contact_area

def visualize_statistics(mesh_dir, statistics_csv, statistic_column):
    """
    Visualize 3D cell meshes with colors based on the given statistic column using Napari.

    Parameters:
    -----------
    mesh_dir: str
        The directory containing the mesh files in PLY format (e.g., 'cell_1_mesh.ply').

    statistics_csv: str
        The path to the CSV file containing cell statistics.

    statistic_column: str
        The name of the column in the CSV file to use for coloring the cells.

    Example:
    --------
    ```python
    # Replace these with the appropriate paths
    mesh_dir = 'path/to/meshes'
    statistics_csv = 'path/to/statistics.csv'
    statistic_column = 'cell_volume'

    visualize_with_napari(mesh_dir, statistics_csv, statistic_column)
    ```
    """
    # Load statistics from the CSV file
    statistics_df = pd.read_csv(statistics_csv)

    # Get the statistic values for the cells
    cell_statistic_values = statistics_df[statistic_column].values

    # Normalize the statistic values to the range [0, 1]
    normalized_statistic_values = (cell_statistic_values - cell_statistic_values.min()) / (cell_statistic_values.max() - cell_statistic_values.min())

    # Create a list of tuples, each containing a mesh and its corresponding vertex values
    mesh_vertex_values = []

    for cell_id, vertex_value in zip(statistics_df['cell_id'], normalized_statistic_values):
        mesh_file = os.path.join(mesh_dir, f'cell_{cell_id-1}.stl')
        if os.path.exists(mesh_file):
            trimesh_mesh = trimesh.load_mesh(mesh_file)
            vertex_values = np.full(len(trimesh_mesh.vertices), vertex_value)
            mesh_vertex_values.append((trimesh_mesh, vertex_values))

    # Visualize the meshes with Napari
    with napari.gui_qt():
        viewer = napari.Viewer()

        for mesh, vertex_values in mesh_vertex_values:
            vertices, faces = mesh.vertices, mesh.faces
            viewer.add_surface((vertices, faces, vertex_values), colormap='viridis', blending='translucent', scale=[0.21, 0.21, 0.39])


def create_vtk_file(mesh_dir, statistics_csv, statistic_column, output_vtk, normalize=True):
    """
    Create a VTK file with 3D cell meshes and colors based on the given statistic column.

    Parameters:
    -----------
    mesh_dir: str
        The directory containing the mesh files in PLY format (e.g., 'cell_1.stl').

    statistics_csv: str
        The path to the CSV file containing cell statistics.

    statistic_column: str
        The name of the column in the CSV file to use for coloring the cells.

    output_vtk: str
        The path to save the output VTK file.

    normalize: bool
        Whether to normalize the statistic values to the range [0, 1]. Default is True.
    """

    # Load statistics from the CSV file
    statistics_df = pd.read_csv(statistics_csv)

    # Get the statistic values for the cells
    cell_statistic_values = statistics_df[statistic_column].values

    if normalize:
        # Normalize the statistic values to the range [0, 1]
        cell_statistic_values = (cell_statistic_values - cell_statistic_values.min()) / (cell_statistic_values.max() - cell_statistic_values.min())

    with open(output_vtk, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("3D Cell Meshes\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        all_points = []
        all_cells = []
        all_cell_types = []
        all_point_data = []

        point_offset = 0
        for cell_id, vertex_value in zip(statistics_df['cell_id'], cell_statistic_values):
            mesh_file = os.path.join(mesh_dir, f'cell_{cell_id-1}.stl')
            if os.path.exists(mesh_file):
                
                
                trimesh_mesh = trimesh.load_mesh(mesh_file)
                if abs(trimesh_mesh.volume) < 200:
                    continue
                # Add mesh vertices to the points list
                all_points.extend(trimesh_mesh.vertices)

                # Add mesh faces to the cells list with an offset
                all_cells.extend([[3] + (face + point_offset).tolist() for face in trimesh_mesh.faces])
                point_offset += len(trimesh_mesh.vertices)

                # Set the cell type to 5 (triangle) for all cells
                all_cell_types.extend([5] * len(trimesh_mesh.faces))

                # Assign the normalized statistic value to each vertex
                all_point_data.extend([vertex_value] * len(trimesh_mesh.vertices))

        # Write points
        vtk_file.write(f"POINTS {len(all_points)} float\n")
        for point in all_points:
            vtk_file.write(f"{point[0]} {point[1]} {point[2]}\n")

        # Write cells
        vtk_file.write(f"CELLS {len(all_cells)} {len(all_cells) * 4}\n")
        for cell in all_cells:
            vtk_file.write(f"{cell[0]} {cell[1]} {cell[2]} {cell[3]}\n")

        # Write cell types
        vtk_file.write(f"CELL_TYPES {len(all_cells)}\n")
        for cell_type in all_cell_types:
            vtk_file.write(f"{cell_type}\n")

        # Write point data
        vtk_file.write(f"POINT_DATA {len(all_point_data)}\n")
        vtk_file.write(f"SCALARS {statistic_column} float 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for point_data in all_point_data:
            vtk_file.write(f"{point_data}\n")



import pandas as pd
import numpy as np
import os
import trimesh

def get_contact_faces(cell_id, statistics_csv, mesh_dir, distance):
    """
    Get the faces of a cell mesh that are in contact with specific neighbors identified by their IDs.

    Parameters:
    -----------
    cell_id: int
        The ID of the cell mesh.

    statistics_csv: str
        The path to the CSV file containing cell statistics.

    mesh_dir: str
        The directory containing the mesh files in PLY format (e.g., 'cell_1.stl').

    distance: float
        The distance threshold for determining contact between meshes.

    Returns:
    --------
    dict: Mapping of face indices to colors based on the neighboring cells.
    """

    contact_faces = {}  # To store face indices and colors

    # Read the statistics CSV file
    statistics_df = pd.read_csv(statistics_csv)

    # Get the neighbor IDs for the specified cell ID
    cell_neighbor_ids_str = statistics_df.loc[statistics_df['cell_id'] == cell_id, 'cell_neighbors'].values[0]
    cell_neighbor_ids = list(map(int, cell_neighbor_ids_str.split()))

    # Load the mesh of the specified cell
    cell_mesh_file = os.path.join(mesh_dir, f'cell_{cell_id - 1}.stl')
    if os.path.exists(cell_mesh_file):
        cell_mesh = trimesh.load_mesh(cell_mesh_file)
    else:
        raise FileNotFoundError(f"Mesh file for cell {cell_id} not found.")

    # Loop over the neighbor IDs for the specified cell
    for neighbor_id in cell_neighbor_ids:
        # Load the mesh of the neighbor cell
        neighbor_mesh_file = os.path.join(mesh_dir, f'cell_{neighbor_id - 1}.stl')
        if os.path.exists(neighbor_mesh_file):
            neighbor_mesh = trimesh.load_mesh(neighbor_mesh_file)
            neighbor_mesh = ExtendedTrimesh(neighbor_mesh.vertices, neighbor_mesh.faces)
        else:
            raise FileNotFoundError(f"Mesh file for neighbor cell {neighbor_id} not found.")

        # Get the potential contact faces between the cell mesh and the neighbor mesh
        potential_contact_faces = neighbor_mesh.get_potential_contact_faces(cell_mesh, distance)

        # Assign a color to the contact faces based on the neighboring cell ID
        color = tuple(np.random.rand(3))  # Generate a random RGB color for each neighboring cell

        # Add the contact faces and their corresponding colors to the dictionary
        for face_index in potential_contact_faces:
            contact_faces[face_index] = color

    return contact_faces

def neighbor_color(mesh_file, contact_faces, output_vtk):
    """
    Write a VTK file with colored faces based on the given contact faces dictionary.

    Parameters:
    -----------
    mesh_file: str
        The path to the mesh file (e.g., 'cell_1.stl').

    contact_faces: dict
        Dictionary mapping face indices to colors.

    output_vtk: str
        The path to save the output VTK file.
    """

    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Create a color array for the faces
    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.float32)

    # Assign colors to the contact faces
    for face_index, color in contact_faces.items():
        face_colors[face_index] = color

    # Write the VTK file
    with open(output_vtk, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Colored Faces\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write the points
        vtk_file.write(f"POINTS {len(mesh.vertices)} float\n")
        for vertex in mesh.vertices:
            vtk_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write the faces
        vtk_file.write(f"POLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
        for face in mesh.faces:
            vtk_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        # Write the face colors
        vtk_file.write(f"CELL_DATA {len(mesh.faces)}\n")
        vtk_file.write("SCALARS face_colors float 3\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for color in face_colors:
            vtk_file.write(f"{color[0]} {color[1]} {color[2]}\n")

def neighbor_color(mesh_file, contact_faces, output_vtk):
    """
    Write a VTK file with colored faces based on the given contact faces dictionary.

    Parameters:
    -----------
    mesh_file: str
        The path to the mesh file (e.g., 'cell_1.stl').

    contact_faces: dict
        Dictionary mapping face indices to colors.

    output_vtk: str
        The path to save the output VTK file.
    """

    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Create a color array for the faces
    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.float32)

    # Assign colors to the contact faces
    for face_index, color in contact_faces.items():
        face_colors[face_index] = color

    # Write the VTK file
    with open(output_vtk, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Colored Faces\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write the points
        vtk_file.write(f"POINTS {len(mesh.vertices)} float\n")
        for vertex in mesh.vertices:
            vtk_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write the faces
        vtk_file.write(f"POLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
        for face in mesh.faces:
            vtk_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        # Write the face colors
        vtk_file.write(f"CELL_DATA {len(mesh.faces)}\n")
        vtk_file.write("SCALARS face_colors float 3\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for color in face_colors:
            vtk_file.write(f"{color[0]} {color[1]} {color[2]}\n")
def neighbor_color(mesh_file, contact_faces, output_vtk):
    """
    Write a VTK file with colored faces based on the given contact faces dictionary.

    Parameters:
    -----------
    mesh_file: str
        The path to the mesh file (e.g., 'cell_1.stl').

    contact_faces: dict
        Dictionary mapping face indices to colors.

    output_vtk: str
        The path to save the output VTK file.
    """

    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Create a color array for the faces
    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.float32)

    # Assign colors to the contact faces
    for face_index, color in contact_faces.items():
        face_colors[face_index] = color

    # Write the VTK file
    with open(output_vtk, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Colored Faces\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write the points
        vtk_file.write(f"POINTS {len(mesh.vertices)} float\n")
        for vertex in mesh.vertices:
            vtk_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write the faces
        vtk_file.write(f"POLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
        for face in mesh.faces:
            vtk_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        # Write the face colors
        vtk_file.write(f"CELL_DATA {len(mesh.faces)}\n")
        vtk_file.write("SCALARS face_colors unsigned_char 3\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for color in face_colors:
            vtk_file.write(f"{int(color[0]*255)} {int(color[1]*255)} {int(color[2]*255)}\n")

def neighbor_color(mesh_file, contact_faces, output_vtk):
    """
    Write a VTK file with colored faces based on the given contact faces dictionary.

    Parameters:
    -----------
    mesh_file: str
        The path to the mesh file (e.g., 'cell_1.stl').

    contact_faces: dict
        Dictionary mapping face indices to colors.

    output_vtk: str
        The path to save the output VTK file.
    """

    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Create a color array for the faces
    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.float32)

    # Assign colors to the contact faces
    for face_index, color in contact_faces.items():
        face_colors[face_index] = color

    # Write the VTK file
    with open(output_vtk, 'w') as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Colored Faces\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write the points
        vtk_file.write(f"POINTS {len(mesh.vertices)} float\n")
        for vertex in mesh.vertices:
            vtk_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write the faces
        vtk_file.write(f"POLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
        for face in mesh.faces:
            vtk_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        # Write the face colors
        vtk_file.write(f"CELL_DATA {len(mesh.faces)}\n")
        vtk_file.write("SCALARS face_colors unsigned_char 3\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for color in face_colors:
            vtk_file.write(f"{int(color[0]*255)} {int(color[1]*255)} {int(color[2]*255)}\n")


if __name__ == '__main__':
    
    
    # # Replace these with the appropriate paths
    output_dir = '/Users/antanas/BC_Project/No_Edge_5000/No_edge/Validated_labels_Antanas_Control_s_10_e_2_d_3'
    statistic_column = 'cell_mean_contact_area'
    
    mesh_dir = os.path.join(output_dir, 'cell_meshes')
    statistics_csv = os.path.join(output_dir, 'all_cell_statistics.csv')

    # cell_id = 6
    # contact_faces = get_contact_faces(cell_id, statistics_csv, mesh_dir, distance = 0.7)
    
    # neighbor_color(os.path.join(mesh_dir, f'cell_{cell_id - 1}.stl'), contact_faces, os.path.join(output_dir, f'cell_{cell_id - 1}.vtk'))
    # #visualize_statistics(mesh_dir, statistics_csv, statistic_column)
    create_vtk_file(mesh_dir, statistics_csv, statistic_column, f'{output_dir}/BC_Antanas_Control_{statistic_column}.vtk')