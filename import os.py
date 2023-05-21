import os
import pandas as pd
import numpy as np
import trimesh
import napari
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap

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


def create_vtk_file(mesh_dir, statistics_csv, statistic_column, output_vtk):
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
    """

    # Load statistics from the CSV file
    statistics_df = pd.read_csv(statistics_csv)

    # Get the statistic values for the cells
    cell_statistic_values = statistics_df[statistic_column].values

    # Normalize the statistic values to the range [0, 1]
    normalized_statistic_values = (cell_statistic_values - cell_statistic_values.min()) / (cell_statistic_values.max() - cell_statistic_values.min())

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
        for cell_id, vertex_value in zip(statistics_df['cell_id'], normalized_statistic_values):
            if cell_id-1 == 0:
                continue
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



if __name__ == '__main__':
    # Replace these with the appropriate paths
    mesh_dir = '/Users/antanas/BC_Project/Control_Segmentation_final/Total_meshes/Validated_labels_final_Franzi_11w_s_10_e_2_d_3/cell_meshes'
    statistics_csv = '/Users/antanas/BC_Project/Control_Segmentation_final/Total_meshes/Validated_labels_final_Franzi_11w_s_10_e_2_d_3/all_cell_statistics.csv'
    statistic_column = 'cell_volume'

    #visualize_statistics(mesh_dir, statistics_csv, statistic_column)
    create_vtk_file(mesh_dir, statistics_csv, statistic_column, '/Users/antanas/BC_Project/Control_Segmentation_final/BC_control_2_s_8_e_3_d_4/vtk_meshes_11w_all.vtk')