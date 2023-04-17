import napari
from skimage import io
import numpy as np
import trimesh as tm
import os
import argparse

#------------------------------------------------------------------------------
def overlay_meshes_and_labels(label_image_path, output_folder, cell_id=None, filtered=False):
    # Load the label image
    label_image = np.load(label_image_path)
    
    # Choose the appropriate mesh folder
    if filtered:
        mesh_folder = os.path.join(output_folder, 'filtered_cell_meshes')
        mesh_files = sorted([os.path.join(mesh_folder, f) for f in os.listdir(mesh_folder) if f.startswith('filtered_') and f.endswith('.stl')])

    else:
        mesh_folder = os.path.join(output_folder, 'cell_meshes')
        mesh_files = sorted([os.path.join(mesh_folder, f) for f in os.listdir(mesh_folder) if f.startswith('cell_') and f.endswith('.stl')])

    if cell_id is not None:
        mesh_files = [os.path.join(mesh_folder, f'cell_{cell_id - 1}.stl')]
        # Extract the single cell label
        label_image = (label_image == cell_id).astype(int)
    
    # Create a viewer
    viewer = napari.Viewer()

    # Add the label image to the viewer
    viewer.add_labels(label_image, name='Label Image', scale=[0.21, 0.21, 0.39])
    
    # Add each mesh to the viewer
    for mesh_file in mesh_files:
        mesh = tm.load(mesh_file)
        vertices, faces = mesh.vertices, mesh.faces
        viewer.add_surface((vertices, faces), name=os.path.basename(mesh_file))

    # Start the napari GUI
    napari.run()
#--------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View cell meshes and labels.')
    parser.add_argument('output_directory', type=str, help='Path to the output directory')
    parser.add_argument('--cell_id', type=int, help='View a specific cell mesh only')
    parser.add_argument('--filtered', action='store_true', help='View filtered cell meshes only')
    args = parser.parse_args()

    processed_labels_path = os.path.join(args.output_directory, 'processed_labels.npy')

    overlay_meshes_and_labels(processed_labels_path, args.output_directory, args.cell_id, args.filtered)

# if __name__ == '__main__':
#     # Load the processed labels (change the path to your output directory)
#     output_directory = '/Users/antanas/BC_Project_s_5_e_2_d_3'
#     processed_labels_path = os.path.join(output_directory, 'processed_labels.npy')
#     processed_labels = np.load(processed_labels_path)
    
#     overlay_meshes_and_labels(processed_labels_path, output_directory, cell_id = 31)

