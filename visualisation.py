import napari
from skimage import io
import numpy as np
import trimesh as tm
import os

#-------------------------------------------------------------------------------------------------------------------------------
# Function to read the meshes from the output directory
def read_meshes(output_directory):
    cell_meshes_directory = os.path.join(output_directory, 'cell_meshes')
    mesh_files = sorted([f for f in os.listdir(cell_meshes_directory) if f.startswith('cell_') and f.endswith('.stl')])
    mesh_lst = [tm.load_mesh(os.path.join(cell_meshes_directory, f)) for f in mesh_files]
    return mesh_lst
#-------------------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------------------------
def overlay_meshes_and_labels(label_image_path, output_folder):
    # Load the label image
    label_image = np.load(label_image_path)
    
    # Load the mesh files
    mesh_folder = os.path.join(output_folder, 'cell_meshes')
    mesh_files = [os.path.join(mesh_folder, f) for f in os.listdir(mesh_folder) if f.endswith('.stl')]
    
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
#--------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Load the processed labels (change the path to your output directory)
    output_directory = '/Users/antanas/GitRepo/EpiStats/test1_s_5_e_1_d_2'
    processed_labels_path = os.path.join(output_directory, 'processed_labels.npy')
    processed_labels = np.load(processed_labels_path)

    overlay_meshes_and_labels(processed_labels_path, output_directory)
