import trimesh
import napari
import numpy as np
from collections import Counter
from itertools import combinations
import pymeshfix


def create_cube_with_extra_face(filename=None):
    """
    This function creates a cube mesh with an extra face. Change the extra vertex/face to create a 
    different test case.

    Parameters:
        filename (str): Optional; The name of the file where the created cube will be saved. If None, the cube is not saved to a file.

    Returns:
        cube (trimesh.Trimesh): The created cube mesh.
    """
    # Define the 8 vertices of the cube
    vertices = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
        [1.5, 1.5, 1.5]  # Extra vertex
    ])

    # Define the 12 triangles composing the cube
    faces = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [4, 5, 0],
        [5, 0, 1],
        [6, 7, 4],
        [7, 4, 5],
        [2, 3, 6],
        [3, 6, 7],
        [0, 2, 4],
        [2, 4, 6],
        [1, 3, 5],
        [3, 5, 7],
        [0, 1, 2]  # Extra Face
    ])

    # Create the mesh
    cube = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    if filename is not None:
        # Save the mesh to an STL file
        cube.export(filename)
    else:
        return cube


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

import numpy as np



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

    # Fill holes
    tin.fill_small_boundaries()

    # Clean (removes self intersections)
    tin.clean(max_iters=10, inner_loops=3)

    # Retrieve the cleaned mesh as numpy arrays
    vclean, fclean = tin.return_arrays()

    # Update the original Trimesh object
    mesh.vertices = vclean
    mesh.faces = fclean

    return mesh


    
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

    # Fill the holes
    mesh = fill_holes_pymeshfix(mesh)
    
    return mesh


if __name__ == '__main__':
    
    cube = create_cube_with_extra_face()
    edges_with_more_than_two_faces = get_edges_with_more_than_two_faces(cube)

    print("Edges with more than two faces attached:")
    for edge in edges_with_more_than_two_faces:
        print(edge) 
        
    cube = remove_non_manifold_faces_and_fill_holes(cube)
    
    ###### Visualise cube (faces removed) using napari ######
    # viewer = napari.Viewer()
    # viewer.add_surface([cube.vertices, cube.faces], name='cube')
    # napari.run()
    #########################################################
    
    edges_with_more_than_two_faces = get_edges_with_more_than_two_faces(cube)
    print("Edges with more than two faces attached after cleaning:")
    for edge in edges_with_more_than_two_faces:
        print(edge) 
    else:
        print("No edges with more than two faces attached")

