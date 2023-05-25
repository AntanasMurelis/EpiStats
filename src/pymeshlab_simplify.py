import os
import sys
import pymeshlab


####################################################################################################
# Basic implementation of remeshing
####################################################################################################

# def remesh(input_dir, min_edge_length=0.1):
    
#     # Delete old 'remeshed_' files
#     for filename in os.listdir(input_dir):
#         if filename.startswith('remeshed_'):
#             os.remove(os.path.join(input_dir, filename))

#     for filename in os.listdir(input_dir):
#         if filename.endswith('.stl') or filename.endswith('.ply'):
#             input_file = os.path.join(input_dir, filename)
#             ms = pymeshlab.MeshSet()
#             ms.load_new_mesh(input_file)

#             # Set the edge length parameter
#             try:
#                 min_edge_length = float(min_edge_length)
#             except ValueError:
#                 print(f"Invalid minimum edge length: {min_edge_length}")
#                 sys.exit(1)

#             # Set parameters for the isotropic explicit remeshing filter
#             targetlen = pymeshlab.AbsoluteValue(min_edge_length)
#             remesh_par = dict(targetlen=targetlen, iterations=5)

#             # Apply the isotropic explicit remeshing filter
#             ms.apply_filter('meshing_isotropic_explicit_remeshing', **remesh_par)

#             # Save the remeshed file
#             output_file = os.path.join(input_dir, 'remeshed_' + filename)
#             ms.save_current_mesh(output_file)
#---------------------------------------------------------------------------------------------------


####################################################################################################
# Implementation of remeshing with cleaning and repairing
####################################################################################################


def remesh(input_dir, min_edge_length=0.1):
    
    # Delete old 'remeshed_' files
    for filename in os.listdir(input_dir):
        if filename.startswith('remeshed_'):
            os.remove(os.path.join(input_dir, filename))

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
            remesh_par = dict(targetlen=targetlen, iterations=5)

            # Apply the isotropic explicit remeshing filter
            ms.apply_filter('meshing_isotropic_explicit_remeshing', **remesh_par)

            # Clean up the mesh
            ms.apply_filter('meshing_remove_duplicate_vertices')
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_null_faces')

            # Make mesh manifold
            ms.apply_filter('meshing_repair_non_manifold_edges')
            ms.apply_filter('meshing_repair_non_manifold_vertices')
            ms.apply_filter('meshing_close_holes')
            
            ms.apply_filter('meshing_re_orient_faces_coherentely')

            # Save the remeshed file
            output_file = os.path.join(input_dir, 'remeshed_' + filename)
            ms.save_current_mesh(output_file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python simplify_mesh.py [input_dir] [min_edge_length]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    min_edge_length = 0.1 if len(sys.argv) < 3 else sys.argv[2]
    remesh(input_dir, min_edge_length)