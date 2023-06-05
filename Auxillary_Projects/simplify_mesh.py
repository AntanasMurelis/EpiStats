# File: simplify_mesh.py

import os
import sys
import pymesh
import numpy as np

def simplify_mesh(input_dir, detail="normal"):
    
    # Delete old 'simplified_' files
    for filename in os.listdir(input_dir):
        if filename.startswith('simplified_'):
            os.remove(os.path.join(input_dir, filename))
    
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.stl') or filename.endswith('.ply'):
            input_file = os.path.join(input_dir, filename)
            mesh = pymesh.load_mesh(input_file)

            # Get the bounding box diagonal length
            bbox_min, bbox_max = mesh.bbox
            diag_len = np.linalg.norm(bbox_max - bbox_min)

            # Set target edge length based on detail level
            try:
                target_len = diag_len * float(detail)
            except ValueError:
                if detail == "normal":
                    target_len = diag_len * 5e-3
                elif detail == "high":
                    target_len = diag_len * 2.5e-3
                elif detail == "low":
                    target_len = diag_len * 1e-2
                else:
                    print(f"Unknown detail level: {detail}")
                    sys.exit(1)
            print("Target resolution: {} mm".format(target_len))

            count = 0
            mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
            mesh, __ = pymesh.split_long_edges(mesh, target_len)
            num_vertices = mesh.num_vertices
            while True:
                mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
                mesh, __ = pymesh.collapse_short_edges(mesh, target_len, preserve_feature=True)
                mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
                if mesh.num_vertices == num_vertices:
                    break

                num_vertices = mesh.num_vertices
                print("#v: {}".format(num_vertices))
                count += 1
                if count > 10: break

            mesh = pymesh.resolve_self_intersection(mesh)
            mesh, __ = pymesh.remove_duplicated_faces(mesh)
            mesh = pymesh.compute_outer_hull(mesh)
            mesh, __ = pymesh.remove_duplicated_faces(mesh)
            mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
            mesh, __ = pymesh.remove_isolated_vertices(mesh)

            # Save the simplified mesh
            output_file = os.path.join(input_dir, 'simplified_' + filename)
            pymesh.save_mesh(output_file, mesh)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python simplify_mesh.py [input_dir] [detail]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    detail = "normal" if len(sys.argv) < 3 else sys.argv[2]
    simplify_mesh(input_dir, detail)
