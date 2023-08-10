from pykdtree.kdtree import KDTree
import numpy as np
import trimesh
from typing import List

class ExtendedTrimesh(trimesh.Trimesh):
    def __init__(self, *args, **kwargs):
        """
        Initialize the ExtendedTrimesh object.
        
        Parameters:
        *args, **kwargs: These are standard parameters for trimesh.Trimesh class. They can be a variety of inputs
        such as the file name of a 3D model, vertices and faces arrays, etc. Refer to trimesh.Trimesh documentation
        for more details.
        """
        super().__init__(*args, **kwargs)
        self.my_kdtree = KDTree(self.vertices)  # Create a KDTree from the vertices of the mesh for quick spatial lookups.


    def get_potential_contact_faces(
            self, 
            other_mesh, 
            distance: float
        ) -> List:
        """
        Compute the indices of potential contact faces of other_mesh that lie within a given distance from this mesh.

        Parameters:
        -----------
            other_mesh (ExtendedTrimesh): 
                Another ExtendedTrimesh object which is to be checked for contact with the current mesh.
        
            distance (float): 
                The distance within which a face of the other_mesh is considered in contact with this mesh.

        Returns:
        --------
            potential_contact_faces (list): 
                List of indices of the potential contact faces in other_mesh.
        """

        # Initialize list to hold potential contact face indices
        potential_contact_faces = []
        # Loop through all the faces of other_mesh
        for face_index in range(len(other_mesh.faces)): 
            # Get the vertices of the current face
            face = other_mesh.vertices[other_mesh.faces[face_index]]
            # Calculate the centroid of the face  
            centroid = np.mean(face, axis=0).reshape(1, -1)
            # Query the KDTree to find the distance and index of the nearest point in this mesh to the face centroid
            dist, idx = self.my_kdtree.query(centroid)  
            # If the distance is less than the given distance, then this face is in potential contact
            if dist < distance:  
                # Add the index of this face to the list of potential contact faces
                potential_contact_faces.append(face_index)  
        
        return potential_contact_faces  


    def calculate_contact_area(self, other_mesh, distance):
        """
        Calculate the contact area between this mesh and other_mesh. It is calculated as the sum of the areas of the faces
        of other_mesh that are in potential contact with this mesh.

        Parameters:
        other_mesh (ExtendedTrimesh): Another ExtendedTrimesh object which is to be checked for contact with this mesh.
        distance (float): The distance within which a face of the other_mesh is considered in contact with this mesh.

        Returns:
        contact_area (float): The total contact area between this mesh and other_mesh.
        """

        # Get indices of potential contact faces
        contact_faces_indices = self.get_potential_contact_faces(other_mesh, distance)
        # Calculate the contact area by summing the areas of the potential contact faces
        contact_area = np.sum(other_mesh.area_faces[contact_faces_indices])
        
        return contact_area
    
#----------------------------------------------------------------------------------------------------------------------
