import trimesh
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import KDTree
from collections import defaultdict
from typing import List, Optional, Type


#----------------------------------------------------------------------------------------------------------------------------
class ExtendedTrimesh(trimesh.Trimesh):
    def __init__(
            self, 
            neighbors: Optional[List[int]] = None,
            *args, **kwargs):
        """
        Initialize the ExtendedTrimesh object.
        
        Parameters:
        -----------
        neighbors: (Optional[List[int]] = None)
            A list of neighbors of this mesh

        *args, **kwargs: 
            Thes refer to the standard parameters for trimesh.Trimesh class (e.g., file name, vertices and faces arrays, etc.)
            Refer to trimesh.Trimesh documentation for more details.
        """
        super().__init__(*args, **kwargs)
        # Create a KDTree from the vertices of the mesh for quick spatial lookups.
        self.kdtree = KDTree(self.vertices)  
        # Neighbors list
        self.neighbors = neighbors if neighbors else None
        # Distance of each vertex to the closest mesh
        self.distances = np.full(len(self.vertices), np.inf)
        # Array storing for each vertex the index of the closest neighbors
        self.closest_neigh_idxs = np.zeros(len(self.vertices)) 
        # K-closest dictionary -> associate to each neighbor index a set of vertices idxs that are within the k-closest 
        # to any other mesh
        self.k_closest_dict = defaultdict(set)
    
    def _get_dist(
            self,
            other: Type['ExtendedTrimesh'],
    ) -> np.ndarray[float]:
        """
        Compute the minimum distance between vertices of self and other, an object of `ExtendedTrimesh` class.

        Parameters:
        -----------
        other: (ExtendedTrimesh)
            A mesh object of `ExtendedTrimesh` class
        
        Returns:
        --------
        other_distances: (np.ndarray[float])
            An array of distances of vertices of `self` from `other` mesh.
        """

        other_distances = np.zeros(len(self.vertices))
        for i, vertex in enumerate(self.vertices):
            # Query the KDtree of `other`
            other_distances[i] = other.kdtree.query(vertex)

        return other_distances
    

    def compute_min_distances(
            self,
            neigh_meshes: List[Type['ExtendedTrimesh']]
    ) -> None:
        """
        Given the list of neighboring meshes, compute for each vertex of `self` the minimum distance between 
        the vertex itself and any other neighboring mesh.
        For each vertex, store the index of the closest neighboring mesh.

        Parameters:
        -----------
        neigh_meshes: (List[Type['ExtendedTrimesh']])
            A list of ExtendedTrimesh objects representing the neighboring meshes.
        """

        for neigh_mesh, neigh_idx in zip(neigh_meshes, self.neighbors):
            curr_distances = self._get_dist(neigh_mesh)
            min_mask = curr_distances <= self.distances
            self.closest_neigh_idxs[min_mask] = neigh_idx
            self.distances[min_mask] = curr_distances[min_mask]

    
    def get_k_closest_vertices(
            self,
            k: int
    ) -> None:
        """
        Populate the `k_closest_dict` dictionary, storing a total of k closest vertex indexes associated
        to the closest neighbor index.

        Parameters:
        -----------
        k: (int)
            Number of closest vertices to consider.
        """

        # Find indices of k closest vertices
        k_closest_idxs = np.argpartition(self.distances, k)[:k]

        # Find indices of neighbors associated to k closest vertices
        k_closest_neigh_idxs = self.closest_neigh_idxs[k_closest_idxs]

        # Store them in `k_closest_dict`
        for i in range(len(k_closest_idxs)):
            self.k_closest_dict[k_closest_neigh_idxs[i]] = k_closest_idxs[i]

#-----------------------------------------------------------------------------------------------------------------