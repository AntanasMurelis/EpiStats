import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from collections import defaultdict
from typing import List, Optional, Type, Literal


#----------------------------------------------------------------------------------------------------------------------------
class ExtendedTrimesh(trimesh.Trimesh):
    """
    Child class of `trimesh.Trimesh`.
    It adds some attributes and methods that allow to efficiently compute distances of vertices to other close by meshes.
    """
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
        # Array storing points to be included in the outer mesh (initialized as empty)
        self.points = []
        # Mean distance among each point and its nearest neighbor
        self.mean_point_distance = None
    

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

        k = min(k, len(self.distances))

        # Find indices of k closest vertices
        k_closest_idxs = np.argpartition(self.distances, k)[:k]

        # Find indices of neighbors associated to k closest vertices
        k_closest_neigh_idxs = self.closest_neigh_idxs[k_closest_idxs]

        # Store them in `k_closest_dict`
        for i in range(len(k_closest_idxs)):
            self.k_closest_dict[k_closest_neigh_idxs[i]].add(k_closest_idxs[i])


    def threshold_vertices(
            self,
            threshold: float,
    ) -> None:
        """
        Discard vertices whose distance is under a threshold, save the remaining in points, and update data structures accordingly.

        Parameters:
        -----------
        mesh: (ExtendedTrimesh)
            An ExtendedTrimesh object to apply thresholding on.
        
        threshold: (float)
            The threshold under which vertices should be discarded.
        """

        under_threshold_mask = self.distances < threshold
        self.points = np.asarray(self.vertices)[under_threshold_mask]
        self.closest_neigh_idxs = self.closest_neigh_idxs[under_threshold_mask]

        # Compute mean distance among every point and its nearest neighbor
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        self.mean_point_distance = np.mean(pc.compute_nearest_neighbor_distance())

        # If `k_closest_dict` has already been created, create a new one from scratch
        if self.k_closest_dict:
            k = sum([len(val) for val in self.k_closest_dict.values()])
            self.get_k_closest_vertices(k)

#-----------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------
class OuterShell:
    """
    Class that allows to create an OuterShell mesh from a collection of smaller ExtendedTrimesh objects.
    """
    def __init__(
            self,
            meshes: List[ExtendedTrimesh],
            neighbors_lst: List[List[int]],
            min_edge_length: float
    ):
        """
        Parameters:
        -----------
        meshes: (List[ExtendedTrimesh])
            A list of ExtendedTrimesh objects that are meant to build the outer shell. 
        neighbors_lst: (List[List[int]])
            A list of neighbors associated to each mesh in meshes
        min_edge_length: (float)
            The length in microns of the shortest edge in the target mesh
        """
        # The list of meshes contained in the shell (private)
        self._meshes = meshes if meshes else None
        # The list of neighbors associated to each mesh in meshes (private)
        self._neighbors_lst = neighbors_lst if neighbors_lst else None
        # The length in microns of the shortest edge in the mesh (private)
        self._min_edge_length = min_edge_length if min_edge_length else None
        # Attribute meant to store the point cloud of point that should generate the outer shell mesh
        self.point_cloud = []
        # Attribute meant to store the final outer shell mesh
        self.mesh = None

    
    def get_shell_point_cloud(
            self,
            dist_threshold: float
    ) -> None:
        """
        Get the point cloud that will be used to define the outer shell mesh.
        It is obtained in the following way:
        - For each mesh in meshes:
            - Compute the min distances between its vertices and any of the neighboring meshes.
            - Discard all vertices whose distance is smaller than the threshold
            - Add the remaining vertices to the outer shell point cloud

        Parameters:
        -----------
        dist_threshold: (float)
            The distance threshold under which vertices are discarded. For simplicity it is expressed in units
            of `min_edge_length`.
        """

        self.point_cloud = []
        for mesh, neigbors in zip(self._meshes, self._neighbors_lst):
            assert isinstance(mesh, ExtendedTrimesh), "Current mesh is not an ExtendedTrimesh object."

            mesh.compute_min_distances(self.meshes[neigbors])
            mesh.threshold_vertices(dist_threshold)

            self.point_cloud.append(mesh.points)

        self.point_cloud = np.vstack(self.point_cloud)

    
    def interpolate_gaps(
            self,
            method: Literal['gp', 'spline'] = 'spline'
    ) -> None:
        """
        Discarding points that are too close to another mesh may generate gaps in the outer shell point cloud.
        This function interpolates points in those gaps using one of the available methods.
        The pitch of the grid is set as the average edge length over all the meshes.
        """



    

        
