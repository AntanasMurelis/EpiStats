import trimesh
import numpy as np
import open3d as o3d
import math
from scipy.spatial import KDTree
import sklearn.gaussian_process as gp
from scipy.interpolate import bisplrep, bisplev
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
            k_percent: Optional[float] = 0.1,
            *args, **kwargs):
        """
        Initialize the ExtendedTrimesh object.
        
        Parameters:
        -----------
        neighbors: (Optional[List[int]] = None)
            A list of neighbors of this mesh

        k_percent: (Optional[float] = 0.1)
            The percentual of vertices over the total to take as the closest.

        *args, **kwargs: 
            Thes refer to the standard parameters for trimesh.Trimesh class (e.g., file name, vertices and faces arrays, etc.)
            Refer to trimesh.Trimesh documentation for more details.
        """
        super().__init__(*args, **kwargs)

        # # Create a KDTree from the vertices of the mesh for quick spatial lookups.
        # self.kdtree = KDTree(np.asarray(self.vertices))  
        # Neighbors list
        self.neighbors = neighbors if neighbors else None
        # Distance of each vertex to the closest mesh
        self.distances = np.full(len(self.vertices), np.inf)
        # Array storing for each vertex the index of the closest neighbors
        self.closest_neigh_idxs = np.zeros(len(self.vertices)) 
        # K-closest dictionary -> associate to each neighbor index a set of vertices idxs that are within the k-closest 
        # to any other mesh
        self.k = k_percent * len(self.vertices)
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
            other_distances[i] = other.kdtree.query(vertex)[0]

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

        under_threshold_mask = self.distances > threshold
        self.points = np.asarray(self.vertices)[under_threshold_mask]
        self.closest_neigh_idxs = self.closest_neigh_idxs[under_threshold_mask]

        # Compute mean distance among every point and its nearest neighbor
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        self.mean_point_distance = np.mean(pc.compute_nearest_neighbor_distance())

        # Create `k_closest_dict` with remaining points
        self.k_closest_dict = defaultdict(set)
        self.get_k_closest_vertices(self.k)
        print(self.k_closest_dict)

#-----------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------
class OuterShell:
    """
    Create an OuterShell mesh from a collection of smaller ExtendedTrimesh objects.
    """
    def __init__(
            self,
            meshes: List[ExtendedTrimesh],
            neighbors_lst: List[List[int]],
            min_edge_length: float
    ):
        """
        Create an OuterShell mesh from a collection of smaller ExtendedTrimesh objects.

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
        # Attribute meant to store the array of points that should generate the outer shell mesh
        self.points = []
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

        shell_points = []
        for i, (mesh, neighbors) in enumerate(zip(self._meshes, self._neighbors_lst)):
            assert isinstance(mesh, ExtendedTrimesh), "Current mesh is not an ExtendedTrimesh object."

            mesh.compute_min_distances([self._meshes[neighbor] for neighbor in neighbors])
            mesh.threshold_vertices(dist_threshold * self._min_edge_length)
            self._meshes[i] = mesh

            shell_points.append(mesh.points)

        self.points = np.vstack(shell_points)

    
    def interpolate_gaps(
            self,
            method: Literal['gp', 'spline'] = 'spline',
            **kwargs
    ) -> None:
        """
        Discarding points that are too close to another mesh may generate gaps in the outer shell point cloud.
        This function interpolates points in those gaps using one of the available methods.
        The step of the grid is set as the average edge length over all the meshes.

        Parameters:
        -----------
        method: (Literal['gp', 'spline'], default='spline')
            The method used to interpolate points in the gaps. 'gp' stands for Gaussian Process, while 'spline' stands 
            for B-spline interpolation.
        """

        assert len(self.points) > 0, "Before interpolation you have to compute the point cloud for the shell."

        # Train the chosen model on existing data
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]

        if method == "gp":
            ker = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
            model = gp.GaussianProcessRegressor(kernel=ker)
            model.fit(np.column_stack(x, y), z)  
        elif method == "spline":
            model = bisplrep(x, y, z)
        else:
            NotImplementedError(f"Method {method} is not among the available ones.")

        ### Get grid of x, y values at the gaps
        # 0. Compute the step of the grid as the average distance between points over all the meshes
        grid_step = np.mean([mesh.mean_point_distance for mesh in self._meshes]) / math.sqrt(2)

        # 1. Get all pairs of neighbors
        neighbor_pairs = set()
        for idx, neighbors in enumerate(self._neighbors_lst):
            for neighbor in neighbors:
                pair = tuple(sorted((idx, neighbor)))
                neighbor_pairs.add(pair)
        
        # 2. For each pair:
        new_shell_points = []
        for idx_1, idx_2 in neighbor_pairs:
            # 2.a. Get closest points for each cell in the pair
            mesh_1, mesh_2 = self._meshes[idx_1], self._meshes[idx_2]
            closest_point_idxs_1 = mesh_1.k_closest_dict[idx_2]
            closest_point_idxs_2 = mesh_2.k_closest_dict[idx_1]
            closest_points = np.concatenate(
                [mesh_1.points[closest_point_idxs_1], mesh_2.points[closest_point_idxs_2]]
            )

            # 2.b. Compute grid taking closest points extema on x and y
            max_x = np.max(closest_points[:, 0])
            min_x = np.min(closest_points[:, 0])
            max_y = np.max(closest_points[:, 1])
            min_y = np.min(closest_points[:, 1])
            x_grid = np.linspace(min_x, max_x, grid_step)
            y_grid = np.linspace(min_y, max_y, grid_step)
            X, Y = np.meshgrid(x_grid, y_grid)

            # 3. Predict on the newly created grid
            if method == "gp":
                z_pred = model(np.column_stack(X, Y))
            elif method =="spline":
                z_pred = bisplev(X, Y, model)
            pred_points = np.column_stack(
                [X.ravel(), Y.ravel(), z_pred.ravel()]
            )

            # 4. Replace existing points in the grid with the newly fitted ones
            mesh_1.points = np.remove(mesh_1.points, closest_point_idxs_1)
            mesh_2.points = np.remove(mesh_2.points, closest_point_idxs_2)
            new_shell_points.append(
                np.concatenate(mesh_1.points, mesh_2.points, pred_points)
            )
        
        self.points = self.vstack(new_shell_points)


        







    

        
