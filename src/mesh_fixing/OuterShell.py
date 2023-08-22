import trimesh
import numpy as np
import open3d as o3d
import math
from scipy.spatial import KDTree
import sklearn.gaussian_process as gp
from scipy.interpolate import bisplrep, bisplev, NearestNDInterpolator
from collections import defaultdict
from typing import List, Optional, Type, Literal

### UTILS ###
#----------------------------------------------------------------------------------------------------------------------------
def sample_points_from_vertices(
        vertices: np.ndarray[float],
        num_samples: int
) -> np.ndarray[float]:
    """
    Given a pair or a triplet of vertices, sample `num_sample` new points.
    If a pair of vertices is given, new points are sampled on the line that joins the vertices.
    If a triplet of vertices is given, new points are sampled in the plane enclosed by the triangle
    defined by the 3 vertices.

    NOTE: In case vertices is a triplet, the number of sampled points is greater then `num_samples`.
    (Math law is something like n**2 // 2 + n - 2 - 3)

    Parameters:
    -----------
    vertices: (np.ndarray[float])
        An array of shape (2, 3) or (3, 3), in case of, respectively, a pair or a triplet of vertices.
    
    num_samples: (np.ndarray[float])
        The number of points to sample.
    """
    
    assert vertices.shape[0] in (2, 3), "The shape of `vertices` must be either (2, 3) or (3, 3)"

    sampled_points = []
    
    if vertices.shape[0] == 2:
        v0, v1 = vertices

        direction_vector = v1 - v0
        sampling_steps = np.linspace(0, 1, num_samples + 2)[1:-1]
        new_points = v0[np.newaxis, :] +  sampling_steps[:, np.newaxis] * direction_vector[np.newaxis, :]
        sampled_points.append(new_points.reshape(-1, 3))

    elif vertices.shape[0] == 3:
        v0, v1, v2 = vertices

        grid_points = np.linspace(0, 1, num_samples)
        grid = np.array(np.meshgrid(grid_points, grid_points)).T.reshape(-1, 2)

        for u, v in grid:
            w = 1 - u - v
            if 0 <= u < 1 and 0 <= v < 1 and 0 <= w < 1:
                point = u * v0 + v * v1 + w * v2
                sampled_points.append(point)

        
    sampled_points = np.vstack(sampled_points)
    return sampled_points
#----------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------------
class ExtendedTrimesh(trimesh.Trimesh):
    """
    Child class of `trimesh.Trimesh`.
    It adds some attributes and methods that allow to efficiently compute distances of vertices to other close by meshes.
    """
    def __init__(
            self, 
            neighbors: Optional[List[int]] = None,
            k_percent: Optional[float] = 0.05,
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
        self.neighbors = [int(neighbor) for neighbor in neighbors] if neighbors else None
        # Distance of each vertex to the closest mesh
        self.distances = np.full(len(self.vertices), np.inf)
        # Array storing for each vertex the index of the closest neighbors
        self.closest_neigh_idxs = np.zeros(len(self.vertices)) 
        # Dictionary that associate to each neighbor index a list of vertices idxs that are the k-closest to that neighbor
        self.k = int(k_percent * len(self.vertices))
        self.k_closest_dict = {}
        # Dictionary that stores the KDTree for each subset of k-closest points
        self.k_closest_kdtrees = {}
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
            k: int,
            build_kdtrees: Optional[bool] = True
    ) -> None:
        """
        First, populate `k_closest_dict`, storing, for each neighbor of `self`, the k closest vertex indexes of `self` to it.
        Then, if required by the user, for each subset of k-closest points build a KDTree for efficient nearest neighbors search.

        Parameters:
        -----------
        k: (int)
            Number of closest vertices to consider.

        build_kdtrees: (Optional[bool] = True)
            If `True`, build a KDTree for each subset of k-closest points.
        """

        # Check if the dictionary is already populated
        if len(self.k_closest_dict) > 0:
            print("K-closest dic is not empty!")
            self.k_closest_dict = {}

        assert k < len(self.distances), f"Cannot select a value for k ({k}) larger then total number of points {len(self.distances)}."

        for neigh_idx in self.neighbors:
            original_idxs = np.arange(len(self.distances))

            # Create a mask for the current neighbor, mask distances and original idxs            
            neigh_idx_mask = self.closest_neigh_idxs == neigh_idx
            masked_idxs = original_idxs[neigh_idx_mask]
            masked_distances = self.distances[neigh_idx_mask]

            # Find indices of k closest vertices
            k_closest_idxs = np.argpartition(masked_distances, k)[:k]

            # Map back to original idxs
            mapped_k_closest_idxs = masked_idxs[k_closest_idxs]

            # Store them in `k_closest_dict`
            self.k_closest_dict[neigh_idx] = mapped_k_closest_idxs
            
            # Build KDTree for the subset of points (if required)
            if build_kdtrees:
                k_closest_pts = self.points[mapped_k_closest_idxs]
                self.k_closest_kdtrees[neigh_idx] = KDTree(np.asarray(k_closest_pts))  


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

        # Remove points whose distance is under the threshold
        under_threshold_mask = self.distances > threshold
        self.distances = self.distances[under_threshold_mask]
        self.points = np.asarray(self.vertices)[under_threshold_mask]
        self.closest_neigh_idxs = self.closest_neigh_idxs[under_threshold_mask]

        # Compute mean distance among every point and its nearest neighbor
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        self.mean_point_distance = np.mean(pc.compute_nearest_neighbor_distance())

        # Create `k_closest_dict` with remaining points
        self.get_k_closest_vertices(self.k)

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
            order: Literal['1', '2'] = '1',
            num_samples_per_group: Optional[int] = 3,
    ) -> None:
        """
        Discarding points that are too close to another mesh may generate gaps in the outer shell point cloud.
        This function places points in those gaps by computing the border points for each pair of neighbors
        (i.e., subsets of points that are reciprocally the closest), and interpolating points between groups of 
        such points. 
        In particular, if order=='1', pairs of border points are considered and new points are sampled on the lines
        that join the pair. If order=='2', triplets of points are considered and new points are sampled on the edges
        between points and on the mid-points of lines that connect points sampled on the edges.

        Parameters:
        -----------
        order: (Literal['1', '2'])
            If `order=='1'`, pairs of border points are considered and new points are sampled on the lines
        that join the pair. If `order=='2'`, triplets of points are considered and new points are sampled on the edges
        between points and on the mid-points of lines that connect points sampled on the edges.
        
        num_samples_per_group: (Optional[int] = 3)
            Number of point to sample for each group (i.e., pair or triplet of points). 
        If `order=='2'`, the actual number of sampled points is `num_samples_per_group * 3`.
        """

        assert len(self.points) > 0, "Before interpolation you have to compute the point cloud for the shell."

        # 1. Get all pairs of neighbors
        neighbor_pairs = set()
        for idx, neighbors in enumerate(self._neighbors_lst):
            for neighbor in neighbors:
                pair = tuple(sorted((idx, neighbor)))
                neighbor_pairs.add(pair)

        # 2. Iterate over the neighbor pairs
        all_sampled_points = []
        for idx_1, idx_2 in neighbor_pairs:
            assert len(mesh_1.k_closest_dict) > 0, f"Mesh {idx_1} has empty k_closest_dict attribute."
            assert len(mesh_2.k_closest_dict) > 0, f"Mesh {idx_2} has empty k_closest_dict attribute."

            # 2.a. Get closest points for each cell in the pair
            mesh_1, mesh_2 = self._meshes[idx_1], self._meshes[idx_2]
            closest_point_idxs_1 = mesh_1.k_closest_dict[idx_2]
            closest_point_idxs_2 = mesh_2.k_closest_dict[idx_1]
            closest_points_1 = mesh_1.points[closest_point_idxs_1]
            closest_points_2 = mesh_2.points[closest_point_idxs_2]
            closest_points_kdtree_1 = mesh_1.k_closest_kdtrees[idx_2]
            closest_points_kdtree_2 = mesh_2.k_closest_kdtrees[idx_1]

            # 2.b. For each point in closest_points_1, find the border point in closest_points_2 set using KDTree 
            _, border_idxs_2 = closest_points_kdtree_2.query(closest_points_1, k=1)
            border_idxs_2 = list(set(border_idxs_2))
            border_points_2 = closest_points_2[border_idxs_2]

            # 2.c. Now for each border point of mesh 2, get the associated border point of mesh 1
            k_neighs = 1 if order == "1" else 2
            _, border_idxs_1 = closest_points_kdtree_1.query(border_points_2, k=k_neighs)

            # 2.d. Store the pair of indices in tuples to avoid mixing up the pairs
            border_idxs_pairs = [(b_idx1, b_idx2) for b_idx1, b_idx2 in zip(border_idxs_1, border_idxs_2)]
            # Store in a set to remove duplicates
            border_idxs_pairs = set(border_idxs_pairs)
            # Get ordered border points for mesh 1
            border_points_1 = closest_points_1[[pair[0] for pair in border_idxs_pairs]]

            # 2.e. Compute the direction vectors for each pair of border points
            direction_vectors = border_points_2 - border_points_1

            # Sample points along the direction vectors
            num_samples = 3
            sampling_steps = np.linspace(0, 1, num_samples + 2)[1:-1, np.newaxis]
            sampled_points = border_points_1[:, np.newaxis, :] +  sampling_steps * direction_vectors[:, np.newaxis, :]
            sampled_points = sampled_points.reshape(-1, 3)
            all_sampled_points.append(sampled_points)

        all_sampled_points = np.concatenate(all_sampled_points)
        self.points = np.vstack([self.points, all_sampled_points])




















    # def interpolate_gaps(
    #         self,
    #         method: Literal['gp', 'spline', 'nearest'] = 'nearest',
    #         **kwargs
    # ) -> None:
    #     """
    #     Discarding points that are too close to another mesh may generate gaps in the outer shell point cloud.
    #     This function interpolates points in those gaps using one of the available methods.
    #     The step of the grid is set as the average edge length over all the meshes.

    #     Parameters:
    #     -----------
    #     method: (Literal['gp', 'spline'], default='spline')
    #         The method used to interpolate points in the gaps. 'gp' stands for Gaussian Process, while 'spline' stands 
    #         for B-spline interpolation.
    #     """

    #     assert len(self.points) > 0, "Before interpolation you have to compute the point cloud for the shell."

    #     # Train the chosen model on existing data
    #     x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]

    #     if method == "gp":
    #         ker = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    #         model = gp.GaussianProcessRegressor(kernel=ker)
    #         model.fit(np.column_stack(x, y), z)  
    #     elif method == "spline":
    #         model = bisplrep(x, y, z)
    #     elif method =="nearest":
    #         model = NearestNDInterpolator(np.column_stack([x, y]), z)
    #     else:
    #         NotImplementedError(f"Method {method} is not among the available ones.")

    #     ### Get grid of x, y values at the gaps
    #     # 0. Compute the step of the grid as the average distance between points over all the meshes
    #     grid_step = np.mean([mesh.mean_point_distance for mesh in self._meshes]) / math.sqrt(2)

    #     # 1. Get all pairs of neighbors
    #     neighbor_pairs = set()
    #     for idx, neighbors in enumerate(self._neighbors_lst):
    #         for neighbor in neighbors:
    #             pair = tuple(sorted((idx, neighbor)))
    #             neighbor_pairs.add(pair)
        
    #     # 2. For each pair:
    #     new_shell_points = []
    #     for idx_1, idx_2 in neighbor_pairs:
    #         # 2.a. Get closest points for each cell in the pair
    #         mesh_1, mesh_2 = self._meshes[idx_1], self._meshes[idx_2]
    #         closest_point_idxs_1 = mesh_1.k_closest_dict[idx_2]
    #         closest_point_idxs_2 = mesh_2.k_closest_dict[idx_1]
    #         closest_points = np.concatenate(
    #             [mesh_1.points[closest_point_idxs_1], mesh_2.points[closest_point_idxs_2]]
    #         )

    #         # 2.b. Compute grid taking closest points extema on x and y
    #         max_x = np.max(closest_points[:, 0])
    #         min_x = np.min(closest_points[:, 0])
    #         num_x = int((max_x - min_x) / grid_step)
    #         max_y = np.max(closest_points[:, 1])
    #         min_y = np.min(closest_points[:, 1])
    #         num_y = int((max_y - min_y) / grid_step)
    #         x_grid = np.linspace(min_x, max_x, num_x)
    #         y_grid = np.linspace(min_y, max_y, num_y)
    #         X, Y = np.meshgrid(x_grid, y_grid)
    #         X, Y = X.ravel(), Y.ravel()

    #         # 3. Predict on the newly created grid
    #         if method == "gp":
    #             z_pred = model(np.column_stack(X, Y))
    #         elif method =="spline":
    #             z_pred = bisplev(x_grid, y_grid, model)
    #         elif model =="nearest":
    #             z_pred = model(X, Y)
            
    #         pred_points = np.column_stack([X, Y, z_pred.ravel()])

    #         # 4. Replace existing points in the grid with the newly fitted ones
    #         remove_mask_1 = np.ones(len(mesh_1.points), dtype=bool)
    #         remove_mask_1[closest_point_idxs_1] = False
    #         mesh_1.points = mesh_1.points[remove_mask_1]
    #         remove_mask_2 = np.ones(len(mesh_2.points), dtype=bool)
    #         remove_mask_2[closest_point_idxs_2] = False
    #         mesh_2.points = mesh_2.points[remove_mask_2]
    #         new_shell_points.append(
    #             np.concatenate([mesh_1.points, mesh_2.points, pred_points], axis=0)
    #         )
        
    #     self.points = np.vstack(new_shell_points)


        







    

        
