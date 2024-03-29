import os
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import KDTree
from typing import List, Optional, Type, Literal, Dict


#----------------------------------------------------------------------------------------------------------------------------
class ExtendedTrimesh(trimesh.Trimesh):
    """
    Child class of `trimesh.Trimesh`.
    It adds some attributes and methods that allow to efficiently compute distances of vertices to other close by meshes.
    """
    def __init__(
            self, 
            neighbors: Optional[List[int]] = None,
            k_percent: Optional[float] = None,
            *args, **kwargs):
        """
        Initialize the ExtendedTrimesh object.
        
        Parameters:
        -----------

        neighbors: (Optional[List[int]] = None)
            A list of neighbors of this mesh

        k_percent: (Optional[float] = None)
            The percentual of vertices over the total to take as the closest.
            If `None`, `k_closest_dict` is not computed.
            Reasonable values are in the range [0.05, 0.1].

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
        self.k = (int(k_percent * len(self.vertices))) if k_percent else None
        self.k_closest_dict = {}
        # Dictionary that stores the KDTree for each subset of k-closest points
        self.k_closest_kdtrees = {}
        # Array storing filtered vertices used to generate the outer mesh (initialized as empty)
        self.filtered_vertices = []
        # Array storing the normals from the vertices used to generate the outer mesh (initialized as empty)
        self.filtered_normals = []
        # Mean distance between a point and its nearest neighbor
        self.avg_point_distance = None
        # Minimum distance between a point and its nearest neighbor
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.vertices)
        self.min_point_distance = np.mean(pc.compute_nearest_neighbor_distance())
        

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
                k_closest_pts = self.filtered_vertices[mapped_k_closest_idxs]
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
        self.filtered_vertices = np.asarray(self.vertices)[under_threshold_mask]
        self.filtered_normals = np.asarray(self.vertex_normals)[under_threshold_mask]
        self.closest_neigh_idxs = self.closest_neigh_idxs[under_threshold_mask]

        # Compute mean distance among every point and its one nearest neighbor
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.filtered_vertices)
        self.avg_point_distance = np.mean(pc.compute_nearest_neighbor_distance())

        if self.k:
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
    ):
        """
        Create an OuterShell mesh from a collection of smaller ExtendedTrimesh objects.

        Parameters:
        -----------

        meshes: (List[ExtendedTrimesh])
            A list of ExtendedTrimesh objects that are meant to build the outer shell. 
        neighbors_lst: (List[List[int]])
            A list of neighbors associated to each mesh in meshes
        """
        # The list of meshes contained in the shell (private)
        self._meshes = meshes if meshes else None
        # The list of neighbors associated to each mesh in meshes (private)
        self._neighbors_lst = neighbors_lst if neighbors_lst else None
        # The length in microns of the shortest edge in the mesh (private)
        self.min_edge_length = np.mean([mesh.min_point_distance for mesh in self._meshes])
        # The mean distance between each point and the nearest neighbor in the point cloud
        self._mean_point_distance = None
        # The array of points coordinates to generate the outer shell mesh from
        self.points = []
        # The array of normals to the outer shell points
        self.point_normals = []
        # The final outer shell mesh
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
        shell_normals = []
        for i, (mesh, neighbors) in tqdm(enumerate(zip(self._meshes, self._neighbors_lst)), total=len(self._meshes)):
            # assert isinstance(mesh, ExtendedTrimesh), "Current mesh is not an ExtendedTrimesh object."

            mesh.compute_min_distances([self._meshes[neighbor] for neighbor in neighbors])
            mesh.threshold_vertices(dist_threshold * self.min_edge_length)
            self._meshes[i] = mesh

            shell_points.append(mesh.filtered_vertices)
            shell_normals.append(mesh.filtered_normals)

        self.points = np.vstack(shell_points)
        self.point_normals = np.vstack(shell_normals)
        self.mean_point_distance = np.mean([
            mesh.avg_point_distance for mesh in self._meshes
        ])


    def interpolate_gaps(
            self,
            order: Literal['1', '2'] = '1',
            n_samples: Optional[int] = 3,
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
        
        n_samples: (Optional[int] = 3)
            Number of point to sample for each pair or triplet of points. 
        If `order=='2'`, the actual number of sampled points is greater than n (see `sample_points_from_vertices`).
        """

        assert len(self.points) > 0, "Before interpolation you must compute the point cloud for the shell."
        assert self._meshes[0].k, (
            "Before interpolation you must compute k_closest_dict for each mesh."
            "To do so set a value for `k_percent` when initializing ExtendedTrimesh objects."
        ) 

        # 1. Get all pairs of neighbors
        neighbor_pairs = set()
        for idx, neighbors in enumerate(self._neighbors_lst):
            for neighbor in neighbors:
                pair = tuple(sorted((idx, neighbor)))
                neighbor_pairs.add(pair)

        # 2. Iterate over the neighbor pairs
        all_sampled_points = []
        for idx_1, idx_2 in tqdm(neighbor_pairs):
            mesh_1, mesh_2 = self._meshes[idx_1], self._meshes[idx_2]
            assert len(mesh_1.k_closest_dict) > 0, f"Mesh {idx_1} has empty k_closest_dict attribute."
            assert len(mesh_2.k_closest_dict) > 0, f"Mesh {idx_2} has empty k_closest_dict attribute."

            # 2.a. Get closest points for each cell in the pair
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

            # 2.d. Extract pairs/triplets of closest points
            if order == "1": 
                border_idxs_pairs = [
                    (b_idx1, b_idx2) 
                    for b_idx1, b_idx2 in zip(border_idxs_1, border_idxs_2)
                ]
                # # Store in a set to remove duplicates
                # border_idxs_pairs = set(border_idxs_pairs)
                # Get ordered border points for mesh 1 (shape is (N, 3))
                border_points_1 = closest_points_1[[pair[0] for pair in border_idxs_pairs]]
                # Append border points 2 to create pair of vertices for interpolation
                points_for_interp = np.concatenate(
                    [border_points_1[:, np.newaxis, :], border_points_2[:, np.newaxis, :]], 
                    axis=1
                )
            elif order == "2": 
                #in this case for each border point of 2, there is a pair of points from 1
                border_idxs_triplets = [
                    (b_idxs1[0], b_idxs1[1], b_idx2) 
                    for b_idxs1, b_idx2 in zip(border_idxs_1, border_idxs_2)
                ]
                # # Store in a set to remove duplicates
                # border_idxs_triplets = np.asarray(set(border_idxs_triplets))
                border_idxs_triplets = np.asarray(border_idxs_triplets)
                # Get ordered border points for mesh 1 (shape is (N, 2, 3))
                border_points_1 = closest_points_1[border_idxs_triplets[:, :2]]
                # Append border points 2 to create pair of vertices for interpolation
                points_for_interp = np.concatenate(
                    [border_points_1, border_points_2[:, np.newaxis, :]], 
                    axis=1
                )

            # 2.e. Sample points between points/vertices
            for pts in points_for_interp:
                sampled_points = sample_points_from_vertices(
                    vertices=pts,
                    num_samples=n_samples
                )
                all_sampled_points.append(sampled_points)

        self.points = np.concatenate([self.points] + all_sampled_points, axis=0)
    


    def estimate_point_normals(
        self,
        num_nearest_neighbors: Optional[int] = 10
    ) -> None:
        """
        Estimate normal vectors to the points in the shell point cloud using the function
        `orient_normals_consistent_tangent_plane` from `open3d.geometry.PointCloud`
        (http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html).

        Parameters:
        -----------

        num_nearest_neighbors: (Optional[int] = 10)
            The number of nearest neighbors used in constructing the Riemannian graph used to 
            propagate normal orientation.
        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(fast_normal_computation=False)
        pcd.orient_normals_consistent_tangent_plane(k=num_nearest_neighbors)
        self.point_normals = np.asarray(pcd.normals)


    def displace_point_cloud(
        self,
        num_min_edge_length: Optional[int] = 1
    ) -> None:
        """
        Make sure the the outer mesh contains all the meshes by displacing the shell points 
        along their normals. The translation extent should be really small not to leave big gaps.

        Parameters:
        -----------

        num_min_edge_length: (Optional[int] = 1)
            The translation extent is computed as (num_min_edge_length * self.min_edge_length)
        """

        translation = num_min_edge_length * self.min_edge_length
        displacements = translation * self.point_normals
        self.points += displacements


    def generate_mesh_from_point_cloud(
            self, 
            algorithm: Literal["ball_pivoting", "poisson"] = "ball_pivoting",
            estimate_normals: Optional[bool] = False,
            **kwargs
    ) -> None:
        """
        Generate a surface mesh from the point cloud using one of the available algorithms, namely
        Ball Pivoting, or Screened Poisson Reconstruction.

        Parameters:
        -----------

        algorithm: (Literal["ball_pivoting", "poisson"] = "poisson")
            The algorithm employed for surface reconstruction. If "ball_pivoting", the Ball Pivoting 
            algorithm is used. If "poisson", the Screen Poisson Reconstruction algorithm is used instead.

        estimate_normals: (Optional[bool] = False)
            If `True`, normals are estimated from scratch from the point cloud using the function
            `orient_normals_consistent_tangent_plane` from `open3d.geometry.PointCloud` with k=10
            (http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html).
            If `False`, the normals extracted from the thresholded vertices of the single meshes instead. 

        **kwargs:
            The parameters needed to tune the algorithms. If nothing is provided default values are used.
            
        NOTE1: about parameters for reconstruction algorithms:
        - Ball Pivoting: 
            1. "radius_factor": a multiplying factor for the average distance between nearest points
            (a proxy for the mean edge length) to get the radius of the ball in the algorithm. 
            (Default: 1.5)
        - Screened Poisson Reconstruction: 
            1. "depth": the tree-depth used for the reconstruction. The higher the more detailed the mesh.
            (Default: 8)
            2. "width": the target width of the finest level of the octree. This parameter is ignored if 
            the depth is specified.
            (Default: 0)
            3. "scale": the ratio between the diameter of the cube used for reconstruction and the diameter 
            of the samples' bounding cube. Often the default value works well enough.
            (Default: 1.1)
            4. "linear_fit": if set to true, let the reconstructor use linear interpolation to estimate the
            positions of iso-vertices.
            (Default: False)

        NOTE2: the algorithms for mesh reconstruction are taken from `Open3D` python module.
        """ 

        assert len(self.points) > 0, "Before interpolation you have to compute a point cloud relative to the shell."
        assert (algorithm in ['ball_pivoting', 'poisson']), ( 
            f"The algorithm {algorithm} is not available. Please choose one among ['ball_pivoting', 'poisson']"
        )

        radius_factor = kwargs["radius_factor"] if "radius_factor" in kwargs else 1.5
        depth = kwargs["depth"] if "depth" in kwargs else 8
        width = kwargs["width"] if "width" in kwargs else 0
        scale = kwargs["scale"] if "scale" in kwargs else 1.1
        linear_fit = kwargs["linear_fit"] if "linear_fit" in kwargs else False

        if estimate_normals:
            self.estimate_point_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.normals = o3d.utility.Vector3dVector(self.point_normals)

        if algorithm == "ball_pivoting":
            radius = radius_factor * self.mean_point_distance 
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector([radius, radius * 2])
            )
        elif algorithm == "poisson":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
            )[0]  

        self.mesh = trimesh.Trimesh(
            np.asarray(mesh.vertices), np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals)
        )

    
    def generate_outer_shell(
        self,
        threshold_distance: Optional[float] =  10,
        interp_gaps: Optional[bool] = False,
        interp_params: Optional[Dict[str, any]] = {},
        estimate_vertex_normals: Optional[bool] = False,
        displace_points: Optional[bool] = False,
        displace_params: Optional[Dict[str, int]] = {},
        reconstruction_algorithm: Literal['ball_pivoting', 'poisson'] = 'poisson',
        reconstruction_params: Optional[Dict[str, float]] = {},
    ) -> None:
        """
        Wrapper for all the function needed to generate the outer shell, from point thresholding,
        to mesh generation from point cloud.

        Parameters:
        -----------

        threshold_distance: (Optional[float] =  10)
            The distance threshold under which vertices are discarded. For simplicity it is expressed in units
            of `min_edge_length`.
        interp_gaps: (Optional[bool] = False)
            Discarding points that are too close to another mesh may generate gaps in the outer shell point cloud.
            If `True`, place points in those gaps by computing the border points for each pair of neighbors
            (i.e., subsets of points that are reciprocally the closest), and interpolating points between groups of 
            such points. 
        interp_params: (Optional[Dict[str, any]] = {})
            Dictionary of parameters used for interpolation. Check `interpolate_gaps` method for more information.
        estimate_vertex_normals: (Optional[bool] = False)
            If `True`, normals are estimated from scratch from the point cloud using the function
            `orient_normals_consistent_tangent_plane` from `open3d.geometry.PointCloud` with k=10
            (http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html).
            If `False`, the normals extracted from the thresholded vertices of the single meshes instead. 
        displace_points: (Optional[bool] = False)
            Make sure the the outer mesh contains all the meshes by displacing the shell points 
            along their normals.
        displace_params: (Optional[Dict[str, int]] = {})
            Dictionary of parameters used for point displacement. Check `displace_point_cloud` method for more information.
        reconstruction_algorithm: (Literal['ball_pivoting', 'poisson'] = 'poisson')
            The algorithm employed for surface reconstruction. If "ball_pivoting", the Ball Pivoting 
            algorithm is used. If "poisson", the Screen Poisson Reconstruction algorithm is used instead.
            For a better result Poisson algorithm should be preferred.
        reconstruction_params: (Optional[Dict[str, float]] = {})
            The parameters needed to tune the algorithms. If nothing is provided default values are used.
        
        NOTE: about interpolation and reconstruction algorithms:
        Interpolation places points in the gaps and hence it is helpful in case the chosen algorithm for surface 
        recontruction is ball pivoting. However, the algorithm for interpolating the gaps is far from being perfect
        and hence should be used carefully.
        The suggestion is to always try to use first the Screened Poisson Reconstruction algorithm.


        NOTE: about parameters for reconstruction algorithms:
        - Ball Pivoting: 
            1. "radius_factor": a multiplying factor for the average distance between nearest points
            (a proxy for the mean edge length) to get the radius of the ball in the algorithm. 
            (Default: 1.5)
        - Screened Poisson Reconstruction: 
            1. "depth": the tree-depth used for the reconstruction. The higher the more detailed the mesh.
            (Default: 8)
            2. "width": the target width of the finest level of the octree. This parameter is ignored if 
            the depth is specified.
            (Default: 0)
            3. "scale": the ratio between the diameter of the cube used for reconstruction and the diameter 
            of the samples' bounding cube. Often the default value works well enough.
            (Default: 1.1)
            4. "linear_fit": if set to true, let the reconstructor use linear interpolation to estimate the
            positions of iso-vertices.
            (Default: False)
        """

        print("    Computing shell point cloud...")
        self.get_shell_point_cloud(dist_threshold=threshold_distance)
        print("    Done!\n")

        if interp_gaps:
            print("    Interpolating gaps between cells...")
            self.interpolate_gaps(**interp_params)
            print("    Done!\n")

        if displace_points:
            print("    Displacing points along normal directions...")
            self.displace_point_cloud(**displace_params)
            print("    Done!\n")

        print("    Reconstructing mesh from point cloud...")
        self.generate_mesh_from_point_cloud(
            algorithm=reconstruction_algorithm,
            estimate_normals=estimate_vertex_normals,
            **reconstruction_params
        )
        print("    Done!\n")

    def mesh_to_file(
            self,
            path_to_file: str,
            overwrite: Optional[bool] = False
    ) -> None:
        """
        Save mesh to a file.

        Parameters:
        -----------

        path_to_file: (str)
            The path to the file where to save the mesh. 
            The only supported format is '.stl'.

        overwrite: (Optional[bool] = False)
            If `True` and a file already exists at the specified path, that file is overwritten.
            Notice that generating the outer shell could be a time consuming process when the number of 
            meshes involved is large. Therefore, overwriting might be something undesirable.
        """

        assert not os.path.exists(path_to_file) and not overwrite, (
            f"Aborting export since a file was found at {path_to_file} and overwrite is set to {overwrite}"
        )

        file_extension = os.path.basename(path_to_file).split(".")[-1]
        assert file_extension == "stl", f"Cannot export '.{file_extension}' file. The only supported format is '.stl'."

        path_to_dir = os.path.dirname(path_to_file)

        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        self.mesh.export(path_to_file)





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
    (Math law is something like n**2 // 2 + n - 5)

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



        







    

        
