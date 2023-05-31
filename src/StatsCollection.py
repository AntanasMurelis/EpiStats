import numpy as np
import pandas as pd
import trimesh as tm
import os
import pickle
import concurrent.futures
from scipy import ndimage
from tqdm import tqdm
from typing import Optional, List, Tuple, Iterable, Dict, Union, Callable
from ExtendedTrimesh import ExtendedTrimesh
from LabelPreprocessing import get_labels_touching_edges, get_labels_touching_background
from time import sleep

#------------------------------------------------------------------------------------------------------------
def compute_cell_surface_areas(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        exclude_labels: Iterable[int],
    ) -> Dict[int, float]:
    """
    Use the meshes of the cells to compute their areas (in micron squared).
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The tirangular meshes of the cell in the standard trimesh format
        associated to the corresponding cell label.

    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns:
    --------
    area_dict: (Dict[int, float])
        The surface area of each cell with the corresponding cell label
    """

    assert cell_mesh_dict.__len__() > 0

    area_dict = {}

    for id, mesh in tqdm(cell_mesh_dict.items(),
                         desc="Computing cell surface area",
                         total=len(cell_mesh_dict)):
        if id not in exclude_labels:
            area_dict[id] = mesh.area
        else:
            area_dict[id] = None
    
    return area_dict
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
def compute_cell_volumes(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        exclude_labels: Iterable[int],
    ) -> Dict[int, float]:
    """
    Use the meshes of the cells to compute their volumes (in micron cubes).
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The tirangular meshes of the cell in the standard trimesh format
        associated to the corresponding cell label

    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns:
    --------
    volume_dict: (Dict[int, float])
        List of the volume of each cell
    """

    assert cell_mesh_dict.__len__() > 0

    volume_dict = {}

    for id, mesh in tqdm(cell_mesh_dict.items(),
                         desc="Computing cell volume",
                         total=len(cell_mesh_dict)):
        if id not in exclude_labels:
            volume_dict[id] = mesh.volume
        else:
            volume_dict[id] = None

    return volume_dict
#------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def compute_cell_principal_axis_and_elongation(
    cell_mesh_dict: Dict[int, tm.base.Trimesh],
    exclude_labels: Iterable[int],
) -> Tuple[Dict[int, tm.base.Trimesh], Dict[int, float]]:
    """
    Compute the principal axis and elongation for each cell in the mesh list.
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The tirangular meshes of the cell in the standard trimesh format
        associated to the corresponding cell label

    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns:
    --------
    cell_cell_elongation_axes_dict (Dict[int, Tuple[float, List[float]]]):
        Dict whose keys are cell ids and values are tuples of elongation and
        principal axes for the cell id.

    """

    # Initialize lists to store the principal axis and elongation for each cell
    cell_elongation_axes_dict = {}

    # Loop over each cell ID
    for cell_id, cell_mesh in tqdm(cell_mesh_dict.items(), 
                                   desc='Computing cell principal axis and elongation',
                                   total=len(cell_mesh_dict)):
        if cell_id not in exclude_labels:
            # Use the inertia tensor of the cell shape to compute its principal axis
            eigen_values, eigen_vectors = tm.inertia.principal_axis(cell_mesh.moment_inertia)

            # Get the index of the smallest eigen value
            smallest_eigen_value_idx = np.argmin(np.abs(eigen_values))
            greatest_eigen_value_idx = np.argmax(np.abs(eigen_values))

            # Get the corresponding eigen vector 
            principal_axes = eigen_vectors[smallest_eigen_value_idx]
            # Compute elongation
            elongation = np.sqrt(eigen_values[greatest_eigen_value_idx] / eigen_values[smallest_eigen_value_idx])
        else:
            principal_axes = None
            elongation = None

        # Store the results in the corresponding lists
        cell_elongation_axes_dict[cell_id] = elongation, principal_axes

    # Return the lists of principal axes and elongations
    return cell_elongation_axes_dict
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def compute_cell_neighbors(
        labeled_img: np.ndarray[int], 
        exclude_labels: Iterable[int],
    ) -> Dict[int, List[int]]:
    """
    Get all the neighbors of a given cell. Two cells are considered neighborhs if 
    a subset of their surfaces are directly touching.
    
    Parameters:
    -----------
    labeled_img: (np.ndarray, 3D, dtype=int)
        A 3D labeled image where the background has a label of 0 and cells are labeled with 
        consecutive integers starting from 1.
    
    exclude_labels: (Iterable[int])
        A collection of cell indexes to exclude from neighbors computation. 
        However, these cells are considered in other cells neighbors counts.

    Returns
    -------
    neighbors_dict: (Dict[int, List[int]])
        A dict whose key is the cell label and the value is a list of the ids of the neighbors
    """

    label_ids = np.unique(labeled_img)

    neighbors_dict = {}

    for label in tqdm(label_ids[1:], desc="Computing cell neighbors"):
        if label not in exclude_labels:
            #Get the voxels of the cell
            binary_img = labeled_img == label

            #Expand the volume of the cell by 2 voxels in each direction
            expanded_cell_voxels = ndimage.binary_dilation(binary_img, iterations=2)

            #Find the voxels that are directly in contact with the surface of the cell
            cell_surface_voxels = expanded_cell_voxels ^ binary_img

            #Get the labels of the neighbors
            neighbors_lst = np.unique(labeled_img[cell_surface_voxels])

            #Remove the label of the cell itself, and the label of the background from the neighbors list
            neighbors_lst = neighbors_lst[(neighbors_lst != label) & (neighbors_lst != 0)]
        else:
            neighbors_lst = []

        # Append to the dictionary
        neighbors_dict[label] = neighbors_lst

    return neighbors_dict
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
def _compute_contact_area(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        cell_id: int, 
        cell_neighbors_lst: List[int], 
        contact_cutoff: float
    ) -> Tuple[float, List[float]]:
    """
    Compute the fraction of the cell surface area which is in contact with other cells.
    
    Parameters:
    -----------
    cell_mesh_dict: (Dict[int, tm.base.Trimesh])
        The triangular meshes of the cell in the standard trimesh format
        with the corresponding cell label as key.

    cell_id: (int)
        The id of the cell for which we want to calculate the contact area fraction.

    cell_neighbors_lst: (List[int])
        List of the ids of the neighbors of the current cell.

    contact_cutoff: (float)
        The cutoff distance in microns for two cells to be considered in contact.

    Returns:
    --------
    contact_fraction: (float)
        The percentage of the total cell surface are in contact with neighboring cells.
    
    contact_area_distribution: (List[float])
        A list of the contact areas between the cell and each one of the neigbors.
    """
    sleep(2)

    if len(cell_neighbors_lst) > 0:
        print(f'Computing contact area for cell {cell_id}... ')
        # start = time()
        cell_mesh = cell_mesh_dict[cell_id]
        cell_mesh = ExtendedTrimesh(cell_mesh.vertices, cell_mesh.faces)
        
        contact_area_distribution = np.zeros(len(cell_neighbors_lst))
        # Loop over the neighbors of the cell
        for i, neighbor_id in enumerate(cell_neighbors_lst):
            # Get potential contact faces using the get_potential_contact_faces method from ExtendedTrimesh class
            neighbour_mesh = ExtendedTrimesh(cell_mesh_dict[neighbor_id].vertices, cell_mesh_dict[neighbor_id].faces)
            contact_area = cell_mesh.calculate_contact_area(neighbour_mesh, contact_cutoff)
            contact_area_distribution[i] = contact_area

        # Calculate the fraction of contact area
        contact_fraction = np.sum(contact_area_distribution) / cell_mesh.area
        # print(f'elspsed time: {time() - start}')
    else:
        print(f'Skipping cell {cell_id}...')
        contact_fraction, contact_area_distribution = None, []

    return contact_fraction, contact_area_distribution
#------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------
def compute_cell_contact_area(
        cell_mesh_dict: Dict[int, tm.base.Trimesh],
        cell_neighbors_dict: Dict[int, List[int]], 
        max_workers: int,
        contact_cutoff: Optional[float] = 0.1, 
    ) -> Dict[any, Union[float, List[float]]]:
    """
    Compute the contact area fraction for each cell in the mesh list.
    
    Parameters:
    -----------
        cell_mesh_dict (Dict[int, tm.base.Trimesh]):
            Dictionary of cell meshes.

        cell_neighbors_lst (List[int]):
            List of neighbors for each cell.

        max_workers (int):
            Maximum number of workers for the ThreadPoolExecutor.
        
        contact_cutoff (Optional[float], default=0.1):
            The cutoff distance in microns for two cells to be considered in contact.

    Returns:
    --------
        cell_contact_area_dict (Dict[int, Union[float, List[float]]]):
            Dictionary where key is cell_id and value is a tuple containing contact area fraction, 
            and contact area distribution.
    """
    print('Computing cell contact area ...')    
    
    cell_contact_area_dict = {}

    # Create a pool of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_cell_id = {
            executor.submit(_compute_contact_area, cell_mesh_dict, cell_id, cell_neighbors, contact_cutoff): cell_id 
            for cell_id, cell_neighbors in cell_neighbors_dict.items()
        }

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(future_to_cell_id):
            cell_id = future_to_cell_id[future]
            try:
                cell_contact_area_dict[cell_id] = future.result()
            except Exception as exc:
                print(f'Cell {cell_id} generated an exception: {exc}')

    return cell_contact_area_dict
#------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------
class StatsCollector:
    '''
    Class to collect the statistics for a given preprocessed and meshed tissue image.

    Parameters:
    -----------
        meshes: (Dict[int, tm.base.Trimesh])
            The triangular meshes of the cell in the standard trimesh format
            associated to the corresponding cell label.

        labels: (np.ndarray[int], 3D)
            A 3D labeled image where the background has a label of 0 and cells are labeled with 
            consecutive integers starting from 1.
        
        features: (List[str])
            A list of features to compute. Can be chosen among:
            ['area', 'volume', 'elongation_and_axes', 'neighbors', 'contact_area'].
        
        output_directory: (str)
            Path to the directory in which statistics will be saved.
            In particular, the dataframe is saved in the subdir `cell_stats`,
            while cached values of the statistics are saved in `cell_stats/cached_stats`.
        
        path_to_img: (str)
            Path to the file containing the preprocessed image.
        
        tissue: (str)
            Name of the tissue under analysis. Can be chosen among:
            ['bladder', 'intestine_villus', 'lung_bronchiole', 'esophagus']
        
        num_workers: (int)
            Number of workers used for computation of conatct area between cells.


    Example:
    --------
    ```
    # Initialize
    stats_collector = StatsCollector(
        meshes=meshes,
        labels=preprocessed_labeled_img,
        features=['area', 'volume', 'elongation_and_axes', 'neighbors', 'contact_area'],
        output_directory='/output',
        path_to_img='/output/processed_labels.tif',
        tissue='lung_bronchiole',
        num_workers=4
    )
    
    # Collect statistics in a pd.DataFrame
    stats_collector.collect_statistics(load_from_cache=False)
    ```
    '''
    def __init__(
            self,
            meshes: Dict[int, tm.base.Trimesh],
            labels: np.ndarray[int],
            # original_ids: List[int],
            features: List[str],
            output_directory: str,
            path_to_img: str,
            tissue: str,
            num_workers: int
        ) -> None:

        #internal attributes
        self._features_to_functions = StatsCollector._feat_to_func_dict()
        self._tissues_to_types = StatsCollector._tissue_to_type_dict()
        
        #public attributes
        self.meshes = meshes
        self.labels = labels
        self.ids = list(self.meshes.keys())
        # self.original_ids = original_ids
        self.features = features 
        self.functions = [
            self._features_to_functions[feature] 
            for feature in self.features
        ]
        self.tissue = tissue
        self.tissue_type = self._tissues_to_types[tissue]
        self.output_dir = output_directory
        self.df_output_dir = os.path.join(self.output_dir, 'cell_stats')
        self.path_to_img = path_to_img
        file_name = os.path.basename(self.path_to_img)
        self.file_ext = os.path.splitext(file_name)[1]
        self.file_name = file_name.replace(self.file_ext, '')
        self.num_workers = num_workers
        self.cache_dir = os.path.join(self.df_output_dir, 'cached_stats')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        #apply filtering to get labels to be excluded from computation
        self.excluded_idxs = self.filter_cells()

        #initialize and save the dataframe to store statistics
        self.df = self._init_dataframe()
        #save the newly created data structure
        self._save_dataframe()

    @staticmethod
    def _feat_to_func_dict() -> Dict[str, Callable]:
        features = ['area', 'volume', 'elongation_and_axes',
                    'neighbors', 'contact_area']
        functions = [
            compute_cell_surface_areas,
            compute_cell_volumes,
            compute_cell_principal_axis_and_elongation,
            compute_cell_neighbors,
            compute_cell_contact_area
        ]

        return dict(zip(features, functions))

    @staticmethod
    def _tissue_to_type_dict() -> Dict[str, str]:
        tissues = ['bladder', 'intestine_villus', 'lung_bronchiole', 'esophagus']
        tissue_types = ['stratified_transitional', 'simple_columnar', 'simple_cuboidal', 'stratified_squamous']

        return dict(zip(tissues, tissue_types))
    
    
    def filter_cells(self) -> List[int]:
        if 'simple' in self.tissue_type:
            idxs_to_filter = get_labels_touching_edges(
                self.labels, self.output_dir
            )
        elif 'stratified' in self.tissue_type:
            idxs_edges = get_labels_touching_edges(
                self.labels, self.output_dir
            )
            idxs_bg = get_labels_touching_background(
                self.labels, self.output_dir
            )
            idxs_to_filter = np.union1d(idxs_edges, idxs_bg)
        
        return idxs_to_filter
        

    def _save_dataframe(
            self,
            overwrite: bool = True
    ) -> None:
        
        if not os.path.exists(self.df_output_dir):
            os.makedirs(self.df_output_dir)

        path_to_file = os.path.join(self.df_output_dir, self.file_name)
        if (not os.path.isfile(path_to_file)) or overwrite:
            self.df.to_csv(path_to_file)

    
    def _init_dataframe(self) -> pd.DataFrame:
        #initialize the data structure
        df = pd.DataFrame(
            data={
                'cell_ID': self.ids,
                'tissue': self.tissue,
                'file_name': self.path_to_img
                # 'original_cell_ID': self.original_ids
            }
        )
        df['mesh_dir'] = [
            os.path.join(self.output_dir, 'cell_meshes', f'cell_{id}.stl')
            for id in self.ids
        ]
        df['exclude_cell'] = [id in self.excluded_idxs for id in self.ids]

        return df

    @staticmethod
    def _unpack_feature_dict(
            feature_dict: Dict[int, any]
        ) -> pd.Series:
        '''
        Unpack the dictionary associated to each feature in a pd.Series.

        Parameters:
        -----------
            feature_dict: (Dict[int, any])
                A dict whose keys are cell ids and values are the associated statistics value.

        Returns:
        --------
            feature_unpacked: (pd.Series[any])
                A pd.Series of the statistics values.
        '''
        feature_unpacked = pd.Series(list(feature_dict.values()))
        return feature_unpacked
    
    
    def _add_to_dataframe(
        self,
        feature_dict: Dict[int, any],
        feature_name: str,
    ) -> None:

        #unpack the dictionary
        feature_data = StatsCollector._unpack_feature_dict(feature_dict)

        #add column to df
        self.df[feature_name] = feature_data 

    def _to_cache(
            self,
            feature_dict: Dict[int, any], 
            feature_name: str
    ) -> None:

        save_name = f'cell_{feature_name}.pickle'
        with open(os.path.join(self.cache_dir, save_name), 'wb') as file:
            pickle.dump(feature_dict, file)

    
    def _from_cache(
            self,
            feature_name: str
    ) -> Dict[int, any]:

        assert os.path.exists(self.cache_dir), 'Cannot load from cache as it is empty.'

        save_name = f'cached_stats/cell_{feature_name}.pickle'
        with open(os.path.join(self.df_output_dir, save_name), 'rb') as file:
            feature_dict = pickle.load(file)
    
        return feature_dict
    

    def _get_args(
            self,
            feature_name: str,
    ) -> List[any]:
        if feature_name == 'neighbors':
            args = (
                self.labels,
                self.excluded_idxs
            )
        elif feature_name == 'contact_area':
            try:
                neighbors_dict = self._from_cache('neighbors')
            except Exception as e:
                raise OSError('Neighbors dictionary not found in the cache.') from e
            args = (
                self.meshes,
                neighbors_dict,
                self.num_workers
            )
        else:
            args = (
                self.meshes,
                self.excluded_idxs
            )

        return args


    def _process_df(
        self,
    ) -> None:
        #count neighbors
        self.df['num_neighbors'] = self.df['neighbors'].apply(lambda x: len(x))

        #split elongation and principal axes
        self.df['elongation'] = self.df['elongation_and_axes'].apply(lambda x: x[0])
        self.df['principal_axes'] = self.df['elongation_and_axes'].apply(lambda x: x[1])
        self.df.drop(columns=['elongation_and_axes'], inplace=True)

        #split and extract statistics from contact area
        self.df['contact_area_fraction'] = self.df['contact_area'].apply(lambda x: x[0])
        self.df['contact_area_distribution'] = self.df['contact_area'].apply(lambda x: x[1])
        self.df['mean_contact_area'] = self.df['contact_area_distribution'].apply(lambda x: np.mean(x))
        self.df['total_contact_area'] = self.df['contact_area_distribution'].apply(lambda x: np.sum(x))
        self.df.drop(columns='contact_area', inplace=True)

        #compute isoperimetric ratio
        self.df["isoperimetric_ratio"] = self.df.apply(lambda x: (x['area']**3)/(x['volume']**2), axis=1)


    def collect_statistics(
            self,
            load_from_cache: Optional[bool] = False
    ) -> None:
        
        for func, feat in zip(self.functions, self.features):
            if load_from_cache:
                print(f'Loading cached cell {feat} ...')
                feat_dict = self._from_cache(feat)
            else:
                args = self._get_args(feat)
                feat_dict = func(*args)

                #cache the dict
                self._to_cache(feat_dict, feat)
            
            #add the new stat to the dataframe
            self._add_to_dataframe(feat_dict, feat)

            #save the dataframe
            self._save_dataframe(overwrite=True)

        #postprocess dataframe
        self._process_df()
        self._save_dataframe(overwrite=True)
#------------------------------------------------------------------------------------------------------------