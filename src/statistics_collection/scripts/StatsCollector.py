import os
import sys
import numpy as np
import pandas as pd
import trimesh as tm
import pickle
from typing import Optional, List, Tuple, Iterable, Dict, Union, Literal

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from LabelPreprocessing import get_labels_touching_edges, get_labels_touching_background
from StatsCompute import *

#--------------------------------------------------------------------------------------------------
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
        ['area', 'volume', 'principal_axis_and_elongation', 'neighbors', 'contact_area'].
    
    output_directory: (str)
        Path to the directory in which statistics will be saved.
        In particular, the dataframe is saved in the subdir `cell_stats`,
        while cached values of the statistics are saved in `cell_stats/cached_stats`.
    
    path_to_img: (str)
        Path to the file containing the preprocessed image.
    
    tissue: (str)
        Name of the tissue under analysis. Available predefined tissues are:
        ['bladder', 'intestine_villus', 'lung_bronchiole', 'esophagus', 'embryo', 'lung']

    tissue_type: (str)
        The type of the tissue under analysis. In case `tissue` is not among the predefined
        ones, `tissue_type` must be specified by the user. Otherwise, the tissue types associated 
        to available tissues are:
        ['stratified_transitional', 'simple_columnar', 'simple_cuboidal', 'stratified_squamous', 
        'Undefined', 'pseudostratified']

    filtering: (List[Literal["cut_cells", "touching_bg"]])
        A list of metods to filter out cells which are not wanted for statistics collection.
        In particular, the options "cut_cells" and "touching_bg" are respectively associated
        to the functions `get_labels_touching_edges` and `get_labels_touching_background` from
        `LabelPreprocessing.py`.

    voxel_size: (Iterable[float])
        The voxel size of the input labeled image.

    slicing_dim: (Literal[0, 1, 2])
        The axis along which 2D slices are taken in the case of 2D statistics collection from 
        canonical axes.

    num_2D_slices: (int)
        The number of 2D slices to extract statistics from.
    
    size_2D_slices: (int)
        The number of pixels of the 2D slices used to extract statistics
    
    num_workers: (int)
        Number of workers used for computation of conatct area between cells.


    Example:
    --------
    ```
    # Initialize
    stats_collector = StatsCollector(
        meshes=meshes,
        labels=preprocessed_labeled_img,
        features=[
            'area', 'volume', 'principal_axis_and_elongation', 
            'neighbors', 'contact_area', '2D_statistics',
            '2D_statistics_apical_basal'
        ],
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
            features: List[str],
            output_directory: str,
            path_to_img: str,
            tissue: str,
            tissue_type: str,
            filtering: List[Literal["cut_cells", "no_filtering", "touching_bg"]],
            voxel_size: Iterable[float],
            slicing_dim: Literal[0, 1, 2],
            num_2D_slices: int,
            size_2D_slices: int,
            num_workers: int
        ) -> None:

        # Utility attributes
        _avail_features = [
            "area", "volume", "principal_axis_and_elongation", 
            "neighbors", "contact_area", "2D_statistics", 
            "2D_statistics_apical_basal"
        ]
        _avail_functions = [
            compute_cell_surface_areas,
            compute_cell_volumes,
            compute_cell_principal_axis_and_elongation,
            compute_cell_neighbors,
            compute_cell_contact_area,
            compute_2D_statistics,
            compute_2D_statistics_along_axes
        ]
        _features_to_functions = dict(zip(_avail_features, _avail_functions))
        _avail_tissues = [
            'bladder', 'intestine_villus', 'lung_bronchiole', 'esophagus', 'embryo', 'lung'
        ]
        _avail_tissue_types = [
            'stratified_transitional', 'simple_columnar', 'simple_cuboidal', 
            'stratified_squamous', 'Undefined', 'pseudostratified'
        ]
        _tissues_to_types = dict(zip(_avail_tissues, _avail_tissue_types))
        _avail_slicing_dims = [0, 2, 1, 0, 2, 1]
        _tissues_to_slicing_dims = dict(zip(_avail_tissues, _avail_slicing_dims))
        _tissues_to_filtering = dict(zip(
            _avail_tissues, 
            [["cut_cells"], ["cut_cells"], ["cut_cells"], ["cut_cells"], ["no_filtering"], ["cut_cells", "touching_bg"]]
        ))
        
        # Attributes related to input samples
        self.features = features
        self.functions = [_features_to_functions[feat] for feat in features]
        self.tissue = tissue
        if tissue_type:
            self.tissue_type = tissue_type
        else:
            self.tissue_type = _tissues_to_types[tissue]
        self.meshes = meshes
        self.labels = labels
        self.ids = list(self.meshes.keys())
        self.voxel_size = voxel_size
        self.num_2D_slices = num_2D_slices
        self.size_2D_slices = size_2D_slices
        if isinstance(slicing_dim, int):
            self.slicing_dim = slicing_dim
        else:
            self.slicing_dim = _tissues_to_slicing_dims[tissue]
        if filtering:
            self.filtering = filtering
        else:
            self.filtering = _tissues_to_filtering[self.tissue]
        
        # Attributes related to file paths and saving directories
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
    

    def filter_cells(self) -> List[int]:
        
        idxs_to_filter = set()
        if "touching_bg" in self.filtering:
            crop_lims = [150, 360] if self.tissue == "lung" else None
            new_idxs_to_filter, _ = get_labels_touching_background(
                labeled_img=self.labels, 
                slicing_dim=self.slicing_dim,
                crop_limits=crop_lims,
                threshold=0.1,
                output_directory=self.output_dir,
            )
            idxs_to_filter.update(new_idxs_to_filter)
        elif "cut_cells" in self.filtering:
            new_idxs_to_filter = get_labels_touching_edges(
                labeled_img=self.labels, 
                output_directory=self.output_dir
            )
            idxs_to_filter.update(new_idxs_to_filter)
        
        return list(idxs_to_filter)
        

    def _save_dataframe(
            self,
            overwrite: bool = True
    ) -> None:
        
        if not os.path.exists(self.df_output_dir):
            os.makedirs(self.df_output_dir)

        path_to_file = os.path.join(self.df_output_dir, f'stats_dataset_{self.tissue}.csv')
        if (not os.path.isfile(path_to_file)) or overwrite:
            self.df.to_csv(path_to_file)

    
    def _init_dataframe(self) -> pd.DataFrame:
        #initialize the data structure
        df = pd.DataFrame(
            data={
                'cell_ID': self.ids,
                'tissue': self.tissue,
                'tissue_type': self.tissue_type,
                'file_name': self.path_to_img,
                # 'original_cell_ID': self.original_ids,
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
        
        #order the dict according to its keys
        sorted_feature_dict = {k: feature_dict[k] for k in sorted(feature_dict)}
        #transform dictionary into a pd.Series
        feature_unpacked = pd.Series(list(sorted_feature_dict.values()))
        return feature_unpacked
    
    
    def _add_to_dataframe(
        self,
        feature_dict: Union[Dict[int, any], Tuple[Dict[int, any], Dict[int, any]]],
        feature_name: str,
    ) -> None:

        if feature_name == '2D_statistics':
            for i in range(3):
                feat_names = ['neighbors_2D', 'area_2D', 'slices']
                #unpack the dictionary
                feature_data = StatsCollector._unpack_feature_dict(feature_dict[i])
                #add column to df
                self.df[feat_names[i]] = feature_data 
        elif feature_name == '2D_statistics_apical_basal':
            for i in range(4):
                feat_names = [
                    'neighbors_2D_principal', 
                    'area_2D_principal',
                    'neighbors_of_neighbors_2D_principal', 
                    'slices_principal'
                ]
                #unpack the dictionary
                feature_data = StatsCollector._unpack_feature_dict(feature_dict[i])
                #add column to df
                self.df[feat_names[i]] = feature_data 
        else:
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
        elif feature_name == 'volume':
            args = (
                self.labels,
                self.excluded_idxs,
                self.voxel_size
            )
        elif feature_name == 'contact_area':
            try:
                neighbors_dict = self._from_cache('neighbors')
            except Exception as e:
                raise OSError('Neighbors dictionary not found in the cache.') from e
            args = (
                self.meshes,
                neighbors_dict,
                self.num_workers,
                max(self.voxel_size)*2
            )
        elif feature_name == '2D_statistics':
            args = (
                self.labels,
                self.slicing_dim,
                self.excluded_idxs,
                np.concatenate((self.voxel_size[:self.slicing_dim], 
                                self.voxel_size[(self.slicing_dim+1):]))
            )
        elif feature_name == '2D_statistics_apical_basal':
            args = (
                self.labels,
                self.meshes,
                self.excluded_idxs,
                self.voxel_size,
                self.num_2D_slices,
                self.size_2D_slices
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
        
        #compute isoperimetric ratio
        if ('area' in self.features) and ('volume' in self.features):
            self.df["isoperimetric_ratio"] = self.df.apply(lambda x: (x['area']**3)/(x['volume']**2), axis=1)

        #count neighbors
        if 'neighbors' in self.features:
            self.df['num_neighbors'] = self.df['neighbors'].apply(lambda x: len(x))

        #split elongation and principal axes
        if 'principal_axis_and_elongation' in self.features:
            self.df['elongation'] = self.df['principal_axis_and_elongation'].apply(lambda x: x[0])
            self.df['principal_axes'] = self.df['principal_axis_and_elongation'].apply(lambda x: x[1])
            self.df.drop(columns=['principal_axis_and_elongation'], inplace=True)

        #split and extract statistics from contact area
        if 'contact_area' in self.features:
            self.df['contact_area_fraction'] = self.df['contact_area'].apply(lambda x: x[0])
            self.df['contact_area_distribution'] = self.df['contact_area'].apply(lambda x: x[1])
            self.df['mean_contact_area'] = self.df['contact_area_distribution'].apply(lambda x: np.mean(x))
            self.df['total_contact_area'] = self.df['contact_area_distribution'].apply(lambda x: np.sum(x))
            self.df.drop(columns='contact_area', inplace=True)

        #compute number of neighbors in 2D slices
        if '2D_statistics' in self.features:
            self.df["num_neighbors_2D"] = self.df['neighbors_2D'].apply(lambda x: [len(l) for l in x])
            neighbors_changes = []
            for num_neighbors_lst in self.df["num_neighbors_2D"]:
                num_neighbors_lst = np.asarray(num_neighbors_lst)
                neighbors_changes.append(
                    np.sum(
                        (num_neighbors_lst[1:] - num_neighbors_lst[:-1]).astype(bool) 
                ))
            self.df["num_neighbors_changes_2D"] = neighbors_changes

        if '2D_statistics_apical_basal' in self.features:
            self.df["num_neighbors_2D_principal"] = self.df['neighbors_2D_principal'].apply(lambda x: [len(l) for l in x])
            neighbors_changes = []
            for num_neighbors_lst in self.df["num_neighbors_2D_principal"]:
                num_neighbors_lst = np.asarray(num_neighbors_lst)
                neighbors_changes.append(
                    np.sum(
                        (num_neighbors_lst[1:] - num_neighbors_lst[:-1]).astype(bool) 
                ))
            self.df["num_neighbors_changes_2D_principal"] = neighbors_changes


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

#--------------------------------------------------------------------------------------------------
