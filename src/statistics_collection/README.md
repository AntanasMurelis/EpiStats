# Statistics Collection pipeline
The directory `src/statistics_collection` contains different scripts to collect morpohological statistics from different types of cellular tissues.

## CONTENT:
We list here the content of this directory, giving a brief explanation of each file:
- `config.json`: parameters used for running the statistics collection pipeline.
- `dataset_preparation.py`: post-process statistics dataset to save them and for plotting.
- `run_collection.py`: run a single instance of statistics collection pipeline.
- `submit_jobs.py`: automatically generate SLURM scripts to run parallel statistics collections on cluster. 

## HOW TO:
#### 1. Statistics collection for a single sample
- Set your parameters in the `config.json` file.
- Run the following command:

```
python path/to/run_collection.py --config path/to/config/file
```

#### 2. Statistics collection for multiple samples in parallel (on cluster)
- Set your parameters in the config file.
- In `config.json` you don't need to specify `input_path`, `tissue`, `tissue_type`, `filtering`, `slicing_dim` and `voxel_size`, since they depend on the specific sample. Instead, you have to insert them in `submit_jobs.py` script as follows:
  <br>
  
  ![set_user_inputs](https://github.com/AntanasMurelis/EpiStats/blob/main/images/submit_jobs_user_inputs.png)

  NOTE: for the sake of convenience, the pre-defined values of `tissue_type`, `filtering`, and `slicing_dim` have been hard coded for the following tissue names:

  `intestine_villus`, `lung_bronchiole`, `bladder`, `esophagus`, `lung`, `embryo`

  Therefore, for these tissues, the user can set the values of `tissue_type`, `filtering`, and `slicing_dim` to `None`.

- Check that the paths files/scripts in `submit_jobs.py` are consistent:
  <br>
  
  ![check_path_1](https://github.com/AntanasMurelis/EpiStats/blob/main/images/info_run_collection_5.png)
  ![check_path_2](https://github.com/AntanasMurelis/EpiStats/blob/main/images/info_run_collection_4.png)

- Generate a SLURM command to launch `submit_jobs.py` on cluster and run it in the terminal:
```
sbatch -n 1 --cpus-per-task=1 --time=4:00:00 --mem-per-cpu=1024 --wrap="python path/to/submit_jobs.py"
```

#### 3. Post-process outputs of statistics collection from multiple tissues
- Post-processing consists of:
  1. Merging datasets associated to different tissues/samples,
  2. Extracting a dataset of only selected numerical features for plotting,
  3. Computing PCs (and related information) for plotting,
  4. Extracting statistics from 2D slices for plotting Lewis law and Aboav-Weaire law.
  5. Saving all the results in the output directory.

- After having specified the requested input parameters (check script), run:
  ```
  python path/to/dataset_preparation.py
  ```

#### 4. Plotting
Check the notebook `src/notebooks/plot_tutorial.ipynb`, a tutorial showing how the dataframes should be loaded and processed, and how to call plotting functions is reported in the directory.


## ADDITIONAL NOTES: 
#### 1. Output format of statistics collection on cluster
The statistics collection pipeline automatically generates output directories with the following format:

![statistics_collections_output_folder](https://github.com/AntanasMurelis/EpiStats/blob/main/images/statistics_collections_output_folder_structure.png)

- The name of the parent directory is taken from the file name of the input image and the letters `s`, `e`, `d` specify, respectively, the number of smoothing, erosion and dilation iterations performed.
- The subdirectory `cell_meshes` contains the mesh files of all the cells in the sample. 
- The subdirectory `cell_stats` contains the statistics dataframe in the `.csv` file format, and a subdirectory `cached_stats` containg `pickle` file of the statistics collected during the run, and that is used for backup purposes.
- The `cut_cell_idxs.txt` just stores the indexes of the so-called cut cells (i.e. touching the image boundaries).
- The `processed_label.tif` file stores the post-processed labeled 3D image. 

#### 2. Set proper time duration for the job
If you want to compute contact area between cells and 2D statistics (area, neighbors) along the apical-basal axis of each cell, be aware that the computation may require some time. Indeed, both algorithms have `O(n^2)` time complexity.
I will provide here some examples as a reference for possible time duration of statistics collection runs (50 slices along apical-basal axes):

- 50 cells --> ~15 mins
- 150 cells --> ~30 mins
- 300 cells --> ~2 hrs
- 500 cells --> ~3/4 hrs

If you are dealing with a very large sample you can:
1. Set a longer reserved time for your jobs in `submit_jobs.py`:
   
![check_path_2](https://github.com/AntanasMurelis/EpiStats/blob/main/images/info_run_collection_6.png)

2. Reduce the number of slices along apical-basal axes (e.g. from 50 to 20)


