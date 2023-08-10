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
- In `config.json` you don't need to specify `input_path`, `tissue`, and `voxel_size`, since they depend on the specific sample. Instead, you have to insert them in `submit_jobs.py` script as follows:
  
- The generate a SLURM command  (e.g., `sbatch -n 1 --cpus-per-task=1 --time=4:00:00 --mem-per-cpu=1024 --wrap="python path/to/submit_jobs.py"`) and run it in the terminal. Before executing be careful that the paths to all the scripts and the config file in `submit_jobs.py` are consistent.

**NOTE: Output format of statistics collection on cluster**
The statistics collection pipeline automatically generates output directories with the follwing format:

![Screenshot 2023-06-13 230257](https://github.com/AntanasMurelis/EpiStats/assets/74301866/36be0a26-b402-4982-b0d5-35c47315d5a4)

- The name of the parent directory is taken from the file name of the input image and the letters `s`, `e`, `d` specify, respectively, the number of smoothing, erosion and dilation iterations performed.
- The subdirectory `cell_meshes` contains the mesh files of all the cells in the sample. 
- The subdirectory `cell_stats` contains the statistics dataframe in the `.csv` file format, and a subdirectory `cached_stats` containg `pickle` file of the statistics collected during the run, and that is used for backup purposes.
- The `cut_cell_idxs.txt` just stores the indexes of the so-called cut cells (i.e. touching the image boundaries).
- The `processed_label.tif` file stores the post-processed labeled 3D image. 

## About plots
A jupyter notebook tutorial showing how the dataframes should be loaded and processed, and how to call plotting functions is reported in the directory `src/tutorial`.

