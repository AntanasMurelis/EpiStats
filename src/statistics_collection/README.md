# Statistics Collection pipeline
The directory `src/statistics_collection` contains different scripts to collect morpohological statistics from different types of cellular tissues.

## Content
We list here the content of this directory, giving a brief explanation of each file:
- `misc.py`: helper functions for the rest of the code.
- `LabelPreprocessing.py`: functions to perform post-processing of labeled 3D images of segmented tissues.
- `GenMeshes.py`: functions to generate meshes from post-processed images of segmented cells.
- `StatsUtils.py`: functions to compute morpholgical statistics either from labeled images or meshes.
- `ExtendedTrimesh.py`: class to compute contact area between cells.
- `StatsCollector.py`: class to collect statistics using the aforementioned functions and to store them in dataframes.
- `config.json`: stores parameters used for running the statistics collection pipeline.
- `run.py`: functions to execute the pipeline.
- `submit_jobs.py`: functions to automatically generate SLURM scripts to run parallel statistics collection from different samples on cluster. 
- `StatsAnalytics.py`: functions to clean/post-process the cell statistics dataframes and prepare it for plots.
- `StatsPlots.py`: functions to get different plots.

## How to run
- To run a single statistics collection you first need to set your parameters in the `config.json` file. Then it is sufficient to run the following: ``python path/to/run.py --config path/to/config/file``.
- To run one or more parallel statistics collection on the cluster you need to set your parameters in the config file (except for `input_path`, `tissue`, and `voxel_size`, which depend on the sample, and hence have to be inserted in the appropriate space in the `submit_jobs.py` script). The generate a SLURM command  (e.g., `sbatch -n 1 --cpus-per-task=1 --time=4:00:00 --mem-per-cpu=1024 --wrap="python path/to/submit_jobs.py"`) and run it in the terminal. Before executing be careful that the paths to all the scripts and the config file in `submit_jobs.py` are consistent.

## Output format
The statistics collection pipeline automatically generates output directories with the follwing format:

![Screenshot 2023-06-13 230257](https://github.com/AntanasMurelis/EpiStats/assets/74301866/36be0a26-b402-4982-b0d5-35c47315d5a4)

- The name of the parent directory is taken from the file name of the input image and the letters `s`, `e`, `d` specify, respectively, the number of smoothing, erosion and dilation iterations performed.
- The subdirectory `cell_meshes` contains the mesh files of all the cells in the sample. 
- The subdirectory `cell_stats` contains the statistics dataframe in the `.csv` file format, and a subdirectory `cached_stats` containg `pickle` file of the statistics collected during the run, and that is used for backup purposes.
- The `cut_cell_idxs.txt` just stores the indexes of the so-called cut cells (i.e. touching the image boundaries).
- The `processed_label.tif` file stores the post-processed labeled 3D image. 

## About plots
A jupyter notebook tutorial showing how the dataframes should be loaded and processed, and how to call plotting functions is reported in the directory `src/tutorial`.

