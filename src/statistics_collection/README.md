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



