
## Content of this directory:
We list here the content of this directory, giving a brief explanation of each file:
- `utils.py`: helper functions used extensively in the rest of the code.
- `LabelPreprocessing.py`: functions to perform post-processing of labeled 3D images of segmented tissues.
- `GenMeshes.py`: functions to generate meshes from post-processed images of segmented cells.
- `StatsCompute.py`: functions to compute morpholgical statistics either from labeled images or meshes.
- `ExtendedTrimesh.py`: class to compute contact area between cells.
- `StatsCollector.py`: class to collect statistics using the aforementioned functions and to store them in dataframes.
- `StatsAnalytics.py`: functions to clean/post-process the cell statistics dataframes and prepare it for plots.
- `StatsPlots.py`: functions to get different plots.
