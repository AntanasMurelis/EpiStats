# Morphological Cell Statistics

This repository contains a Python script and associated tools for calculating morphological statistics of cells in 3D images. The main script processes a labeled 3D image and computes various cell statistics, such as cell areas, volumes, elongation, and contact area fraction. The results are saved as CSV files for further analysis. In addition, cell meshes meshes are saved in their respective directory. The script also generates plots for the cell statistics.

## Requirements

* Python 3.7 or later
* NumPy
* pandas
* scikit-image
* trimesh
* seaborn
* matplotlib
* Trimesh
* scipy
* tqdm
* Hydra
* OmegaConf

## Installation
Install venv if not already done
```pip3 install virtualenv```

Create a virtual environment
```python3 -m venv EpiStatsVenv```

Activate the virtual env
```source ./EpiStatsVenv/bin/activate```

Install the required packages
```pip3 install -r requirements.txt```

# Usage

The main script can be run with the following command:

```bash
python path/to/main.py
```

The script uses Hydra for managing configurations. The configurations are stored in a yaml file located in the `conf` directory. The yaml file is named `config.yaml` by default. Here is an example of a configuration file:

```yaml
labeled_img: path/to/labeled_img.tiff
img_resolution: [0.21, 0.21, 0.39]
contact_cutoff: 0.5
clear_meshes_folder: False
smoothing_iterations: 5
erosion_iterations: 1
dilation_iterations: 2
output_folder: path/to/output/directory
meshes_only: False
overwrite: False
preprocess: True
calculate_contact_area_fraction: True
max_workers: None
plot: "both"
plot_type: violin
volume_lower_threshold: None
volume_upper_threshold: None
```

Please modify the paths and values according to your local setup.

## Output

The output folder will contain the following files and directories:
*	**all_cell_statistics.csv**: Contains the statistics for all cells in the labeled 3D image.
*	**filtered_cell_statistics.csv**: Contains the statistics for cells that are not touching the background.
*	**cell_meshes**: Directory containing the mesh files for all cells in the labeled 3D image.
*   **plots**: The generated plots for the cell statistics.

The CSV files will include the following columns:

* **cell_id**: The cell identifier, corresponding to the label in the labeled 3D image.
* **cell_area**: The surface area of the cell in square microns.
* **cell_volume**: The volume of the cell in cubic microns.
* **cell_isoperimetric_ratio**: The isoperimetric ratio of the cell, which is a measure of its compactness.
* **cell_neighbors**: A space-separated list of neighboring cell IDs.
* **cell_nb_of_neighbors**: The number of neighboring cells.
* **cell_principal_axis**: The principal axis of the cell, formatted as a space-separated string of floating-point values.
* **cell_elongation**: The elongation of the cell, which is a measure of its shape anisotropy.
* **cell_contact_area_fraction**: The fraction of the cell's surface area that is in contact with neighboring cells.

