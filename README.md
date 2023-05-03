# Morphological Cell Statistics

This repository contains a Python script and associated tools for calculating morphological statistics of cells in 3D images. The main script processes a labeled 3D image and computes various cell statistics, such as cell areas, volumes, elongation, and contact area fraction. The results are saved as CSV files for further analysis. In addition, cell meshes and filtered cell meshes are saved in their respective directories. The script also generates plots for the cell statistics.

## Requirements

* Python 3.7 or later
* NumPy
* pandas
* scikit-image
* trimesh
* seaborn
* matplotlib

## Installation
Install venv if not already done
```pip3 install virtualenv```

Create a virtual environment
```python3 -m venv collect_cell_stats```

Activate the virtual env
```source ./collect_cell_stats/bin/activate```

Install the required packages
```pip3 install -r requirements.txt```

# Usage

1. Modify the **path/to/your_python_script.py** in the bash script (**run_jobs.sh**) to point to the Python script containing the collect_cell_morphological_statistics function and argparse.

2. Modify the **labeled_img** variable in the bash script (**run_jobs.sh**) to point to your labeled 3D image file.

3. Modify the **img_resolution** and **contact_cutoff** variables in the bash script (**run_jobs.sh**) to set the appropriate values for your image data.

4. Make sure the bash script (run_jobs.sh) is executable:

```
chmod +x run_jobs.sh

```

5. Run the bash script (run_jobs.sh) on your computing cluster to submit the jobs with the desired parameter combinations:
```
./run_jobs.sh
```


The cell statistics will be saved as CSV files in the output folder with the specified naming convention (e.g., *output_s_5_e_1_d_3* for 5 smoothing iterations, 1 erosion iteration, and 3 dilation iterations). Plots for the cell statistics will also be generated.

## Output

The output folder will contain the following files and directories:
*	**all_cell_statistics.csv**: Contains the statistics for all cells in the labeled 3D image.
*	**filtered_cell_statistics.csv**: Contains the statistics for cells that are not touching the background.
*	**cell_meshes**: Directory containing the mesh files for all cells in the labeled 3D image.
*	**filtered_cell_meshes**: Directory containing the mesh files for cells that are not touching the background.
*   **plots**: Directory containing the generated plots for the cell statistics.

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

## Visualization

The `visualise_cells.py` script can be used to visualize the cell meshes and their corresponding label images. It can display all cells, a single cell, or the filtered cell meshes depending on the provided arguments.

Usage:

```bash
python visualise_cells.py output_directory [--cell_id CELL_ID] [--filtered]
```

* **output_directory**: Path to the output directory where the processed labels and cell meshes are stored.

* **cell_id CELL_ID**: (Optional) View a specific cell mesh only. Provide the cell ID as an integer.

* **filtered**: (Optional) View filtered cell meshes only.

## Plotting

The `generate_plots` function in the main script can generate plots for the cell statistics. It supports 'violin' plots, 'box' plots, and 'histogram'. The plots can be saved in a specified output folder or displayed without saving.

![Statistics of Cubes example](/images/filtered_cell_plots.png)

## Note
- When tested against the cubic test geometry, the measurement of the area introduces a ~5% error from the  real area. 
- In the same conditions the measurement of the volume introduces a ~1.5% error from the real volume.
- The contact fraction introduces a ~2% error from the real value.


