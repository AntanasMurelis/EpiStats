#!/bin/bash

# Set the path to your Python script containing the collect_cell_morphological_statistics function and argparse
python_script="./collect_statistics.py"

# Set the required arguments for the collect_cell_morphological_statistics function
labeled_img="/Users/antanas/BC_Project/Control_Segmentation/Validated_labels_extended.tif"
img_resolution="0.21 0.21 0.39"
contact_cutoff="0.8"

# Set the optional arguments for the collect_cell_morphological_statistics function
clear_meshes_folder="--clear_meshes_folder"
smoothing_iterations="--smoothing_iterations 5"
erosion_iterations="--erosion_iterations 1"
dilation_iterations="--dilation_iterations 2"
output_folder="--output_folder BC_Project"
meshes_only="--meshes_only"
overwrite="--overwrite"
preprocess="--preprocess"
calculate_contact_area_fraction="--calculate_contact_area_fraction True"
max_workers="--max_workers 4"
plot="--plot both"
plot_type="--plot_type violin"

# Run the Python script with the arguments
python $python_script $labeled_img $img_resolution $contact_cutoff $clear_meshes_folder $smoothing_iterations $erosion_iterations $dilation_iterations $output_folder $meshes_only $overwrite $preprocess $calculate_contact_area_fraction $max_workers $plot $plot_type
