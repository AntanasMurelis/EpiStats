#!/bin/bash

# Set the path to your Python script containing the collect_cell_morphological_statistics function and argparse
python_script="./collect_statistics.py"

# Set the required arguments for the collect_cell_morphological_statistics function
img_resolution="0.236, 0.236, 0.487"
contact_cutoff="0.5"

# Set the optional arguments for the collect_cell_morphological_statistics function
clear_meshes_folder="--clear_meshes_folder"
meshes_only="--meshes_only"
overwrite="--overwrite"
preprocess="--preprocess"
plot="--plot filtered"
plot_type="--plot_type violin"
calculate_contact_area_fraction="--calculate_contact_area_fraction"

# Set the fixed parameters
smoothing_iterations=6
erosion_iterations=2
dilation_iterations=3

# Set the path to the directory containing the input files
input_directory="/path/to/your/input/files"

# Iterate through the input files in the directory
for labeled_img in $input_directory/*.tif; do
  # Create a job script for the current input file
  file_name=$(basename -- "$labeled_img")
  base_name="${file_name%.*}"
  job_script="job_${base_name}.sh"
  echo "#!/bin/bash" > $job_script
  echo "#SBATCH --job-name=collect_stats_${base_name}" >> $job_script
  echo "#SBATCH --output=logs/collect_stats_${base_name}.out" >> $job_script
  echo "#SBATCH --error=logs/collect_stats_${base_name}.err" >> $job_script
  echo "#SBATCH --time=24:00:00" >> $job_script
  echo "#SBATCH --partition=<your_partition>" >> $job_script
  echo "#SBATCH --nodes=1" >> $job_script
  echo "#SBATCH --ntasks-per-node=8" >> $job_script
  echo "#SBATCH --mem-per-cpu=16G" >> $job_script
  
  # Set the output folder for the current labeled image
  output_folder="--output_folder /cluster/scratch/amurelis/BC_control_2/${base_name}"
  
  echo "python $python_script $labeled_img $img_resolution $contact_cutoff --smoothing_iterations $smoothing_iterations --erosion_iterations $erosion_iterations --dilation_iterations $dilation_iterations $output_folder $overwrite $preprocess" >> $job_script

  # Submit the job script
  sbatch $job_script
done
