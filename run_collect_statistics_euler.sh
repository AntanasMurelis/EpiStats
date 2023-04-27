#!/bin/bash

# Set the path to your Python script containing the collect_cell_morphological_statistics function and argparse
python_script="./collect_statistics.py"

# Set the required arguments for the collect_cell_morphological_statistics function
labeled_img="/cluster/home/amurelis/BladderCancer/Validated_labels_final.tif"
img_resolution="0.21 0.21 0.39"
contact_cutoff="0.8"

# Set the optional arguments for the collect_cell_morphological_statistics function
clear_meshes_folder="--clear_meshes_folder"
output_folder="--output_folder /cluster/scratch/amurelis/BC_control_2"
meshes_only="--meshes_only"
overwrite="--overwrite"
preprocess="--preprocess"
plot="--plot filtered"
plot_type="--plot_type violin"
calculate_contact_area_fraction="--calculate_contact_area_fraction"

# Iterate through the desired parameter combinations
for smoothing_iterations in {5..8}; do
  for erosion_iterations in {1..4}; do
    for dilation_iterations in {2..5}; do
      if [ $dilation_iterations -gt $erosion_iterations ]; then
      # Create a job script for the current parameter combination
      job_script="job_s_${smoothing_iterations}_e_${erosion_iterations}_d_${dilation_iterations}.sh"
      echo "#!/bin/bash" > $job_script
      echo "#SBATCH --job-name=collect_stats_s_${smoothing_iterations}_e_${erosion_iterations}_d_${dilation_iterations}" >> $job_script
      echo "#SBATCH --output=logs/collect_stats_s_${smoothing_iterations}_e_${erosion_iterations}_d_${dilation_iterations}.out" >> $job_script
      echo "#SBATCH --error=logs/collect_stats_s_${smoothing_iterations}_e_${erosion_iterations}_d_${dilation_iterations}.err" >> $job_script
      echo "#SBATCH --time=24:00:00" >> $job_script
      echo "#SBATCH --partition=<your_partition>" >> $job_script
      echo "#SBATCH --nodes=1" >> $job_script
      echo "#SBATCH --ntasks-per-node=8" >> $job_script
      echo "#SBATCH --mem-per-cpu=16G" >> $job_script
      echo "python $python_script $labeled_img $img_resolution $contact_cutoff --smoothing_iterations $smoothing_iterations --erosion_iterations $erosion_iterations --dilation_iterations $dilation_iterations $output_folder $overwrite $preprocess" >> $job_script

      # Submit the job script
      sbatch $job_script
      fi
    done
  done
done
