#!/bin/bash

# Set the path to your Python script containing the collect_cell_morphological_statistics function and argparse
python_script="./create_meshes_for_simulation.py"

# Set the required arguments for the collect_cell_morphological_statistics function
img_path="data/curated_labels_clean/lung_samples/lung_new_sample_b_curated_segmentation_central_crop_relabeled_filled.tif"
output_dir="meshes_for_simulation"
voxel_size="0.1625 0.1625 0.25"

# Set the optional arguments for the collect_cell_morphological_statistics function
dilation_iters="--dilation_iters 4"
erosion_iters="--erosion_iters 2"
combined_mesh="--combined_mesh"
labels_to_convert="--labels_to_convert"

# Set the path to the directory containing the input files
input_directory="/cluster/work/cobi/federico/EpiStats/data/curated_labels_clean/bladder_samples/"

# Iterate through the input files in the directory
for smoothing_iterations in {5,10,20,50,100}; do
  # Create a job script for the current number of simulations
  smoothing_iters="--smoothing_iters ${smoothing_iterations}"
  echo "#!/bin/bash" > $job_script
  echo "#SBATCH --job-name=convert_meshes_sim_iters_${smooothing_iterations}" >> $job_script
  echo "#SBATCH --output=logs/convert_meshes_sim_iters_${smooothing_iterations}.out" >> $job_script
  echo "#SBATCH --error=logs/convert_meshes_sim_iters_${smooothing_iterations}.err" >> $job_script
  echo "#SBATCH --time=24:00:00" >> $job_script
  # echo "#SBATCH --partition=<your_partition>" >> $job_script
  echo "#SBATCH --nodes=1" >> $job_script
  echo "#SBATCH --ntasks-per-node=8" >> $job_script
  echo "#SBATCH --mem-per-cpu=16G" >> $job_script
  echo "python $python_script $img_path  $output_dir $voxel_size $smoothing_iters $erosion_iters $dilation_iters $combined_mesh $labels_to_convert >> $job_script

  # Submit the job script
  sbatch $job_script
done