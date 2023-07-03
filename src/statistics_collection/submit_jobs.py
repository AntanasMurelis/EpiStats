import os
import json
import subprocess
from typing import Optional

#-----------------------------------------------------------------------------
def create_slurm(
    config_file: str, 
    name: str,
    max_workers: int,
    jobs_dir: Optional[str] = './jobs',
) -> str:
    '''
    Create a bash script to run the training on cluster.

    Parameters:
    -----------
        config_file: (str)
            The path to the config file to use for the current run.

        name: (str)
            A string naming the current config file.

        jobs_dir: (Optional[str], default='./jobs')
            The directory where to store the job files.

        max_workers: (int)
            The number of workers for parallel computation

    Returns:
    --------
        script_file: (str)
            The path to the bash file to execute.
    '''

    bash_script = f"""\
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task={max_workers}
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16384

python src/statistics_collection/run.py --config {config_file}   
    """

    script_file = f'submit_training_{name}.sh'

    save_script_dir = jobs_dir
    if not os.path.exists(save_script_dir):
        os.makedirs(save_script_dir)
    with open(os.path.join(save_script_dir, script_file), 'w') as file:
        file.write(bash_script)

    return os.path.join(save_script_dir, script_file)
#-----------------------------------------------------------------------------


#Read the config.json file
with open("src/statistics_collection/config.json", "r") as file:
    curr_config = json.load(file)

#Set ranges of parameters to modify in the config
tissues = ['intestine_villus', 'lung_bronchiole', 'bladder', 'esophagus', 'lung']
voxel_sizes = [
    [0.325, 0.325, 0.25],
    [0.1625, 0.1625, 0.25],
    [0.21, 0.21, 0.39],
    [0.1625, 0.1625, 0.25],
    [0.1, 0.1, 0.1]
]
COMMON_ROOT = './curated_labels/'
input_files = [
    'intestine_sample2_b_curated_segmentation_relabel_seq.tif',
    'lung_new_sample_b_curated_segmentation_central_crop_relabel_seq.tif',
    'bladder_control_curated_segmentation.tif',
    'esophagus_Z2_curated_crop.tif',
    'lung_pseudostratified_from_harold.tif'
]
input_paths = [os.path.join(COMMON_ROOT, input_file) for input_file in input_files]

#For each tissue create a new config file and a new job submit script
for tissue, voxel_size, input_path in zip(tissues, voxel_sizes ,input_paths):
    #update config file
    curr_config["tissue"] = tissue
    curr_config["voxel_size"] = voxel_size
    curr_config["input_path"] = input_path
    
    #save the new config dictionary
    save_config_dir = './run_euler/configs'
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    with open(os.path.join(save_config_dir, f"config_{tissue}.json"), "w") as f_out:
        json.dump(curr_config, f_out)
    
    #call a function to create a slurm script 
    path_to_job_script = create_slurm(
        config_file=os.path.join(save_config_dir, f"config_{tissue}.json"), 
        name=tissue,
        max_workers=curr_config["max_workers"],
        jobs_dir='./run_euler/jobs'
    )
    
    #execute the created job  
    subprocess.run(['sbatch', path_to_job_script], stdin=subprocess.PIPE)
