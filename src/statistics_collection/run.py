'''
Run this script to collect statistics from epithelial tissues.
'''

import os
import argparse
from misc import create_output_directory, load_labeled_img, read_config
from StatsCollector import StatsCollector
from LabelPreprocessing import process_labels
from GenMeshes import convert_labels_to_meshes

def main(config_path):

    # Read args from config file
    args =  read_config(config_path)

    # Load labeled image
    labeled_img = load_labeled_img(args.input_path)

    # Create output directory
    output_dir = create_output_directory(
        output_folder=args.output_path, 
        input_img_path=args.input_path,
        smoothing_iterations=args.smoothing_iters, 
        erosion_iterations=args.erosion_iters, 
        dilation_iterations=args.dilation_iters
    )

    # Preprocess the labeled image
    preprocessed_labeled_img = process_labels(
        labeled_img=labeled_img, 
        erosion_iterations=args.erosion_iters,
        dilation_iterations=args.dilation_iters,
        output_directory=output_dir,
        overwrite=False
    )

    # Create meshes
    meshes = convert_labels_to_meshes(
        img=preprocessed_labeled_img,
        voxel_resolution=args.voxel_size,
        smoothing_iterations=args.smoothing_iters,
        output_directory=output_dir,
        overwrite=False,
        pad_width=10,
        mesh_file_format=args.mesh_format
    )

    # Collect statistics
    stats_collector = StatsCollector(
        meshes=meshes,
        labels=preprocessed_labeled_img,
        features=args.features,
        output_directory=output_dir,
        path_to_img=os.path.join(output_dir, 'processed_labels.tif'),
        tissue=args.tissue,
        voxel_size=args.voxel_size,
        num_2D_slices=args.num_2D_slices,
        size_2D_slices=args.size_2D_slices,
        num_workers=args.max_workers
    )

    stats_collector.collect_statistics(load_from_cache=args.load_from_cache)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Read JSON configuration file')
    parser.add_argument('--config', help='Path to the config JSON file')
    args = parser.parse_args()

    main(args.config)

