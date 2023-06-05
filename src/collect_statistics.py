import argparse
import os
from SegmentationStatisticsCollector import collect_cell_morphological_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect cell morphological statistics")
    
    parser.add_argument("labeled_img", type=str, help="Path to the labeled image (3D numpy array or image file)")
    parser.add_argument("img_resolution", type=float, nargs=3, help="Image resolution in microns (x_res, y_res, z_res)")
    parser.add_argument("contact_cutoff", type=float, help="Cutoff distance in microns for two cells to be considered in contact")
    
    parser.add_argument("--clear_meshes_folder", action="store_true", help="Delete all current meshes in the 'cell_meshes' directory before saving new meshes")
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="Number of smoothing iterations applied to the mesh (default: 5)")
    parser.add_argument("--erosion_iterations", type=int, default=1, help="Number of iterations for erosion during label extension (default: 1)")
    parser.add_argument("--dilation_iterations", type=int, default=2, help="Number of iterations for dilation during label extension (default: 2)")
    parser.add_argument("--output_folder", type=str, default="output", help="Name of the folder where the processed labels, cell_meshes, and filtered_cell_meshes will be saved (default: 'output')")
    parser.add_argument("--meshes_only", action="store_true", help="Save meshes only without computing statistics")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite preprocessed labels if they already exist")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing on the input labeled image")
    parser.add_argument("--calculate_contact_area_fraction", action="store_true", help="Calculate the contact area fraction for each cell (default: True)")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of workers for parallel computation (default: number of CPU cores - 1)")
    parser.add_argument("--plot", type=str, choices=['filtered', 'both', 'all', None], default=None, help="Generate plots for the cell statistics. Options: 'filtered', 'both', 'all', or None (default: None)")
    parser.add_argument("--plot_type", type=str, default="violin", help="Type of plot to generate (default: 'violin')")
    parser.add_argument("--volume_lower_threshold", type=int, default=None, help="Minimum volume to include a cell in the filtered labels (default: None)")
    parser.add_argument("--volume_upper_threshold", type=int, default=None, help="Maximum volume to include a cell in the filtered labels (default: None)")
    
    args = parser.parse_args()

    filtered_cell_statistics = collect_cell_morphological_statistics(
        labeled_img=args.labeled_img,
        img_resolution=args.img_resolution,
        contact_cutoff=args.contact_cutoff,
        clear_meshes_folder=args.clear_meshes_folder,
        smoothing_iterations=args.smoothing_iterations,
        erosion_iterations=args.erosion_iterations,
        dilation_iterations=args.dilation_iterations,
        output_folder=args.output_folder,
        meshes_only=args.meshes_only,
        overwrite=args.overwrite,
        preprocess=args.preprocess,
        calculate_contact_area_fraction=args.calculate_contact_area_fraction,
        max_workers=args.max_workers,
        plot=args.plot,
        plot_type=args.plot_type,
        volume_lower_threshold=args.volume_lower_threshold,
        volume_upper_threshold=args.volume_upper_threshold
    )