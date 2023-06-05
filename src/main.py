from Statistics import collect_cell_morphological_statistics
import hydra
from omegaconf import DictConfig
import numpy as np
from tests.CubeLatticeTest import *
from tests.SphereTest import *

import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="../conf", config_name="config.yml")
def main(cfg: DictConfig) -> None:
    labeled_img_path = cfg.labeled_img
    img_resolution = np.array(cfg.img_resolution)
    contact_cutoff = cfg.contact_cutoff
    smoothing_iterations = cfg.smoothing_iterations
    erosion_iterations = cfg.erosion_iterations
    dilation_iterations = cfg.dilation_iterations
    output_folder = cfg.output_folder
    meshes_only = cfg.meshes_only
    overwrite = cfg.overwrite
    preprocess = cfg.preprocess
    calculate_contact_area_fraction = cfg.calculate_contact_area_fraction
    max_workers = cfg.max_workers
    plot = cfg.plot
    plot_type = cfg.plot_type
    volume_lower_threshold = cfg.volume_lower_threshold
    volume_upper_threshold = cfg.volume_upper_threshold


    # Call the function with the parameters
    collect_cell_morphological_statistics(
        labeled_img=labeled_img_path,
        img_resolution=img_resolution,
        contact_cutoff=contact_cutoff,
        smoothing_iterations=smoothing_iterations,
        erosion_iterations=erosion_iterations,
        dilation_iterations=dilation_iterations,
        output_folder=output_folder,
        meshes_only=meshes_only,
        overwrite=overwrite,
        preprocess=preprocess,
        max_workers=max_workers,
        calculate_contact_area_fraction=calculate_contact_area_fraction,
        plot=plot,
        plot_type=plot_type,
        volume_lower_threshold=volume_lower_threshold,
        volume_upper_threshold=volume_upper_threshold
    )

if __name__ == "__main__":
    main()