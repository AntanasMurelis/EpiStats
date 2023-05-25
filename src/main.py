from tests.CubeLatticeTest import *
from tests.SphereTest import *
from Statistics import collect_cell_morphological_statistics

if __name__ == "__main__":
    
    # Test the mesh generation part by meshing of the test lattice cube images. 
    img = generate_cube_lattice_image(
        nb_x_voxels=200,
        nb_y_voxels=200,
        nb_z_voxels=200,
        cube_side_length=5,
        nb_cubes_x=5,
        nb_cubes_y=5,
        nb_cubes_z=5,
        interstitial_space=-1,
    )

    
    # img = "/Users/antanas/Downloads/relabeled/cell_boundary_time_10_relabeled.tif"
    

    cell_statistics_df = collect_cell_morphological_statistics(labeled_img=img, img_resolution = np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.6, clear_meshes_folder=False, output_folder="./Test_1000", preprocess = True, meshes_only=False, overwrite=True,
                                                              smoothing_iterations=5, erosion_iterations=2, dilation_iterations=5, max_workers=10, calculate_contact_area_fraction=True, plot = 'all', plot_type = 'violin', volume_lower_threshold=5, volume_upper_threshold=5000)

    # cell_statistics_df = collect_cell_morphological_statistics(labeled_img = 'path to .tif or 3D array of your image', img_resolution = np.array([0.21, 0.21, 0.39]), contact_cutoff = 0.2, clear_meshes_folder=True, 
    #                                                             output_folder="Cube_test", preprocess = False, meshes_only=False, overwrite=True, 
    #                                                             max_workers=4, calculate_contact_area_fraction=True, plot = 'all', plot_type = 'violin')
    