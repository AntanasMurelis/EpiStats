import numpy as np
from tifffile import imwrite

"""
This scripts contains a set of functions that can be used to generate a cube lattice image. The image can be used
as a ground truth to test the scripts that collect the statistics on the segmentation image.
"""

#------------------------------------------------------------------------------------------------------------
def add_cube_to_image(
        voxel_img,
        cube_label,
        x_voxel_start,
        x_voxel_end,
        y_voxel_start, 
        y_voxel_end,
        z_voxel_start, 
        z_voxel_end,
    ):
    """Given a voxel image, this method add a cube to it

    Parameters:
    -----------
        voxel_img: (np.array, 3D)
            The voxel image where the cube should be added

        cube_label: (int)
            The label of the cube in the voxel image

        x_voxel_start (int): 
            The index of the first voxel where the cube should start on the x axis

        x_voxel_end (int):
            The index of the last voxel where the cube should end on the x axis

        y_voxel_start (int):
            The index of the first voxel where the cube should start on the y axis
        
        y_voxel_end (int):
            The index of the last voxel where the cube should end on the y axis

        z_voxel_start (int):
            The index of the first voxel where the cube should start on the z axis

    Returns:
    --------
        voxel_img: (np.array, 3D)
            The voxel image with the cube added

    """
    assert cube_label > 0, "cube_label must be positive, 0 is for the background"
    assert x_voxel_start >= 0, "x_voxel_start must be positive"
    assert y_voxel_start >= 0, "y_voxel_start must be positive"
    assert z_voxel_start >= 0, "z_voxel_start must be positive"

    assert x_voxel_start < x_voxel_end, "x_voxel_start must be smaller than x_voxel_end"
    assert y_voxel_start < y_voxel_end, "y_voxel_start must be smaller than y_voxel_end"
    assert z_voxel_start < z_voxel_end, "z_voxel_start must be smaller than z_voxel_end"

    assert x_voxel_end <= voxel_img.shape[0], "x_voxel_end must be smaller than the size of the image in x axis"
    assert y_voxel_end <= voxel_img.shape[1], "y_voxel_end must be smaller than the size of the image in y axis"
    assert z_voxel_end <= voxel_img.shape[2], "z_voxel_end must be smaller than the size of the image in z axis"

    voxel_img[x_voxel_start:x_voxel_end-1, y_voxel_start:y_voxel_end-1, z_voxel_start:z_voxel_end-1] = cube_label
    return voxel_img
#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
def generate_cube_lattice_image(
    nb_x_voxels, 
    nb_y_voxels,
    nb_z_voxels,    
    cube_side_length, 
    nb_cubes_x,
    nb_cubes_y,
    nb_cubes_z,
    interstitial_space=0,
):
    """Create an image with a lattice of cubes. The cubes are aligned with the axis of the image. The background 
    by default has a value of 0, the cube labels start at 1. Use this function to generate the ground truth for
    the tests.

    Parameters:
    -----------

    nb_x_voxels (int):
        The number of voxels in the x axis of the image
    nb_y_voxels (int):
        The number of voxels in the y axis of the image
    nb_z_voxels (int):
        The number of voxels in the z axis of the image

    cube_side_length (int):
        The side length of the cube in number of of voxels
    
    nb_cubes_x (int):
        The number of cubes in the x axis of the image
    nb_cubes_y (int):
        The number of cubes in the y axis of the image
    nb_cubes_z (int):
        The number of cubes in the z axis of the image

    interstitial_space (int, optional):
        The number of voxels between each cube. Defaults to 0.

    Returns:
    img: (np.array, 3D)
        The image with the lattice of cubes
    """

    assert nb_cubes_x >= 1, "nb_cubes_x must be positive"
    assert nb_cubes_y >= 1, "nb_cubes_y must be positive"
    assert nb_cubes_z >= 1, "nb_cubes_z must be positive"

    #Generate the image with the background (background = 0)
    img = np.zeros((nb_x_voxels, nb_y_voxels, nb_z_voxels), dtype=np.uint8)

    #Add the cubes to the image
    cube_label = 1


    #Start adding the cubes along the x axis, the first cube is added at the origin
    for x_cube in range(nb_cubes_x):
        x_voxel_start = x_cube * (cube_side_length + interstitial_space)
        x_voxel_end = x_voxel_start + cube_side_length

        #Add the cubes along the y axis
        for y_cube in range(nb_cubes_y):
            y_voxel_start = y_cube * (cube_side_length + interstitial_space)
            y_voxel_end = y_voxel_start + cube_side_length

            #Add the cubes along the z axis
            for z_cube in range(nb_cubes_z):
                z_voxel_start = z_cube * (cube_side_length + interstitial_space)
                z_voxel_end = z_voxel_start + cube_side_length

                img = add_cube_to_image(
                    voxel_img=img,
                    cube_label=cube_label,
                    x_voxel_start=x_voxel_start,
                    x_voxel_end=x_voxel_end,
                    y_voxel_start=y_voxel_start,
                    y_voxel_end=y_voxel_end,
                    z_voxel_start=z_voxel_start,
                    z_voxel_end=z_voxel_end,
                )

                cube_label += 1

    return img

#------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    #Generate 3 images: 
    #   - 1 with only one cube in the image
    #   - 2 with a lattice of contiguous cubes (no interstitial space)
    #   - 3 with a lattice of cubes with interstitial space

    #Image 1
    img_1 = generate_cube_lattice_image(
        nb_x_voxels=100,
        nb_y_voxels=100,
        nb_z_voxels=100,
        cube_side_length=10,
        nb_cubes_x=1,
        nb_cubes_y=1,
        nb_cubes_z=1,
        interstitial_space=0,
    )

    #Image 2
    img_2 = generate_cube_lattice_image(
        nb_x_voxels=100,
        nb_y_voxels=100,
        nb_z_voxels=100,
        cube_side_length=10,
        nb_cubes_x=9,
        nb_cubes_y=9,
        nb_cubes_z=9,
        interstitial_space=0,
    )

    #Image 3
    img_3 = generate_cube_lattice_image(
        nb_x_voxels=100,
        nb_y_voxels=100,
        nb_z_voxels=100,
        cube_side_length=10,
        nb_cubes_x=5,
        nb_cubes_y=5,
        nb_cubes_z=5,
        interstitial_space=5,
    )

    #Save the 3 images
    imwrite("img_1.tif", img_1)
    imwrite("img_2.tif", img_2)
    imwrite("img_3.tif", img_3)