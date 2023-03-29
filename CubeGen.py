import numpy as np
from misc import cube_filter, add_padding

def cube_gen(x, y, z, **kwargs):
    """Generates a cube with dimensions x, y and z

    Args:
        x (int): The size in x axis
        y (int): The size in y axis
        z (int): The size in z axis

    Returns:
        3D-array: Numpy array of cube
    """
    img = np.ones([x, y, z])
    img[1:x-1, 1:y-1, 1:z-1] = 0

    return img

def gen_lattice(x, y, z, n_x, n_y, n_z, filter = None):
    """Generates a lattice of cubes
        Args:
        x (int): The size in x axis (of cube)
        y (int): The size in y axis
        z (int): The size in z axis
        n_x (int): Number of cubes in x axis
        n_y (int): Number of cubes in y axis
        n_z (int): Number of cubes in z axis
        filter (fun, optional): Function generating a cube of label. Defaults to None.

    Returns:
        tuple: Voxel values of the cube and its segmented label. If filter is None returns cube array only 
    """
    
    x_img = cube_gen(x, y, z)
    for _ in range(n_x-1):
        cube = cube_gen(x, y, z)
        x_img = np.append(x_img, cube, 0)

    y_img = x_img
    for _ in range(n_y-1):
        y_img = np.append(y_img, x_img, 1)
        
    z_img = y_img
    for _ in range(n_z-1):
        z_img = np.append(z_img, y_img, 2)

    img = z_img
    
    # Generates the corresponding label for the image:
    if filter != None:
        cell = 1
        row_label = filter(x, y, z, cell)
        for _ in range(1, n_x):
            cell += 1
            cube = filter(x, y, z, cell)
            row_label = np.append(row_label, cube, 0)
        plane_label = row_label

        for _ in range(1, n_y):
            plane_label = np.append(plane_label, row_label + n_x * _, 1)
            
        cube_label = plane_label
        for _ in range(1, n_z):
            multiplicator =  n_y * n_x * _
            new_cube_label = plane_label + multiplicator
            cube_label = np.append(cube_label, new_cube_label, 2)
        
        return add_padding(img, 1), add_padding(cube_label, 1)
        
    return add_padding(img)

def gen_tight_lattice(x, y, z, n_x, n_y, n_z):
    """Generates a lattice of tight cubes
    Args:
        x (int): The size in x axis (of cube)
        y (int): The size in y axis
        z (int): The size in z axis
        n_x (int): Number of cubes in x axis
        n_y (int): Number of cubes in y axis
        n_z (int): Number of cubes in z axis
    Returns:
        3D-array: A periodic lattice of tightly packed cubes
    """
    
    img = np.ones([x*n_x, y*n_y, z*n_z])
    img[1:(n_x*x)-1, 1:(n_y*y)-1, 1:(n_z*z)-1] = 0

    x_ = x
    y_ = y
    z_ = z
    
    while(x_ < x*n_x):
        img[x_, :, :] = 1
        x_= x_ + x
    while(y_ < y*n_y):
        img[:, y_, :] = 1
        y_= y_ + y
    while(z_ < z*n_z):
        img[:, :, z_] = 1
        z_= z_ + z
        
    return img