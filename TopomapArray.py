# -*- coding: utf-8 -*-
"""
These functions are used to pick subplot coordinates when given the XYZ 
coordinates of a sensor array.

@author: KensingtonSka (Rhys Hobbs)
"""
import numpy as np
import matplotlib.pyplot as plt
__version__ = '1.0'

def project_onto_zplane(xyz2use, projection='z-shift',
                        scale_seperation_distance=-1, ch_names=None):
    ''' Projects a set of xyz coordinates onto the z-plane.
        
        Parameters
        ----------
        xyz2use : np.array
            A (n, 3) numpy array of xyz coordinates (in that order).
            
        projection : str, default='z-shift'
            How to project the xyz coordinates onto the z-plane.
                'z': This projection simply sets the z-coordinates to zero.
                'z-scale': This projection scales the y-coordinates by the 
                           magnitude of the z-coordinate. This sets
                           scale_seperation_distance = 1 to ensure no points 
                           are substantially far.
                'stereographic': A standard stereographic projection.

        scale_seperation_distance : int, float, default=1.5
            Number of standard deviations a xy sensor can be away from the 
            mean of the other sensors before it's considered and outlier. If 
            it is conisdered an outlier it's xy coordinates are scaled until 
            it's within the scale_seperation_distance range.
            
        ch_names : list of str
            Used for plotting final coordinate positions. An n length list of 
            strings which identify each coordinate.
          
        Returns
        ----------
        xy2use : np.array
          The result of projecting the xyz coordinates (passed in `xyz2use`) onto the z-plane.
    '''
    ssp = scale_seperation_distance
    ### Build montage positions by applying a cartesian stereographic projection:
    #1) Adjust the xy plane to be tangent ot the top of the head:
        #Find xy center most sensor:
    xy = [np.sqrt(r[0]**2 + r[1]**2)for r in xyz2use]
    xy_i = xy.index(min(xy))
    xyz2use = xyz2use - xyz2use[xy_i]
    
    #2) Apply projection:
    if projection == 'z':
        print('Setting z-coordinates to 0.')
        xy2use = np.array([np.array([r[0], r[1]]) for r in xyz2use])
        
    elif projection == 'z-shift':
        print('Scaling y-coordinate by the magnitude of the z-coordinate.')
        xy2use = np.array([np.array([r[0], r[1]]) for r in xyz2use])
        sign = np.nan_to_num(np.sign(xy2use[:,1]))
        xy2use[:,1] = xy2use[:,1] + np.abs(xyz2use[:, 2])*sign
        
        #Turn on seperation scaling loop:
        ssp=1
        
    elif projection == 'stereographic':
        print('Applying a cartesian stereographic projection.')
        xy2use = np.array([np.array([r[0], r[1]])/(1 - r[2]) for r in xyz2use])
    
    if (not isinstance(ssp, float) and not isinstance(ssp, int)) or isinstance(ssp, bool):
        raise ValueError('scale_seperation_distance must be a numeric!')
    if ssp >= 0:
        # If any sensors are very far from the others then we scale their position:
        exit_clause = False
        print('Entering scale_seperation_distance loop...')
        while not exit_clause:
            # Calculate distance to sensors:
            sen_pos = np.sqrt(xy2use[:,0]**2 + xy2use[:,1]**2)
            outliers = sen_pos > (np.mean(sen_pos) + ssp*np.std(sen_pos))
            
            if any(outliers):
                # Scale the outliers by the mean:
                xy2use[outliers, :] = xy2use[outliers, :]/np.mean(sen_pos)
            else:
                exit_clause = True
    
    # Plot the positions of the sensors:
    if ch_names is not None:
        plt.figure()
        plt.scatter(xy2use[:,0], xy2use[:,1])
        for i, xy in enumerate(xy2use):
            plt.text(xy[0], xy[1], ch_names[i])
        plt.xlabel('x')
        plt.ylabel('y')
            
    return xy2use


def gen_grid_size(xy2use, sbp=1, verbose=True):
    '''
        ## Determining the grid size:
        I'll give a short discription of the logic used here.
        First, I define all xy positions as the lower left corner of 
        a squares. Second, I assert a minimum number of squares that are 
        permitted to be between two plots, 'sbp'. 
          With these assertions in place I can choose an appropriate grid by 
        inspecting the distance between the closest and furtherest points in xy2use.
        The closest points will tell me how many grid squares are in a unit of xy 
        distance and the furtherest will tell me the total grid size.    
        
        Parameters
        ----------
        xy2use : np.array
            Numpy array of the x and y coordinates of the sensor positions.
            
        sbp : int, default=2
          Shortest box path. The number of gridsquares seperating the two closest points (inclusive-1).
          
        verbose : bool, default=True
          Print grid units and the estimated grid size.
          
        Returns
        ----------
        grid_size : int list
          The number of grid spaces to have along the x and y directions of the subplot.
    '''
    ## Closest points:
    # I dont want to work out every combination of distance. That would be n! 
    # points. So instead we sort all the x and y values, compute the abs-distance 
    # between each, and inspect the points relating to the closest two.
    x_sort = np.argsort(xy2use[:,0])
    y_sort = np.argsort(xy2use[:,1])
    
    x_len_diff = [np.abs( xy2use[x_sort[i],0] - xy2use[x_sort[i+1],0] )
                                            for i in range(0, len(x_sort)-1)]
    y_len_diff = [np.abs( xy2use[y_sort[i],1] - xy2use[y_sort[i+1],1] )
                                            for i in range(0, len(x_sort)-1)]
    
    # Index of the closest points in the x/y directions:
    minx_i = x_len_diff.index(min(x_len_diff))
    minx_i = [x_sort[minx_i], x_sort[minx_i+1]]
    miny_i = y_len_diff.index(min(y_len_diff))
    miny_i = [y_sort[miny_i], y_sort[miny_i+1]]
    
    # Shortest distances:
    xy_per_grid = [np.abs(xy2use[minx_i[0],1]-xy2use[minx_i[1],1]),
                   np.abs(xy2use[miny_i[0],0]-xy2use[miny_i[1],0])]
    xy_per_grid = min(xy_per_grid) / (sbp+3) #Units: xy distance per unit grid
    
    if xy_per_grid <= 1e-10:
        raise ValueError('Two points in the xy array you are using are the same.')
    
    # ## Furtherest points:
    xy_grid_size = [np.abs(max(xy2use[:,0])-min(xy2use[:,0])),
                    np.abs(max(xy2use[:,1])-min(xy2use[:,1]))]
    
    #Squares in the final grid:
    grid_size = [int(np.ceil(val/xy_per_grid)) for val in xy_grid_size]
    if verbose:
        print('xy distance per unit gridsquare:', xy_per_grid)
        print('Estimated gridsize needed:', grid_size)
    return grid_size


def project_onto_grid(xy2use, grid_size, rotation_matrix=None, ch_names=None):
    ''' Projects a set of xy coordinates onto a grid of a specified size.
            
        Parameters
        ----------
        xy2use : np.array
            A (n, 2) numpy array of xy coordinates (in that order).
            
        grid_size : list of ints
            The size of the grid that the xy coordinates will be projected 
            onto. Example: [14, 15], where the first is the x axis and the 
            second is the y.

        rotation_matrix : int, str, default=None
            Number of degrees to rotate the xy coordinates by if the final 
            result is in the wrong orientation. Additionally the strings
            '90', '180', and '270' are also accepted.
            
        ch_names : list of str
            Used for plotting final coordinate positions. An n length list of 
            strings which identify each coordinate.
          
        Returns
        ----------
        grid_pos : int list
          The x y positions of each point in xy2use on the grid_size[0] x grid_size[1] subplot grid.
    '''
    ## Choose the grid positions:
    # Default rotation mats:
    if rotation_matrix == '90':
        rotation_matrix = np.array([[0, -1],
                                    [1,  0]])
    elif rotation_matrix == '180':
        rotation_matrix = np.array([[-1,  0],
                                    [ 0, -1]])
    elif rotation_matrix == '270' or rotation_matrix == '-90':
        rotation_matrix = np.array([[ 0, 1],
                                    [-1, 0]])
    if isinstance(rotation_matrix, float) or isinstance(rotation_matrix, int):
        print(f'Assuming %f is in degrees (not radians)' % rotation_matrix)
        deg = rotation_matrix % 360
        rotation_matrix = np.array([[np.cos(np.pi*deg/180), -np.sin(np.pi*deg/180)],
                                    [np.sin(np.pi*deg/180),  np.cos(np.pi*deg/180)]])
    
    # If needed, rotate the xy data:
    if rotation_matrix is not None:
        xy2use = np.array([ np.matmul(xy, rotation_matrix) for xy in xy2use])
        
    # Constracut linear maps from xy space to grid space:
    xy2g_gradient = np.array([[(grid_size[0]-2)/(max(xy2use[:,0])-min(xy2use[:,0])), 0],
                              [0, (grid_size[1]-2)/(max(xy2use[:,1])-min(xy2use[:,1]))]])
    xy2g_const = np.array([1 - (min(xy2use[:,0])*xy2g_gradient[0,0]), 
                           1 - (min(xy2use[:,1])*xy2g_gradient[1,1])])
    
    # Convert all points to grid positions:
    grid_pos = np.matmul(xy2use, xy2g_gradient) + xy2g_const
    grid_pos = np.round(grid_pos).astype(int)
    grid_pos = grid_pos-grid_pos.min()
    
    # Plot the positions of the sensors if requested:
    if ch_names is not None:
        plt.figure()
        plt.scatter(grid_pos[:,0], grid_pos[:,1])
        for i, xy in enumerate(grid_pos):
            plt.text(xy[0], xy[1], ch_names[i])
    
    if grid_pos.max() < max(grid_size):
        print(f'Full grid size is %i larger than needed.' % (max(grid_size)-grid_pos.max()))
    
    return grid_pos
