""" Utility functions for finite element analysis result output. """

import numpy as np
import matplotlib.pyplot as plt

def save_data(matrix, filename):
    """ Store matrix in an txt file with given filename. """
    np.savetxt(filename, matrix, fmt='%13.6e')


def deformedShapePlot(fem_result, mag_factor, has_original_shape=True):
    """Plot the deformed shape of the structure.
    
    Parameters
    ----------
    fem_result: The instance of the FEM solver.
    mag_factor: The magnification factor of deformation.
    has_original_shape: Set true if the undeformed shape is required.
    
    """

    interpolated_data = fem_result.interpolate_shape(mag_factor)
    fig, ax = plt.subplots()

    if has_original_shape:
        for i in range(fem_result.sol_in.num_of_elements):
            _plot_undeformed_element(fem_result.sol_in, fem_result.sol_in.nodal_data, i, ax, 'k-')

    for i in range(fem_result.sol_in.num_of_elements):
        _plot_deformed_element(ax, 'r-', interpolated_data[i,:,:])

    myAxSetting = {
        'xlabel' : 'X',  
        'ylabel' : 'Y',
        'aspect' : 'equal',
        'title' : 
        'Deformed Shape (Red), Scale Factor : {:.2f}'.format(mag_factor),
    }   
    ax.set(**myAxSetting)
    plt.show()

def _plot_undeformed_element(sol_in, nodal_data, element_index, ax, lineStr):
    """ Plot a single undeformed element. """
    (node1_index, node2_index) = sol_in.get_element_property('node_ind', 
                                                          element_index)
    x = [nodal_data[node1_index,1], nodal_data[node2_index,1]]
    y = [nodal_data[node1_index,2], nodal_data[node2_index,2]]
    ax.plot(x, y, lineStr)

def _plot_deformed_element(ax, lineStr, element_interpolated_data):
    """ Plot a single deformed element. """
    x = element_interpolated_data[:,0]
    y = element_interpolated_data[:,1]
    ax.plot(x, y, lineStr)


    
