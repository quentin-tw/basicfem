""" Utility functions for finite element analysis result output. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections

def save_data(array, filename, dir_name=None):
    """ A wrapper function for numpy.savetxt to store an array in a txt file. 
    
    parameter
    ---------
    matrix : numpy.ndarray
        The numpy array to be saved into txt file.
    filename : str
        The filename for the txt file.
    dir_name : str
        The name of the directory to save the txt file. Only name is needed.
        Example:
        frame_output
        frame_output/frame_data

    """

    if dir_name is None:
        np.savetxt(filename, array, fmt='%13.6e')
    else:
        np.savetxt('./' + dir_name + '/'+filename, array, fmt='%13.6e')


def plot_deformed_shape_1D(fem_result, scale_factor, output_dir_path, 
                           has_original_shape=True):
    """Plot the deformed shape of the 1D element structures (truss and frames).
    
    Parameters
    ----------
    fem_result : BaseSolver
        A Solver object contains the analysis result.
    scale_factor : float
        The deformation scale factor assigned in the param.
    output_dir_path : str
        The path to the output directory.
    has_original_shape : bool
        If True, the undeformed shape will also be plotted on the same figure.
    
    """

    interpolated_data = fem_result.interpolate_shape(scale_factor)
    fig, ax = plt.subplots()

    if has_original_shape:
        for i in range(fem_result.sol_in.num_of_elements):
            _plot_undeformed_element(fem_result.sol_in, i, ax, 'k-')

    for i in range(fem_result.sol_in.num_of_elements):
        _plot_deformed_element(ax, 'r-', interpolated_data[i,:,:])

    myAxSetting = {
        'xlabel' : 'X',  
        'ylabel' : 'Y',
        'aspect' : 'equal',
        'title' : 
        'Deformed Shape (Red), Scale Factor : {:.2f}'.format(scale_factor),
    }   
    ax.set(**myAxSetting)
    plt.savefig(output_dir_path + '/deformaion_plot.png')

def _plot_undeformed_element(sol_in, element_index, ax, line_str):
    """ Plot a single undeformed 1D element. 

    A helper function for plot_deformed_shape_1D.

    Parameters
    ----------
    sol_in : SolverInput
        An SolverInput object contains all input data.
    element_index : int
        The index points to a SolverInput.element_data row.
    ax : matplotlib.axes._subplots.AxesSubplot
        A matplotlib axes object to be plotted.
    line_str : str
        A string specified the line or point plotting style for the element.
        Example:
        'k-'
        'r--'
    """
    nodal_data = sol_in.nodal_data


    (node1_index, node2_index) = sol_in.get_element_property('node_ind', 
                                                          element_index)
    x = [nodal_data[node1_index,1], nodal_data[node2_index,1]]
    y = [nodal_data[node1_index,2], nodal_data[node2_index,2]]
    ax.plot(x, y, line_str)

def _plot_deformed_element(ax, lineStr, element_interpolated_data):
    """ Plot a single deformed 1D element. 

    A helper function for plot_deformed_shape_1D.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        A matplotlib subplot axes object to be plotted.
    line_str : str
        A string specified the line or point plotting style for the element.
        Example:
        'k-'
        'r--'
    element_interpolated_data : numpy.ndarray
        The interpolated coordinated data inside an element. Note that 
        interpolation only applied in frame structure, since truss structure
        will only subject to axial forces.

    """

    x = element_interpolated_data[:,0]
    y = element_interpolated_data[:,1]
    ax.plot(x, y, lineStr)

def _polycollection_plot(nodes, elements, values, 
                        title_string, output_dir_path):
    """ Generate the contour plot by matplotlib's polycollection class.

    A helper function for plot_countour.

    Parameters
    ----------
    nodes : numpy.ndarray
        The array of nodal coordinates, which is the nodal_data without
        nodal labels.
    elements : numpy.ndarray
        The array of nodal index groups of elements, which is the element_data
        without the element label and the property label, and adjust the nodal
        label to nodal index.
    values : numpy.ndarray
        The array of values correspond with each value corresponds to an 
        elements.
    title_string : str
        A string to be show as the title of the plot
    output_dir_path : str
        The path to the output directory.

    """

    y = nodes[:,0]
    z = nodes[:,1]

    def quatplot(y,z, quatrangles, values, ax=None, **kwargs):

        yz = np.c_[y,z] 
        verts= yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        pc.set_array(values)
        ax.add_collection(pc)
        ax.autoscale()
        return pc

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    pc = quatplot(y,z, np.asarray(elements), values, ax=ax, 
            edgecolor="crimson", cmap="rainbow",linewidths = 0)

    v1 = np.linspace(values.min(), values.max(), 7, endpoint=True)
    cbar = fig.colorbar(pc, ax=ax,ticks = v1)
    cbar.set_label('', rotation=270)     
    ax.set(title=title_string, xlabel='X', ylabel='Y')
    plt.savefig(output_dir_path + '/contour_plot_' + title_string + '.png')
    
def plot_contour(fem_result, value_type, direction, title_string, 
                    scale_factor, output_dir_path, plot_on_deformed=True):
    """ Generate the contour plot of given values for 2D elements.

    The contour can be plotted on the deformed grid or undeformed grid. Only a
    single color is assigned to an element.

    Note that currenly the nodal labeling must starts from 1 and 
    increment continuously for the given strucuture, which is generally
    satisfied by the msh file provided by the Gmsh software.

    Parameters
    ----------
    fem_result : BaseSolver
        A Solver object contains the analysis result.
    value_type : str
        A string that designated the type of value to be plotted. Currently 
        accepts either 'stress' or 'strain'.
    direction : str
        A string designate the direction of the stress or strain tensor. 
        Currently accepts 'xx', 'yy', or 'xy' 
    title_string : str
        A string to be show as the title of the plot
    scale_factor : float
        The deformation scale factor assigned in the param.        
    output_dir_path : str
        The path to the output directory.
    plot_on_deformed : bool
        Flag that determines if the contour is plotted on the deformed
        structures. Default is True.
    
    """

    num_nodes = fem_result.sol_in.node_per_element
    if plot_on_deformed:
        nodes = _shift_nodal_coord(fem_result.sol_in.nodal_data, 
                                fem_result.displacements, scale_factor)
    else:
        nodes = fem_result.sol_in.nodal_data[:,1:3]
    # the corresponding nodes (nodal indices) for each elements
    # assume that nodal labels start from one and are in sequence
    elements = fem_result.sol_in.element_data[:,1:num_nodes + 1] - 1
    elements = elements.astype(int)

    if direction == 'xx':
        dir_ind = 0
    elif direction == 'yy':
        dir_ind = 1
    elif direction == 'xy':
        dir_ind = 2
    else:
        print("invalid direction label, xx is used")
        dir_ind = 0

    if value_type == 'stress':
        elementVal = fem_result.stress[:, dir_ind]

    elif value_type == 'strain':
        elementVal = fem_result.strain[:, dir_ind]  
    
    _polycollection_plot(nodes, elements, elementVal, 
                         title_string, output_dir_path)
    
def _show_overlap(nodes, nodes2, elements, titleStr, output_dir_path):
    """ 
    Generate the grid plot of both deformed and undeformed shape 
    by matplotlib's polycollection class. A helper function for 
    plot_deformation_shape_2D.

    The function is similar to _polycollection_plot, the difference is that 
    the grid is not filled with colors, and both deformed and undeformed grid
    are plotted together in the figure.

    Parameters
    ----------
    nodes : numpy.ndarray
        The array of nodal coordinates of undeformed shape, which is 
        the nodal_data without nodal labels.
    nodes2 : numpy.ndarray
        The array of nodal coordinates of deformed shape, which is the 
        nodal_data without nodal labels, and add the effect of deformation and
        scale factor.
    elements : numpy.ndarray
        The array of nodal index groups of elements, which is the element_data
        without the element label and the property label, and adjust the nodal
        label to nodal index.
    title_string : str
        A string to be show as the title of the plot
    output_dir_path : str
        The path to the output directory.

    """
    y = nodes[:,0].copy()
    z = nodes[:,1].copy()
    y2 = nodes2[:,0].copy()
    z2 = nodes2[:,1].copy()

    def quatplot(y,z,y2,z2, quatrangles, ax=None):
        yz = np.c_[y,z] 
        yz2 = np.c_[y2,z2]
        verts= yz[quatrangles]
        verts2= yz2[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, 
        facecolors = 'none', edgecolors = 'k',linewidths = 0.2)
        pc2 = matplotlib.collections.PolyCollection(verts2, 
        facecolors = 'none', edgecolors = 'r',linewidths = 0.2)
        ax.add_collection(pc)
        ax.add_collection(pc2)
        ax.autoscale()
        return pc2

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    pc2 = quatplot(y,z,y2,z2, np.asarray(elements), ax=ax) 
    ax.set(title=titleStr, xlabel='X', ylabel='Y')
    plt.savefig(output_dir_path + '/deformaion_plot.png')


def plot_deformation_shape_2D(nodal_data, element_data, 
        displacements, scale_factor, output_dir_path):
    """ Generate deformation plot for general 2d elements.
    
    The undeformed shape will also be superimposed on the figure.
    
    Parameter
    ---------
    nodal_data : numpy.array
        the nodal_data from the SolverInput object
    element_data : numpy.array
        the element_data from the SolverInput object
    displacements : numpy.array
        the displacements from the BaseSolver object  
    scale_factor : float
        The deformation scale factor assigned in the param.
    output_dir_path : str
        The path to the output directory.
    
    """

    numOfNodes = element_data.shape[1]-2
    nodes = nodal_data[:,1:3].copy() 
    nodes_new = nodal_data[:,1:3].copy()
    disp_m = displacements * scale_factor
    for i in range(0,nodal_data.shape[0]):
        nodes_new[i,0] += disp_m[2*i]
        nodes_new[i,1] += disp_m[2*i+1]

    # the corresponding nodes (nodal indices) for each elements
    # assume that nodal labels start from one and are in sequence
    elements = element_data[:,1:numOfNodes+1] - 1 
    elements = elements.astype(int)
    _show_overlap(nodes, nodes_new, elements,"deformed shape plot", 
                        output_dir_path)

def _shift_nodal_coord(nodal_data, displacements, scale_factor):
    """ Return the nodal coordinates after deformation.

    A helper function for plot_countour.

    Parameter
    ---------
    nodal_data : numpy.array
        the nodal_data from the SolverInput object
    displacements : numpy.array
        the displacements from the BaseSolver object  
    scale_factor : float
        The deformation scale factor assigned in the param.
    
    Return
    ------
    numpy.ndarray
        An array that contains the shifted coordinate by the deformation
    """
    shifted_nodal_coord = nodal_data[:,1:3].copy()
    scaled_disp = displacements * scale_factor
    for i in range(nodal_data.shape[0]):
        shifted_nodal_coord[i, 0] += scaled_disp[2 * i]
        shifted_nodal_coord[i, 1] += scaled_disp[2 * i + 1]
    return shifted_nodal_coord


