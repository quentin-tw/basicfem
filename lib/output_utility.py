""" Utility functions for finite element analysis result output. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections

def save_data(matrix, filename, dir_name=None):
    """ Store matrix in an txt file with given filename. """
    if dir_name is None:
        np.savetxt(filename, matrix, fmt='%13.6e')
    else:
        np.savetxt('./' + dir_name + '/'+filename, matrix, fmt='%13.6e')


def plot_deformed_shape_1D(fem_result, mag_factor, output_dir_path, 
                           has_original_shape=True):
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
            _plot_undeformed_element(fem_result.sol_in, 
                fem_result.sol_in.nodal_data, i, ax, 'k-')

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
    plt.savefig(output_dir_path + '/deformaion_plot.png')

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

def _polycollection_plot(nodes, elements, values, 
                        title_string, output_dir_path):

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
    
def plot_contour(fem_result, value_type, direction, titleStr, 
                    deformation_scale, output_dir_path, plot_on_deformed=True):
    num_nodes = fem_result.sol_in.node_per_element
    if plot_on_deformed:
        nodes = _shift_nodal_coord(fem_result.sol_in.nodal_data, 
                                fem_result.displacements, deformation_scale)
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
                         titleStr, output_dir_path)
    
def _show_overlap(nodes, nodes2, elements, titleStr, output_dir_path):

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


def plot_deformation_shape_2D(N, E, disp, deformedScale, output_dir_path):
    numOfNodes = E.shape[1]-2
    nodes = N[:,1:3].copy() 
    nodes_new = N[:,1:3].copy()
    disp_m = disp * deformedScale
    for i in range(0,N.shape[0]):
        nodes_new[i,0] += disp_m[2*i]
        nodes_new[i,1] += disp_m[2*i+1]

    # the corresponding nodes (nodal indices) for each elements
    # assume that nodal labels start from one and are in sequence
    elements = E[:,1:numOfNodes+1] - 1 
    elements = elements.astype(int)
    _show_overlap(nodes, nodes_new, elements,"deformed shape plot", 
                        output_dir_path)

def _shift_nodal_coord(nodal_data, displacements, deformation_scale):
    shifted_nodal_coord = nodal_data[:,1:3].copy()
    scaled_disp = displacements * deformation_scale
    for i in range(nodal_data.shape[0]):
        shifted_nodal_coord[i, 0] += scaled_disp[2 * i]
        shifted_nodal_coord[i, 1] += scaled_disp[2 * i + 1]
    return shifted_nodal_coord


