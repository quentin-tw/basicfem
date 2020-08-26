import numpy as np
import matplotlib.pyplot as plt

def save_data(matrix, filename):
    np.savetxt(filename, matrix, fmt='%13.6e')

# return new node location that includes the solved displacement
def _shiftNodes(disp, nodal_data, dof, magFactor):
    new_nodal_data = np.zeros(nodal_data.shape)
    for i in range(0,nodal_data.shape[0]):
        new_nodal_data[i,0] = nodal_data[i,0]
        new_nodal_data[i,1] = nodal_data[i,1] + magFactor * disp[i*dof]
        new_nodal_data[i,2] = nodal_data[i,2] + magFactor * disp[i*dof+1]
    return new_nodal_data

# single line plot for 2 nodes
def _plot2DNodes(sol_in, nodal_data, element_index, ax,lineStr):
    (node1_index, node2_index) = sol_in.get_element_property('node_ind', 
                                                          element_index)
    x = [nodal_data[node1_index,1],nodal_data[node2_index,1]]
    y = [nodal_data[node1_index,2],nodal_data[node2_index,2]]
    ax.plot(x, y, lineStr)

# plot the deformed and undeformed shape with assigned scale factor
def shapePlot(disp, sol_in, magFactor):
    new_nodal_data = _shiftNodes(disp, sol_in.nodal_data, sol_in.dof, magFactor)
    fig, ax = plt.subplots()
    for n in range(sol_in.num_of_elements):
        _plot2DNodes(sol_in, sol_in.nodal_data, n, ax,'k-')
        _plot2DNodes(sol_in, new_nodal_data, n, ax,'r-')
    
    myAxSetting = {
        'xlabel' : 'X',  
        'ylabel' : 'Y',
        'aspect' : 'equal',
        'title' : 
        'Deformed Shape (Red), Scale Factor : {:.2f}'.format(magFactor),
    }   
    ax.set(**myAxSetting)
    plt.show()