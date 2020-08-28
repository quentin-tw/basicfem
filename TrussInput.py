import numpy as np
import pandas as pd

class TrussInput:
    """The object representing the input data of the structure to be solved. 

    *** Now supports 2D truss structures only. 3D truss, beams and 2D elements
        under constructions.
        Accepts input data via an excel file only. Additional interface under
        construction.

    Parameters
    ----------
    filename: The name of the input file containing struture's data.

    problem_type : The type of structure to be solved.
        'truss' : 2D or 3D truss problems.
        'frame' : 2D or 3D frame problems.
        'planar' : Problem of 2D elements.

    Attributes
    ----------
    nodal_data : an array containing the nodal labels and nodal coordinates. 
                 The structure is as follows:
                 1st column : node labels
                 2nd column : x coordinate of the node
                 3rd column : y coordinate of the node
                 4th column : z coordinate of the node (if applicable)

    element_data : an array containing cooresponding nodes and properties of 
                   the elements. The structure is as follows:
                   1st column : element labels
                   2nd column : first nodes
                   3rd column : second nodes
                   4th column : property number (see properties array)

    bound_con : an array containing the nodal fixed boundary conditions of 
                the problem. The 1st column is the corresponding nodal labels,
                the remaining columns corresponds to conditions in x, y, and 
                z coordinates. 1 represents fixed, 0 represents fixed.
                Nodes not included in this array are treated as free nodes.
    
    bound_con_nonhomo : an array containing the non-homogeneous conditions.
                        Similar structure as bound_con, except that columns 
                        corresponding x, y, and z coodinates are the magnitude
                        of displacements.

    nodal_forces : an array containing the external nodal forces. Similar 
                   structure as bound_con, except that the columns
                   corresponding x, y, and z coordinates are the magnitude of
                   displacements.
    
    properties : an array containing the properties of all elements. The structure
                 is as follows:
                 1st column : property labels corresponds to array 'element_data'
                 2nd column : material number corresponds to array 'materials'
                 3rd column : cross-sectional area of the cooresponding elements.
                 4th column : z Moment of inertia of the corresponding elements.
    
    materials : an array containing the properties of all elements. The structure
                is as follows:
                1st column : material labels corresponds to array 'properties'
                2nd column : Young's modulus
                3rd column : Density of the material
    
    node_per_element: Number of node per element. 2 for truss and frame problems.

    dof : Degree of freedoms of nodes.

    num_of_nodes : total number of nodes of the structure.

    num_of_elements : total number of elements of the structure.

    Method
    ----------
    get_element_property(property_type, element_num, is_label=False)                                    
        returns the property of a single elements.
    
    """
    
    def __init__(self, filename, problem_type):
        if problem_type == 'truss':
            self._read_excel_input(filename)
            self.node_per_element = 2
            self.dof = self.nodal_data.shape[1]-1
            self.num_of_nodes = self.nodal_data.shape[0]
            self.num_of_elements = self.element_data.shape[0]

    def _read_excel_input(self, filename):
        """get input data of the structure from a excel file."""

        self.nodal_data = pd.read_excel(filename, sheet_name = 'N').values
        self.element_data = pd.read_excel(filename, sheet_name = 'E').values
        self.bound_con = pd.read_excel(filename, sheet_name = 'BC').values
        self.bound_con_nonhomo = \
            pd.read_excel(filename, sheet_name = 'BC_NH').values
        self.nodal_forces = pd.read_excel(filename, sheet_name = 'F').values
        self.properties = pd.read_excel(filename, sheet_name = 'P' ).values
        self.materials = pd.read_excel(filename, sheet_name = 'M').values

    def get_element_property(self, property_type, element_num, is_label=False):
        """Returns the property of a single elements
        
            parameters
            ----------
            property_type: a string that indicates the property to return.
                           'A' : area
                           'E' : Young's modulus
                           'I' : moment of inertia
                           'rho' : material density
                           'Yc' : compressive yield strength
                           'Yt' : tensile yield strenth
                           'L' : element length
                           'dcos' : directional cosine values of the element
                           'node_ind' : indicies in the array 'nodal_data'
                                        corresponding to the element

            element_num : an integer indicates the index or the label of 
                          the element. 

            is_label : a boolean indicates if the variable 'element_num' is a
                       label. If it is true, the function will search for the
                       element property using the element's label. If it is 
                       false, the function will search for the element property
                       using the index of the array 'element_data'.
                       The default is false.          
        
        """

        if is_label:
            element_num = \
                np.argwhere(self.element_data[:,0] == element_num).item(0)
            
        property_label = self.element_data[element_num, 3]
        property_index = np.argwhere(self.properties[:,0]==property_label).item(0)
        material_label = self.properties[property_index,1]
        material_index = np.argwhere(self.materials[:,0]==material_label).item(0)

        if property_type == 'A':
            result = self.properties[property_index,2]
        elif property_type == 'E':
            result = self.materials[material_index,1]
        elif property_type == 'I':
            result = self.properties[property_index,3]
        elif property_type == 'rho':
            result = self.materials[material_index,2]
        elif property_type == 'Yc': # compressive yield strength 
            result = self.materials[material_index,3]
        elif property_type == 'Yt': # tensile yield strength
            result = self.materials[material_index,4]        
        elif property_type == 'L' or property_type == 'dcos' \
                or property_type == 'node_ind':
            result = self._node_derived_properties(property_type, element_num)

        return result

    def _node_derived_properties(self, property_type, element_num):
        """returns the property related to the nodal data"""

        node1_index = np.argwhere(self.nodal_data[:,0] == 
            self.element_data[element_num,1]).item(0)
        node2_index = np.argwhere(self.nodal_data[:,0] == 
            self.element_data[element_num,2]).item(0)
        
        if property_type == 'node_ind':
            return (node1_index, node2_index)

        length = 0
        for i in range(1, self.nodal_data.shape[1]):
            length += (self.nodal_data[node1_index, i] 
                       - self.nodal_data[node2_index, i]) ** 2 
        length = length ** 0.5
        
        if property_type == 'L':
            return length
        else:
            cx = (self.nodal_data[node2_index,1] 
                  - self.nodal_data[node1_index,1]) / length
            cy = (self.nodal_data[node2_index,2]
                  - self.nodal_data[node1_index,2]) / length
            if self.dof == 2:
                cz = 0
            else:
                cz = (self.nodal_data[node2_index,3] 
                      - self.nodal_data[node1_index,3]) / length
            return np.array([cx,cy,cz])
    