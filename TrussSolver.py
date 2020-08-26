# 2D Truss Solver
# This code solves 2D truss structure
# assigned by an input excel data, and calculate the nodal displacement,
# element stress, and the mass of the structure.
# The deformed shape plot is also provided with a given scale factor.

import numpy as np
import matplotlib.pyplot as plt
from SolverInput import *

class TrussSolver():

    # The main entry point of the program.
    def __init__(self, sol_in):
        self.K_global = self.get_K_global(sol_in)
        self.displacements = self.solve_displacement(sol_in)
        self.stress = self.get_stress(sol_in)

    # return global stiffness matrix by traversing element by element.
    def get_K_global(self, sol_in):
        
        K_global = np.zeros((sol_in.dof * sol_in.num_of_nodes,
                            sol_in.dof * sol_in.num_of_nodes))

        for n in range(sol_in.num_of_elements):
            Kel_local = self.get_Kel_local(sol_in, n)
            T = self.get_T(sol_in, n)
            Kel_global = T.T @ Kel_local @ T
            element_conn = self.get_element_conn(sol_in, n)
            
            # fill in the global stiffness matrix by given element stiffness matrix
            for j in range(sol_in.node_per_element * sol_in.dof):
                for k in range(sol_in.node_per_element * sol_in.dof):
                    K_global[element_conn[j],element_conn[k]] += Kel_global[j,k]
        
        return K_global

    # solve and return nodal displacement using global stiffness matrix 
    # and force vector.
    def solve_displacement(self, sol_in):
        constraintFlags = self.constraint_node_flags(sol_in)  \
                        + self.constraint_node_flags(sol_in)
        
        # indices used for solving displacement
        keepList = np.nonzero(constraintFlags == 0)[0] 
        
        # Note that K_global[keepList,keepList] will not yield desired result.
        K_reduced = self.K_global[keepList][:,keepList]

        # including effect of BC_NH
        displacements, F_reduced = self.initialize_disp_and_force(keepList, sol_in) 
        
        d_reduced = np.linalg.solve(K_reduced,F_reduced)
        displacements[keepList] = d_reduced

        return displacements

    # solve and return stress of each element. 
    # The indexing sequence follows the E matrix.
    def get_stress(self, sol_in):
        
        stress = np.zeros(sol_in.num_of_elements)
        for n in range(sol_in.num_of_elements):
            YM = sol_in.get_element_property('E', n) # Young's modulus
            
            L = sol_in.get_element_property('L', n)

            T = self.get_T(sol_in, n)

            C = YM/L *( np.array([-1,0,1,0]) @ T )
            element_conn = self.get_element_conn(sol_in, n)
            d_el = self.displacements[element_conn]
            stress[n] = C @ d_el
        return stress

    # returns the element stiffness matrix in local coordinates, 
    # given a row in E matrix, a P matrix, a M matrix, and the 
    # element length.
    def get_Kel_local(self, sol_in, element_index):
        area = sol_in.get_element_property('A', element_index)
        youngs_modulus = sol_in.get_element_property('E', element_index)
        element_length = sol_in.get_element_property('L', element_index)
        coefficient = area*youngs_modulus/element_length
        Kel_local = coefficient \
                    * np.array([
                                ( 1, 0,-1, 0),
                                ( 0, 0, 0, 0),
                                (-1, 0, 1, 0),
                                ( 0, 0, 0, 0)
                            ])
        return Kel_local 

    # Get the transformation matrix T of given element.
    def get_T(self, sol_in, element_index):
        if sol_in.dof == 2:        
            # Calculate cos and sin value for given element
            directional_cosine = sol_in.get_element_property('dcos', element_index)
            C = directional_cosine[0]
            S = directional_cosine[1]
            T = np.array([
                        ( C, S, 0, 0),
                        (-S, C, 0, 0),
                        ( 0, 0, C, S),
                        ( 0, 0,-S, C)])

        return T

    # return a list which its indices represent the indices of 
    # element stiffness matrix, whereas the elements of the list 
    # points to the indices of the global stiffness matrix.
    # The list is called connectivity matrix.
    def get_element_conn(self, sol_in, element_index):
        node1_index, node2_index = sol_in.get_element_property('node_ind', 
                                                            element_index)

        node1_to_global_index = list(range(node1_index * sol_in.dof,
                                        node1_index * sol_in.dof + sol_in.dof))
        node2_to_global_index = list(range(node2_index * sol_in.dof,
                                        node2_index * sol_in.dof + sol_in.dof))
        indlist_element_to_global = node1_to_global_index \
                                + node2_to_global_index # concatenate
        return indlist_element_to_global        

    # return a 1D numpy of flags, where 1 in the element represents 
    # a fixed dof to the corresponded index.
    def constraint_node_flags(self, sol_in):

        flag_list = np.zeros(sol_in.num_of_nodes * sol_in.dof)

        for row in range(sol_in.bound_con.shape[0]):
            for element in range(1,sol_in.dof+1):             
                if sol_in.bound_con[row,element] != 0:
                    # DOF * del_N_Ind represents the first element
                    # for node index del_N_Ind 
                    del_N_Ind = int( np.nonzero(sol_in.nodal_data[:,0] 
                                                == sol_in.bound_con[row,0])[0])
                    flag_list[ sol_in.dof*del_N_Ind  + (element-1) ] = 1
        
        return flag_list

    # fill-in the force vector from F matrix.
    def fill_nodal_forces(self, sol_in):

        forces = np.zeros(sol_in.dof*sol_in.nodal_data.shape[0])
        for row in range(sol_in.nodal_forces.shape[0]):
            force_ind = int(np.nonzero(sol_in.nodal_data[:,0] == sol_in.nodal_forces[row,0])[0])
            
            for i in range(0,sol_in.dof):
                forces[sol_in.dof*force_ind+i] = sol_in.nodal_forces[row,i+1]
    
        return forces

    # initialize the displacement and force vector for solving displacement.
    # Non-homogeneous boundary conditions are also considered.
    def initialize_disp_and_force(self, keepList, sol_in):


        displacements = np.zeros(sol_in.dof * sol_in.nodal_data.shape[0])
        force = self.fill_nodal_forces(sol_in)

        # modify for non-homogeneous boundary condition.
        # If BC_NH empty, this will be skipped.
        for row in range(sol_in.bound_con_nonhomo.shape[0]): 
            for element in range(1, sol_in.dof+1): 
                if sol_in.bound_con_nonhomo[row,element] != 0:
                    ind_nodal_data = np.nonzero(sol_in.nodal_data[:,0] 
                                        == sol_in.bound_con_nonhomo[row,0])[0]
                    ind_in_disp = ind_nodal_data * sol_in.dof + (element - 1)
                    displacements[ind_in_disp] = sol_in.BC_NH[row,element] # fill in NHBC
                    force[keepList] = force[keepList] \
                                    - self.K_global[keepList][:,ind_nodal_data] \
                                    * displacements[ind_in_disp]
        
        force_reduced = force[keepList]
        
        return displacements, force_reduced      

    # Solved for the reaction forces of nodes (unused).
    def solveForce(self, dMatrix): 
        force = self.K_global @ dMatrix
        return force