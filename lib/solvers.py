""" This module contains all the main FEM solver classes """

import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractclassmethod
from lib.solver_input import *

class BaseSolver(metaclass=ABCMeta):
    """ The base class for all the solvers """

    def __init__(self, SolverInput):
        self.sol_in = SolverInput
        self.global_conn = []  #  global connectivity matrix for K
        self.K_global = self.get_K_global()
        self.displacements = self.solve_displacement()

    def get_K_global(self):
        """ Returns the global stiffness matrix. """

        total_dof = self.sol_in.dof * self.sol_in.num_of_nodes
        K_global = np.zeros((total_dof, total_dof))

        for n in range(self.sol_in.num_of_elements):
            Kel_global = self.get_Kel_global(n)
            element_conn = self.get_element_conn(n)
            self.global_conn.append(element_conn)

            # fill in the global stiffness matrix by given 
            # element stiffness matrices
            for j in range(self.sol_in.node_per_element * self.sol_in.dof):
                for k in range(self.sol_in.node_per_element * self.sol_in.dof):
                    K_global[element_conn[j],element_conn[k]] \
                        += Kel_global[j,k]
        
        return K_global
    
    @abstractmethod
    def get_Kel_global(self, element_index):
        """ 
        Returns the element stiffness matrix in global coordinates
        for given element index. 

        """

        pass

    def solve_displacement(self):
        """ 
        Solve and return the nodal displacements using 
        global stiffness matrix and force vector. 
        
        """

        constraintFlags = self.constraint_node_flags()  \
                        + self.constraint_node_flags()
        
        # indices used for solving displacements
        keepList = np.nonzero(constraintFlags == 0)[0] 
        
        # Note: K_global[keepList,keepList] will not yield desired result.
        K_reduced = self.K_global[keepList][:,keepList]

        # including the effect of BC_NH
        displacements, F_reduced = self.initialize_disp_and_force(keepList)
        
        d_reduced = np.linalg.solve(K_reduced,F_reduced)
        displacements[keepList] = d_reduced

        return displacements

    def get_element_conn(self, element_index):
        """
        return a list which its indices represent the indices of 
        element stiffness matrix, whereas the elements of the list 
        points to the indices of the global stiffness matrix.
        The list is called connectivity matrix.

        """
        node_indices = self.sol_in.get_element_property('node_ind', 
                                                            element_index)

        element_conn = []
        for i in range(self.sol_in.node_per_element):
            node_indices_global = list(range(node_indices[i] * self.sol_in.dof,
                                             node_indices[i] * self.sol_in.dof 
                                                + self.sol_in.dof))
            element_conn.extend(node_indices_global)
        return element_conn        

    def constraint_node_flags(self):
        """ 
        return a 1D numpy of flags, where 1 in the element represents 
        a fixed dof to the corresponded index.
        
        """

        flag_list = np.zeros(self.sol_in.num_of_nodes * self.sol_in.dof)

        for row in range(self.sol_in.bound_con.shape[0]):
            for element in range(1,self.sol_in.dof+1):             
                if self.sol_in.bound_con[row,element] != 0:
                    # DOF * del_N_Ind represents the first element
                    # for node index del_N_Ind 
                    del_N_Ind = int( 
                        np.nonzero(self.sol_in.nodal_data[:,0] 
                                   == self.sol_in.bound_con[row,0])[0])
                    flag_list[ self.sol_in.dof*del_N_Ind  + (element-1) ] = 1
        
        return flag_list

    def fill_nodal_forces(self):
        """ fill-in the force vector from F matrix. """ 

        forces = np.zeros(self.sol_in.dof*self.sol_in.nodal_data.shape[0])
        for row in range(self.sol_in.nodal_forces.shape[0]):
            force_ind = int(np.nonzero(self.sol_in.nodal_data[:,0] == 
                                        self.sol_in.nodal_forces[row,0])[0])
            
            for i in range(0,self.sol_in.dof):
                forces[self.sol_in.dof*force_ind+i] \
                    = self.sol_in.nodal_forces[row,i+1]
    
        return forces

    def initialize_disp_and_force(self, keepList):
        """ 
        initialize the displacement and force vector for solving displacement.
        Non-homogeneous boundary conditions are also considered.

        """

        displacements = np.zeros(self.sol_in.dof 
                                 * self.sol_in.nodal_data.shape[0])
        force = self.fill_nodal_forces()

        # modify the displacements and force matrices 
        # for non-homogeneous boundary condition.
        # If BC_NH empty, this section will be skipped.
        for row in range(self.sol_in.bound_con_nonhomo.shape[0]): 
            for element in range(1, self.sol_in.dof+1): 
                if self.sol_in.bound_con_nonhomo[row,element] != 0:
                    ind_nodal_data = np.nonzero(self.sol_in.nodal_data[:,0] 
                                    == self.sol_in.bound_con_nonhomo[row,0])[0]
                    ind_in_disp = ind_nodal_data \
                    * self.sol_in.dof + (element - 1)

                    # fill in NHBC
                    displacements[ind_in_disp] = self.sol_in.BC_NH[row,element] 

                    force[keepList] = force[keepList] \
                                - self.K_global[keepList][:,ind_nodal_data] \
                                * displacements[ind_in_disp]
        
        force_reduced = force[keepList]
        
        return displacements, force_reduced      

    def solveForce(self, dMatrix): 
        """ Solved for the reaction forces of nodes. """

        force = self.K_global @ dMatrix
        return force

class TrussSolver2D(BaseSolver):
    """ Solver class for 2D truss structures. """

    def __init__(self, SolverInput):
        BaseSolver.__init__(self, SolverInput)
        self.stress = self.get_stress()

    def get_stress(self):
        """ Implementation of get_stress method for BaseSolver class """
        
        stress = np.zeros(self.sol_in.num_of_elements)
        for n in range(self.sol_in.num_of_elements):
            YM = self.sol_in.get_element_property('E', n) # Young's modulus
            
            L = self.sol_in.get_element_property('L', n)

            T = self.get_T(n)

            C = YM/L *( np.array([-1,0,1,0]) @ T )
            element_conn = self.get_element_conn(n)
            d_el = self.displacements[element_conn]
            stress[n] = C @ d_el
            
        return stress

    def get_Kel_global(self, element_index):
        """ Implementation of get_Kel_global method for BaseSolver class"""
        
        Kel_local = self.get_Kel_local(element_index)
        T = self.get_T(element_index)
        return T.T @ Kel_local @ T

    def get_Kel_local(self, element_index):
        """ 
        returns the element stiffness matrix in local coordinates 
        for 2D truss structural elements.
        
        """
        coefficient = self.element_stiffness(element_index)
        Kel_local = coefficient \
                    * np.array([
                                ( 1, 0,-1, 0),
                                ( 0, 0, 0, 0),
                                (-1, 0, 1, 0),
                                ( 0, 0, 0, 0)
                            ])
        return Kel_local 

    def get_T(self, element_index):
        """ Get the transformation matrix T of given element. """
        # Calculate cos and sin value for given element
        directional_cosine = self.sol_in.get_element_property('dcos', 
                                                               element_index)
        C = directional_cosine[0]
        S = directional_cosine[1]
        T = np.array([
                    ( C, S, 0, 0),
                    (-S, C, 0, 0),
                    ( 0, 0, C, S),
                    ( 0, 0,-S, C)])

        return T
    
    def element_stiffness(self, element_index):
        """ return the stiffness of a single truss element. """
        area = self.sol_in.get_element_property('A', element_index)
        youngs_modulus = self.sol_in.get_element_property('E', element_index)
        element_length = self.sol_in.get_element_property('L', element_index)
        return area * youngs_modulus / element_length
    
    def interpolate_shape(self, mag_factor, sub_element_num=100):
        """ 
        Note: Since truss uses linear shape functions, no interpolation is used.
        The method returns the deformed nodal positon of elements to make 
        output utility usage consistent.

        """

        new_nodal_data = np.zeros(self.sol_in.nodal_data.shape)
        for i in range(0,self.sol_in.num_of_nodes):
            new_nodal_data[i,0] = self.sol_in.nodal_data[i,0]
            new_nodal_data[i,1] = self.sol_in.nodal_data[i,1] \
                + mag_factor * self.displacements[i*self.sol_in.dof]
            new_nodal_data[i,2] = self.sol_in.nodal_data[i,2] \
                + mag_factor * self.displacements[i*self.sol_in.dof + 1]
        
        interpolated_nodal_data = np.zeros((self.sol_in.num_of_elements, 
                                           2, 2))
        # need to offset by the new nodal position
        for element_index in range(self.sol_in.num_of_elements):
            nodal_indices = self.sol_in.get_element_property('node_ind', 
                                                        element_index)
            interpolated_nodal_data[element_index,0,:] = \
                new_nodal_data[nodal_indices[0], 1:3]
            interpolated_nodal_data[element_index,1,:] = \
                new_nodal_data[nodal_indices[1], 1:3]
            
        return interpolated_nodal_data


class FrameSolver2D(TrussSolver2D):
    """ Solver class for 2D frame structures. """
    
    def __init__(self, SolverInput):
        BaseSolver.__init__(self, SolverInput)

    def fill_Kel_local(self, input_matrix, conn):
        element_dof = self.sol_in.dof * self.sol_in.node_per_element
        Kel_local_component = np.zeros((element_dof, element_dof))
        for i in range(len(input_matrix)):
            for j in range(len(input_matrix)):
                Kel_local_component[conn[i], conn[j]] = input_matrix[i,j]
        return Kel_local_component


    def Kel_local_axial(self, element_index):
        stiffness = self.element_stiffness(element_index)
        Kel_axial_1D = stiffness * np.array([(1, -1), (-1, 1)])
        axial_conn = [0, self.sol_in.dof]
        return self.fill_Kel_local(Kel_axial_1D, axial_conn)
    
    def Kel_local_bending(self, element_index):
        E = self.sol_in.get_element_property('E', element_index)
        I = self.sol_in.get_element_property('I', element_index)
        L = self.sol_in.get_element_property('L', element_index)
        Kel_bending_primitive =  E*I/L**3 * np.array([
                                    ( 12  , 6*L   ,-12  , 6*L   ),
                                    (  6*L, 4*L**2, -6*L, 2*L**2),
                                    (-12  ,-6*L   , 12  ,-6*L   ),
                                    (  6*L, 2*L**2, -6*L, 4*L**2)
                                   ])
        bending_conn = [1, 2, 1 + self.sol_in.dof, 2 + self.sol_in.dof]
        return self.fill_Kel_local(Kel_bending_primitive, bending_conn)
    
    def get_Kel_local(self, element_index):
        return self.Kel_local_axial(element_index) \
                + self.Kel_local_bending(element_index)

    def get_T(self, element_index):
        """ Get the transformation matrix T of given element. """
        # Calculate cos and sin value for given element
        directional_cosine = self.sol_in.get_element_property('dcos', 
                                                                element_index)
        C = directional_cosine[0]
        S = directional_cosine[1]
        T = np.array([
                    ( C, S, 0, 0, 0, 0),
                    (-S, C, 0, 0, 0, 0),
                    ( 0, 0, 1, 0, 0, 0),
                    ( 0, 0, 0, C, S, 0),
                    ( 0, 0, 0,-S, C, 0),
                    ( 0, 0, 0, 0, 0, 1)])

        return T
    
    def interpolate_shape(self, mag_factor,sub_element_num=100):
        interpolated_nodal_data = np.zeros((self.sol_in.num_of_elements, 
                                           sub_element_num+1, 2))
        

        # need to offset by the new nodal position
        for element_index in range(self.sol_in.num_of_elements):
            interpolated_nodal_data[element_index,:,:] = \
                self.get_interpolate_points(element_index, mag_factor, 
                                             sub_element_num)
            
        return interpolated_nodal_data

    def get_interpolate_points(self, element_index, mag_factor, sub_element_num):
        # get local displacement of the two nodes
        node_ind = self.sol_in.get_element_property('node_ind', 
                                                    element_index)
        dof = self.sol_in.dof
        disp_ind = [list(range(node_ind[0] * dof, (node_ind[0] + 1) * dof)), 
                    list(range(node_ind[1] * dof, (node_ind[1] + 1) * dof))]
        
        element_disp = np.zeros(dof * 2)
        element_disp[0:dof] = self.displacements[disp_ind[0]]
        element_disp[dof:len(element_disp)] = self.displacements[disp_ind[1]]

        
        local_disp = self.get_T(element_index) @ element_disp
        L = self.sol_in.get_element_property('L', element_index)
        # position for sub element points after deflection
        interpolation_pts = np.zeros((sub_element_num+1, 2))
        interpolation_pts[:,0] = np.linspace(0, L, sub_element_num+1)
        

        axial_nodal_disp = local_disp[[0,3]]
        beam_nodal_disp = local_disp[[1,2,4,5]]
        
        directional_cosine = self.sol_in.get_element_property('dcos', element_index)
        C = directional_cosine[0]
        S = directional_cosine[1]

        transform_2d = np.array([(C, S), (-S, C)])
        

        for i, x in enumerate(interpolation_pts[:,0]):
            
            interpolation_pts[i,0] += self.get_interpolated_axial_disp(
                        x, L, element_index, axial_nodal_disp) * mag_factor
            interpolation_pts[i,1] = self.get_beam_deflection(
                        x, L, element_index, beam_nodal_disp) * mag_factor
            interpolation_pts[i,:] = transform_2d.T @ interpolation_pts[i,0:2]\
                        + self.get_first_nodal_position(element_index)
        return interpolation_pts

    
    def get_first_nodal_position(self, element_index):
        nodal_indices = self.sol_in.get_element_property('node_ind', 
                                                          element_index)
        return self.sol_in.nodal_data[nodal_indices[0],1:3].copy()
        
    def get_beam_deflection(self, x, L, element_index, beam_nodal_disp):
        
        # Hermite cubic interpolation function
        N1 = 1/L**3 * (2 * x**3 - 3 * x**2 * L + L**3)
        N2 = 1/L**3 * (x**3 * L - 2 * x**2 * L**2 + x * L**3)
        N3 = 1/L**3 * (-2 * x**3 + 3 * x**2 * L)
        N4 = 1/L**3 * (x**3 * L - x**2 * L**2)
        
        N = np.array([N1, N2, N3, N4])

        return N @ beam_nodal_disp
    
    def get_interpolated_axial_disp(self, x, L, 
                                    lement_index, axial_nodal_disp):
        
        N = np.array([1 - x/L, x/L])  # linear axial interpolation function
        return N @ axial_nodal_disp


class TriangularElementSolver(BaseSolver):
    
    def __init__(self, SolverInput):
        BaseSolver.__init__(self, SolverInput)
        self.stress, self.strain = self.get_stress_and_strain()

    def stress_strain_matrix(self, element_index, is_plane_strain=False):
        nu = self.sol_in.get_element_property('nu', element_index)
        E = self.sol_in.get_element_property('E', element_index)
        if not is_plane_strain:
            D = np.array([(1,nu,0),(nu,1,0),(0,0,(1-nu)/2)]) * E/(1-nu**2)
        else:
            D = np.array([(1-nu,nu,0),(nu,1-nu,0),(0,0,(1-2*nu)/2)]) \
                * E/(1+nu)/(1-2*nu)
        return D

    def strain_displacement_matrix(self, element_index):
        Coord = self.sol_in.get_element_property('nodal_coord', element_index)
        dydeta = -Coord[0,1] + Coord[2,1]  # -y1 + y3
        dydxi = -Coord[0,1] + Coord[1,1]   # -y1 + y2
        dxdeta = -Coord[0,0] + Coord[2,0]  # -x1 + x3
        dxdxi = -Coord[0,0] + Coord[1,0]   # -x1 + x2
        
        B11 =  dydeta*(-1) - dydxi*(-1)  # (-y1 + y3)*(-1) - (-y1 + y2) * (-1) 
        B13 =  dydeta*( 1) - dydxi*( 0)  # (-y1 + y3)*( 1) - (-y1 + y2) * ( 0)
        B15 =  dydeta*( 0) - dydxi*( 1)  # (-y1 + y3)*( 0) - (-y1 + y2) * ( 1)
        
        B22 = -dxdeta*(-1) + dxdxi*(-1)  # -(-x1 + x3)*(-1) + (-x1 + x2) * (-1) 
        B24 = -dxdeta*( 1) + dxdxi*( 0)  # -(-x1 + x3)*( 1) + (-x1 + x2) * ( 0) 
        B26 = -dxdeta*( 0) + dxdxi*( 1)  # -(-x1 + x3)*( 0) + (-x1 + x2) * ( 1) 

        BJ = np.array([(B11,0,B13,0,B15,0),(0,B22,0,B24,0,B26),
                       (B22,B11,B24,B13,B26,B15)])
        Jdet = np.linalg.det( np.array([(dxdxi,dydxi),(dxdeta,dydeta)]))

        B = BJ/Jdet
        return B

    def get_Jacobian_det(self, element_index):
        Coord = self.sol_in.get_element_property('nodal_coord', element_index)
        dydeta = -Coord[0,1] + Coord[2,1] 
        dydxi = -Coord[0,1] + Coord[1,1] 
        dxdeta = -Coord[0,0] + Coord[2,0]
        dxdxi = -Coord[0,0] + Coord[1,0]
        Jdet = np.linalg.det( np.array([(dxdxi,dydxi),(dxdeta,dydeta)]) )
        return Jdet
    
    def get_Kel_global(self, element_index):
        B = self.strain_displacement_matrix(element_index)
        D = self.stress_strain_matrix(element_index)
        Jdet = self.get_Jacobian_det(element_index)
        thickness = self.sol_in.get_element_property('t', element_index)
        Kel = thickness * Jdet * 0.5 * (B.T @ D @ B) 
        return Kel

    def get_element_strain(self, element_index, nodal_displacement):
        B = self.strain_displacement_matrix(element_index)
        return B @ nodal_displacement
    
    def get_element_stress(self, element_index, element_strain):
        D = self.stress_strain_matrix(element_index)
        return D @ element_strain   
    
    def get_stress_and_strain(self):
        num_elements = self.sol_in.num_of_elements
        strain = np.zeros((num_elements, 3))
        stress = np.zeros((num_elements, 3))
        for el in range(num_elements):
            el_disp = np.asarray(self.displacements[self.global_conn[el]])
            strain[el, :] = self.get_element_strain(el, el_disp)
            Coord = self.sol_in.get_element_property('nodal_coord', 
                                                      el)
            stress[el,:] = self.get_element_stress(el, strain[el, :])
        
        return stress, strain
    