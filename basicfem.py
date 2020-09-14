""" basicfem.py is the main script to execute the Basicfem solver. """

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractclassmethod
from lib.solvers import *
from lib.output_utility import *
from lib.solver_input import *

def main():
    """ 
    Main function to read in command line input, processing directories and
    run respective solvers. The script takes two argument -- an input directory
    name and an output directory name. The input directory should locate in the 
    local directory and needs to contain necessary txt files for basicfem 
    solvers. It can also be an xlsx file that contains all the required input. 
    The output directory will be generated if not exists, or being overwritten.
    
    """

    if len(sys.argv) != 3:
        print("usage: python basicfem.py <input_directory> <output_directory>")
        print("       <input_directory> can also be a single xlsx file",
              "with all required input.")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    input_dir = input_dir.lstrip('./')
    if not os.path.exists(input_dir):
        print("Input directory or file does not exist.")
        sys.exit(1)
    
    if os.path.isfile(input_dir):
        param = read_param(input_dir)
    else:
        param = read_param('./' +sys.argv[1] + '/param.txt')

    cwd = os.getcwd()
    output_path = cwd + '/' + sys.argv[2]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    if param is None:
        param = {}
        problem_type, scale_factor = set_param_guide()
    else:
        problem_type = param['problem_type']
        scale_factor = float(param['deformation_scale_factor'])

    if problem_type.lower() == "truss":
        input_ = TrussInput2D(input_dir)
        result = TrussSolver2D(input_)
        save_data(result.stress,'element_stresses.txt', dir_name=sys.argv[2])
        save_data(result.displacements, 'nodal_displacements.txt', 
                  dir_name=sys.argv[2])
        plot_deformed_shape_1D(result, scale_factor, output_path)

    elif problem_type.lower() == "frame":
        input_ = TrussInput2D(input_dir)
        result = FrameSolver2D(input_)
        save_data(result.displacements, 'nodal_displacements.txt',
                  dir_name=sys.argv[2])
        plot_deformed_shape_1D(result, scale_factor, output_path)

    elif problem_type.lower() == '2d':
        input_ = TriangularElementInput(input_dir)
        result = TriangularElementSolver(input_)
        plot_deformation_shape_2D(input_.nodal_data, input_.element_data, 
        result.displacements, scale_factor, output_path)
        if 'contour_over_deformed_mesh' in param:
            plot_on_deformed_flag = str(param['contour_over_deformed_mesh']) \
                                        .lower() == 'true'
            plot_contour_batch(result, scale_factor, output_path, 
                        plot_on_deformed=plot_on_deformed_flag)
        else:
            plot_contour_batch(result, scale_factor, output_path)
        
        save_data(result.displacements, 'nodal_displacements', 
                  dir_name=sys.argv[2])
        save_data(result.stress, 'element_stresses', dir_name=sys.argv[2])
    
    print("Solver process completed.")

def read_param(param_filename):
    """ Read the param file to set the solver related parameters.

    The param file will either be an param.txt file inside the input directory,
    or the Param page inside the input xlsx file.

    Parameters
    ----------
    param_filename : str
        The xlsx file name, or the processed path to the param.txt file.
    
    Returns
    -------
    dict
        The dictionary contains solver related paramters.

    """

    if not os.path.isfile(param_filename):
        return None
    
    if param_filename.endswith('.xls') or param_filename.endswith('.xlsx'):
        from pandas import read_excel
        arr = read_excel(param_filename, sheet_name = 'Param').values
        param = {arr[i,0] : arr[i,1] for i in range(len(arr))}
    else:
        f = open(param_filename, 'r')
        param = {}
        for line in f:
            line = line.split('#', 1)[0]
            line = line.rstrip()
            if line != '':
                k, v = line.strip().split('=')
                param[k.strip()] = v.strip()
        f.close()
    return param

def set_param_guide():
    """ Command line prompt to guide the users setting up parameters.
    
    In the case when param file does not exist, this function will guide the
    users to provide the necessary paramters for solver. Note that this guide
    will keep most of the parameters default, only set those that is needed to
    be assigned for solver to run.
    
    Returns
    -------
    tuple
        Parameters necessary for the solver.
    
    """

    print("param file/page does not exists. Please specify:")
    problem_type = input("Problem type? (`truss`, `frame`, or `2d`): ")
    while problem_type not in ['truss','frame','2d']:
        print('Invalid problem type.')
        problem_type = input("Problem type? (`truss`, `frame`, or `2d`): ")
    scale_factor = input("Deformation scale factor? (default = 1) ")
    try:
        scale_factor = float(scale_factor)
    except:
        scale_factor = 1.0
        print('No valid input. Deformation scale factor = 1 will be used.')
    
    return problem_type, scale_factor 
    
def plot_contour_batch(fem_result, scale_factor, output_path, 
                       plot_on_deformed=True):
    """ Generate all the contour plot available. Only used for 2d elements. 
    
    Parameters
    ----------
    fem_result: BaseSolver
        A Solver object contains the analysis result.
    scale_factor : float
        The deformation scale factor assigned in the param.
    output_path : str 
        The path to the output directory assigned by the user.
    plot_on_deformed : bool
        Flag that determines if the contour is plotted on the deformed
        structures. Default is True.

    """
    
    if plot_on_deformed:
        plot_contour(fem_result, "stress", 'xx', 'sigma_xx', scale_factor, 
                output_path)
        plot_contour(fem_result, "stress", 'yy', 'sigma_yy', scale_factor, 
                        output_path)
        plot_contour(fem_result, "stress", 'xy', 'sigma_xy', scale_factor, 
                        output_path)
        plot_contour(fem_result, "strain", 'xx', 'strain_xx', scale_factor, 
                        output_path)
        plot_contour(fem_result, "strain", 'yy', 'strain_yy', scale_factor, 
                        output_path)
        plot_contour(fem_result, "strain", 'xy', 'strain_xy', scale_factor, 
                        output_path)
    else:
        plot_contour(fem_result, "stress", 'xx', 'sigma_xx', scale_factor, 
                output_path, False)
        plot_contour(fem_result, "stress", 'yy', 'sigma_yy', scale_factor, 
                        output_path, False)
        plot_contour(fem_result, "stress", 'xy', 'sigma_xy', scale_factor, 
                        output_path, False)
        plot_contour(fem_result, "strain", 'xx', 'strain_xx', scale_factor, 
                        output_path, False)
        plot_contour(fem_result, "strain", 'yy', 'strain_yy', scale_factor, 
                        output_path, False)
        plot_contour(fem_result, "strain", 'xy', 'strain_xy', scale_factor, 
                        output_path, False)

if __name__ == "__main__":
    main()