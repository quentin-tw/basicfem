""" Main entry of the program """

import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractclassmethod
from Solver import *
from output_utility import *
from SolverInput import *

def main():
    if len(sys.argv) != 3:
        print("usage: python basicfem.py <filename> <param filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print("Input file does not exist.")
        sys.exit(1)

    param = read_param(sys.argv[2])
    
    problem_type = param['problem_type']
    mag_factor = float(param['deformation_magnification_factor'])
    if problem_type.lower() == "truss":
        input_ = TrussInput2D(filename)
        result = TrussSolver2D(input_)
        save_data(result.stress,'truss_element_stresses.txt')
    elif problem_type.lower() == "frame":
        input_ = TrussInput2D(filename)
        result = FrameSolver2D(input_)
    
    save_data(result.displacements,param['displacements_file_name'])
    deformedShapePlot(result, mag_factor)


def read_param(param_filename):
    """ read the param file given by the user. """


    f = open(param_filename, 'r')
    param = {}
    for line in f:
        if line.strip() != '':
            k, v = line.strip().split('=')
            param[k.strip()] = v.strip()
    f.close()
    return param

if __name__ == "__main__":
    main()