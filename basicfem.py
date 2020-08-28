""" Main entry of the program """

import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from Solver import *
from output_util import *
from SolverInput import *

def main():
    if len(sys.argv) != 2:
        print("usage: python basicfem.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print("Input file does not exist.")
        sys.exit(1)
    
    input_ = TrussInput2D(filename)
    result = TrussSolver2D(input_)
    save_data(result.stress,'sigma.txt')
    save_data(result.displacements,'displacement.txt')
    shapePlot(result.displacements, input_, 1)


if __name__ == "__main__":
    main()