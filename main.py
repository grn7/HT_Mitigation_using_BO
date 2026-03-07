# main file 
# instatiate objects, call functions and put the whole thing together

import numpy as np
from init import EHWP_Grid
from function import objective_function

if __name__ == "__main__": # this runs only when u run main
    n = int(input("Enter the dimension of the EHWP grid: "))
    i = int(input("Enter the number of infected RPUs: "))
    EHWP = EHWP_Grid(n,i) # make object
    EHWP.display_grid() # uses grid created during object initialisation

    # testing the objective function
    path = np.array([0,1,1,1,2,1]) # path is (0,1);(1,1);(2,1) , test on 3*3 
    cost = objective_function(path,EHWP.grid,n)
    print(f"Final wire length: {cost}")

    input("Press Enter to exit and close the grid") # keeps the grid alive until u r looking at it 