# main file 
# instatiate objects, call functions and put the whole thing together

from init import EHWP_Grid

if __name__ == "__main__": # this runs only when u run main
    n = int(input("Enter the dimension of the EHWP grid: "))
    i = int(input("Enter the number of infected RPUs: "))
    grid = EHWP_Grid(n,i) # make object
    grid.display_grid() # uses grid created during object initialisation