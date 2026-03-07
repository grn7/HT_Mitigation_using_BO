# this file computes the value of the function to minimise using manhattan distance and penalties 
import numpy as np
INFECTED_RPU_PENALTY_WEIGHT = 100

def manhattan_dist(p1,p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def objective_function(path, grid, n): 
    # path consists of coordinates it is passing through
    # path is a single row , has 2n elements
    # grid is n*n
    reshaped_path = path.reshape(n,2) # convert to n rows, 2 columns
    wire_length = 0
    
    for i in range(n-1):
        # wire length is summation of manhattan dist bw waypoints
        # let us say we have an infected RPU at 2,2
        # here we are assuming that we can safely route from 2,1 to 2,3 without using the RPU at 2,2
        dist = manhattan_dist(tuple(reshaped_path[i]),tuple(reshaped_path[i+1]))
        coord = reshaped_path[i]

        if( dist == 0):
            # if the same coordinate repeats it means that the path has ended so we break
            # before we break check if the final node has an HT 
            wire_length += (grid[coord[0],coord[1]] == 1)*n*n*INFECTED_RPU_PENALTY_WEIGHT
            break

        wire_length += dist

        if( i == n-2): # do this , otherwise the last one's penalty will never show up 
            next_coord = reshaped_path[i+1]
            penalty = (grid[next_coord[0],next_coord[1]] == 1)*n*n*INFECTED_RPU_PENALTY_WEIGHT + (grid[coord[0],coord[1]] == 1)*n*n*INFECTED_RPU_PENALTY_WEIGHT
            wire_length += penalty
        else:
            wire_length += (grid[coord[0],coord[1]] == 1)*n*n*INFECTED_RPU_PENALTY_WEIGHT # penalty if it is infected
        
    return wire_length
    

    

