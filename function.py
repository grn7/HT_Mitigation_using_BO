# this file computes the value of the function to minimise using manhattan distance and penalties 
import numpy as np
INFECTED_RPU_PENALTY_WEIGHT = 100

def manhattan_dist(p1,p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def objective_function(path, grid, k, n): # Now takes k (number of RPUs) as well
    # path consists of coordinates it is passing through
    # path is a single row , has 2*k elements
    reshaped_path = path.reshape(k, 2) 
    wire_length = 0
    visited = set() # Keep track of used RPUs to prevent overlapping mapping
    
    for i in range(k):
        coord = tuple(reshaped_path[i])

        # Uniqueness penalty.you cannot map two different operations to the same RPU
        if coord in visited:
            wire_length += 5000 
        visited.add(coord)

        # Penalty if it is infected
        if grid[coord[0], coord[1]] == 1:
            wire_length += n * n * INFECTED_RPU_PENALTY_WEIGHT
        else:
            # soft penalty: Check adjacent cells to create a gradient for the GP
            adjacent_infected = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = coord[0] + dr, coord[1] + dc
                if 0 <= nr < n and 0 <= nc < n:
                    if grid[nr, nc] == 1:
                        adjacent_infected += 1
            wire_length += adjacent_infected * (n * 5)

        # Wire length is summation of manhattan dist bw waypoints
        if i < k - 1: # Calculate distance to the NEXT point in the function graph
            next_coord = tuple(reshaped_path[i+1])
            dist = manhattan_dist(coord, next_coord)
            wire_length += dist
        
    return wire_length