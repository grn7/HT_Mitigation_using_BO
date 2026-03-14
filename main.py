# main file 
# instatiate objects, call functions and put the whole thing together

import numpy as np
import matplotlib.pyplot as plt
import torch
import re # To parse the function string for operators
from init import EHWP_Grid
from function import objective_function
from BO import BO

def get_manhattan_path(start, end):
    return [start, (start[0], end[1]), end]

if __name__ == "__main__": # this runs only when u run main
    n = int(input("Enter the dimension of the EHWP grid: "))
    i = int(input("Enter the number of infected RPUs: "))
    func_string = input("Enter the function to run (e.g., (a*((b+c)*(d/e))): ")
    n_iter = int(input("Enter the number of BO iterations: "))

    # Parse the function string to count operators to determine how many RPUs we need
    operators = re.findall(r'[\+\-\*\/]', func_string)
    k = len(operators)
    if k == 0:
        k = 1 # Fallback just in case they just enter a single variable
    print(f"\nDetected {k} operations. The BO will search for a path connecting {k} RPUs.")

    EHWP = EHWP_Grid(n,i) # make object

    # initialize BO
    print("\n Starting Bayesian Optimization")
    bo_optimizer = BO(n, k, EHWP.grid, objective_function) 

    # run the loop and get tensors of all tried paths and their costs
    train_X, train_Y = bo_optimizer.run_optimization(n_iter=n_iter, n_init=10)

    # extract the best result
    best_idx = torch.argmin(train_Y).item()
    best_cost = train_Y[best_idx].item()

    # get corresponding path and convert to an integer array
    best_path_cont = train_X[best_idx].detach().numpy()
    best_path_rounded = np.round(best_path_cont).astype(int)

    print("\nOptimization complete")
    print(f"Best cost found: {best_cost}")
    print(f"Best path coordinates as a flat 1D array: {best_path_rounded}")

    # plotting the convergence plot
    plt.figure(figsize=(12,8)) # creates new blank window
    costs = train_Y.squeeze().detach().numpy() # squeeze removes unnecessary outer bracket
    
    # Calculate moving average
    window_size = 5
    # np.convolve slides a window of size 5 across the costs and averages them
    # 'valid' mode means it only calculates where the window is full (starts at 5th element)
    moving_avg = np.convolve(costs, np.ones(window_size)/window_size, mode='valid')
    
    # We need an x-axis offset because the moving average array is shorter than the costs array
    # It starts at index 4 (the 5th evaluation)
    x_moving_avg = np.arange(window_size - 1, len(costs))

    plt.plot(costs, alpha = 0.3, label='Sampled cost', color='blue')
    plt.plot(x_moving_avg, moving_avg, color='red', linewidth=2, label=f'{window_size}-Iter Moving Average')
    
    # both are y axis
    # x axis is the number of evaluations
    # when u give matplotlib only a single list of numbers it automatically assumes x axis is just the index of each number
    plt.xlabel('Evaluations (Init + Iterations)')
    plt.ylabel('Cost (Wire Length + Penalties)')
    plt.title('Bayesian Optimization Convergence')
    plt.legend()
    plt.grid(True)
    plt.show(block=False) # Non Blocking so the next plot can draw

    # overlay path on the grid
    EHWP.display_grid()
    ax = plt.gca() # get current active axes ( the grid we just drew)

    # reshape flat path into 2D 
    reshaped_path = best_path_rounded.reshape(k, 2) 

    # draw path on grid using Manhattan lines
    full_visual_path = []
    for step in range(k - 1):
        segment = get_manhattan_path(reshaped_path[step], reshaped_path[step+1])
        if step > 0:
            segment = segment[1:] # Prevent overlapping coordinates
        full_visual_path.extend(segment)
        
    full_visual_path = np.array(full_visual_path)

    y_coords_line = full_visual_path[:, 0] 
    x_coords_line = full_visual_path[:, 1] 

    # Plot the L-shaped wires
    ax.plot(x_coords_line, y_coords_line, color='blue', linewidth=4, label='Optimal Route (Wires)')
    
    # Plot the chosen RPUs as dots
    ax.scatter(reshaped_path[:, 1], reshaped_path[:, 0], color='cyan', s=150, zorder=5, edgecolors='black', label='Assigned RPUs')
    
    # Number the RPUs to show the sequence
    for step_idx in range(k):
        # Place the text at the (x, y) coordinate, centered in the circle
        ax.annotate(str(step_idx + 1), 
                    (reshaped_path[step_idx, 1], reshaped_path[step_idx, 0]), 
                    color='black', weight='bold', fontsize=10, 
                    ha='center', va='center', zorder=6)

    ax.legend(loc="upper right") # puts the legend in upper right corner 

    # keep all windows open until user closes them
    print("\n Close plot windows to exit the program")
    plt.show()