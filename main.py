# main file 
# instatiate objects, call functions and put the whole thing together

import numpy as np
import matplotlib.pyplot as plt
import torch
import re # To parse the function string for operators
from init import EHWP_Grid
from function import objective_function
from BO import BO
from GA import GA

def get_manhattan_path(start, end):
    return [start, (start[0], end[1]), end]

if __name__ == "__main__": # this runs only when u run main
    n = int(input("Enter the dimension of the EHWP grid: "))
    i = int(input("Enter the number of infected RPUs: "))
    func_string = input("Enter the function to run (e.g., (a*((b+c)*(d/e))): ")
    n_iter = int(input("Enter the number of iterations( same for both BO and GA): "))
    mutation_rate = float(input("Enter the mutation rate for GA: "))

    # Parse the function string to count operators to determine how many RPUs we need
    operators = re.findall(r'[\+\-\*\/]', func_string)
    k = len(operators)
    if k == 0:
        k = 1 # Fallback just in case they just enter a single variable
    print(f"\nDetected {k} operations. The BO will search for a path connecting {k} RPUs.")

    EHWP = EHWP_Grid(n,i) # make object

    # initialize BO
    print("\nStarting Bayesian Optimization")
    bo_optimizer = BO(n, k, EHWP.grid, objective_function) 

    # run the loop and get tensors of all tried paths and their costs
    train_X, train_Y = bo_optimizer.run_optimization(n_iter=n_iter, n_init=10)

    # extract the best result
    best_idx = torch.argmin(train_Y).item()
    bo_best_cost = train_Y[best_idx].item()

    # get corresponding path and convert to an integer array
    best_path_cont = train_X[best_idx].detach().numpy()
    bo_best_path_rounded = np.round(best_path_cont).astype(int)

    print("\nBO complete")
    print(f"Best BO cost found: {bo_best_cost}")
    print(f"Best BO path coordinates as a flat 1D array: {bo_best_path_rounded}")

    # start GA
    print("\nStarting GA")
    ga_optimizer = GA(n, k, EHWP.grid, objective_function, pop_size=10, mutation_rate=mutation_rate)
    ga_all_costs, ga_best_path, ga_best_cost = ga_optimizer.run_optimization(n_iter=n_iter)

    print("\nGA Optimization complete")
    print(f"Best GA cost found: {ga_best_cost}")
    print(f"Best GA path: {ga_best_path}")

    # plotting the convergence plot
    plt.figure(figsize=(12,8)) # creates new blank window
    bo_costs = train_Y.squeeze().detach().numpy() # squeeze removes unnecessary outer bracket
    
    # Calculate moving average
    window_size = 5
    # np.convolve slides a window of size 5 across the costs and averages them
    # 'valid' mode means it only calculates where the window is full (starts at 5th element)
    bo_moving_avg = np.convolve(bo_costs, np.ones(window_size)/window_size, mode='valid')
    ga_moving_avg = np.convolve(ga_all_costs, np.ones(window_size)/window_size, mode='valid')
    
    # We need an x-axis offset because the moving average array is shorter than the costs array
    # It starts at index 4 (the 5th evaluation)
    x_moving_avg = np.arange(window_size - 1, len(bo_costs))

    plt.plot(bo_costs, alpha = 0.3, label='BO Sampled cost', color='blue')
    plt.plot(x_moving_avg, bo_moving_avg, color='darkblue', linewidth=2, label=f'BO {window_size}-Iter Moving Average')

    plt.plot(ga_all_costs, alpha=0.3, label='GA Sampled Cost', color='red')
    plt.plot(x_moving_avg, ga_moving_avg, color='darkred', linewidth=2, label=f'GA {window_size}-Iter Moving Average')
    
    # both are y axis
    # x axis is the number of evaluations
    # when u give matplotlib only a single list of numbers it automatically assumes x axis is just the index of each number
    plt.xlabel('Evaluations (Init + Iterations)')
    plt.ylabel('Cost (Wire Length + Penalties)')
    plt.title('Algorithm Convergence Comparison (BO vs GA)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False) # Non Blocking so the next plot can draw


    # --- CHANGED: Separate the grid plots into two distinct windows ---

    # 1. Plot BO Result Window
    EHWP.display_grid() 
    ax_bo = plt.gca() 

    # reshape flat path into 2D 
    bo_reshaped_path = bo_best_path_rounded.reshape(k, 2) 

    # draw path on grid using Manhattan lines
    bo_full_visual_path = []
    for step in range(k - 1):
        segment = get_manhattan_path(bo_reshaped_path[step], bo_reshaped_path[step+1])
        if step > 0:
            segment = segment[1:] # Prevent overlapping coordinates
        bo_full_visual_path.extend(segment)
        
    bo_full_visual_path = np.array(bo_full_visual_path)

    # Plot the L-shaped wires and RPUs for BO
    ax_bo.plot(bo_full_visual_path[:, 1], bo_full_visual_path[:, 0], color='blue', linewidth=4, label='BO Route')
    ax_bo.scatter(bo_reshaped_path[:, 1], bo_reshaped_path[:, 0], color='cyan', s=150, zorder=5, edgecolors='black', label='BO RPUs')

    # Number the RPUs to show the sequence for BO
    for step_idx in range(k):
        ax_bo.annotate(str(step_idx + 1), 
                    (bo_reshaped_path[step_idx, 1], bo_reshaped_path[step_idx, 0]), 
                    color='black', weight='bold', fontsize=10, 
                    ha='center', va='center', zorder=6)
        
    ax_bo.legend(loc="upper right", title="BO Results") 


    # 2. Plot GA Result Window
    EHWP.display_grid() 
    ax_ga = plt.gca() 

    # reshape flat path into 2D 
    ga_reshaped_path = ga_best_path.reshape(k,2)

    ga_full_visual_path = []
    for step in range(k-1):
        segment = get_manhattan_path(ga_reshaped_path[step], ga_reshaped_path[step+1])
        if step > 0:
            segment = segment[1:]
        ga_full_visual_path.extend(segment)
    ga_full_visual_path = np.array(ga_full_visual_path) 

    # Plot the L-shaped wires and RPUs for GA
    ax_ga.plot(ga_full_visual_path[:, 1], ga_full_visual_path[:, 0], color='red', linewidth=4, linestyle='--', label='GA Route')
    ax_ga.scatter(ga_reshaped_path[:, 1], ga_reshaped_path[:, 0], color='yellow', s=150, zorder=5, edgecolors='black', label='GA RPUs')

    # Number the RPUs to show the sequence for GA
    for step_idx in range(k):
        ax_ga.annotate(str(step_idx + 1), 
                    (ga_reshaped_path[step_idx, 1], ga_reshaped_path[step_idx, 0]), 
                    color='black', weight='bold', fontsize=10, 
                    ha='center', va='center', zorder=6)

    ax_ga.legend(loc="upper right", title="GA Results") 

    # keep all windows open until user closes them
    print("\n Close plot windows to exit the program")
    plt.show()