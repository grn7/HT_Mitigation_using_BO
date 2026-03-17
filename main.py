# main.py - Extended version with GA, BO, and SA 
# Instantiate objects, call functions and put the whole thing together

import numpy as np
import matplotlib.pyplot as plt
import torch
import re  # To parse the function string for operators
from init import EHWP_Grid
from function import objective_function
from BO import BO
from GA import GA
from SA import SA  


def get_manhattan_path(start, end):
    """Generate L-shaped path between two points"""
    return [start, (start[0], end[1]), end]


def plot_convergence_comparison(bo_costs_raw, ga_costs_raw, sa_costs_raw, window_size=5):
    """
    Plot convergence comparison for all three algorithms
    
    Args:
        bo_costs_raw: raw sampled costs from BO (tensor or numpy)
        ga_costs_raw: raw sampled costs from GA (list)
        sa_costs_raw: raw sampled costs from SA (list)
        window_size: moving average window
    """
    plt.figure(figsize=(14, 8))
    
    # Convert BO tensor to numpy if needed
    if torch.is_tensor(bo_costs_raw):
        bo_costs = bo_costs_raw.squeeze().detach().numpy()
    else:
        bo_costs = np.array(bo_costs_raw)
    
    ga_costs = np.array(ga_costs_raw)
    sa_costs = np.array(sa_costs_raw)
    
    # Ensure all arrays have the same length for fair comparison
    # They should all have length = n_init + n_iter (typically 100)
    min_length = min(len(bo_costs), len(ga_costs), len(sa_costs))
    print(f"Plotting with {min_length} evaluations (min across algorithms)")
    
    bo_costs = bo_costs[:min_length]
    ga_costs = ga_costs[:min_length]
    sa_costs = sa_costs[:min_length]
    
    # Calculate moving averages
    bo_moving_avg = np.convolve(bo_costs, np.ones(window_size)/window_size, mode='valid')
    ga_moving_avg = np.convolve(ga_costs, np.ones(window_size)/window_size, mode='valid')
    sa_moving_avg = np.convolve(sa_costs, np.ones(window_size)/window_size, mode='valid')
    
    # X-axis for moving averages (starts at window_size-1)
    x_moving_avg = np.arange(window_size - 1, min_length)
    
    # Ensure moving averages have the same length as x_moving_avg
    # They should, but let's be safe
    bo_moving_avg = bo_moving_avg[:len(x_moving_avg)]
    ga_moving_avg = ga_moving_avg[:len(x_moving_avg)]
    sa_moving_avg = sa_moving_avg[:len(x_moving_avg)]
    
    # Plot raw sampled costs (semi-transparent)
    plt.plot(range(min_length), bo_costs, alpha=0.2, color='blue', label='BO Sampled Cost')
    plt.plot(range(min_length), ga_costs, alpha=0.2, color='red', label='GA Sampled Cost')
    plt.plot(range(min_length), sa_costs, alpha=0.2, color='yellow', label='SA Sampled Cost')
    
    # Plot moving averages (solid lines)
    plt.plot(x_moving_avg, bo_moving_avg, color='darkblue', linewidth=2.5, 
             label=f'BO {window_size}-Iter Moving Avg')
    plt.plot(x_moving_avg, ga_moving_avg, color='darkred', linewidth=2.5, 
             label=f'GA {window_size}-Iter Moving Avg')
    plt.plot(x_moving_avg, sa_moving_avg, color='darkgreen', linewidth=2.5, 
             label=f'SA {window_size}-Iter Moving Avg')
    
    plt.xlabel('Evaluations (Init + Iterations)', fontsize=12)
    plt.ylabel('Cost (Wire Length + Penalties)', fontsize=12)
    plt.title('Algorithm Convergence Comparison: BO vs GA vs SA', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    return bo_costs, ga_costs, sa_costs


def plot_grid_with_path(grid_obj, path, k, title, color='blue', linestyle='-', 
                        marker_color='cyan', window_title="Grid Plot"):
    """
    Plot EHWP grid with overlaid path
    
    Args:
        grid_obj: EHWP_Grid instance
        path: flat 1D array of coordinates (length 2*k)
        k: number of RPUs
        title: plot title
        color: line color
        linestyle: line style
        marker_color: RPU marker color
        window_title: title for the figure window
    """
    grid_obj.display_grid()
    ax = plt.gca()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(window_title)
    
    # Reshape path
    reshaped_path = path.reshape(k, 2)
    
    # Generate full Manhattan path for visualization
    full_visual_path = []
    for step in range(k - 1):
        segment = get_manhattan_path(reshaped_path[step], reshaped_path[step+1])
        if step > 0:
            segment = segment[1:]  # Prevent overlapping coordinates
        full_visual_path.extend(segment)
    
    full_visual_path = np.array(full_visual_path)
    
    # Plot L-shaped wires
    ax.plot(full_visual_path[:, 1], full_visual_path[:, 0], 
            color=color, linewidth=4, linestyle=linestyle, label=f'{title} Route')
    
    # Plot RPU markers
    ax.scatter(reshaped_path[:, 1], reshaped_path[:, 0], 
               color=marker_color, s=150, zorder=5, edgecolors='black', 
               label=f'{title} RPUs')
    
    # Number the RPUs to show sequence
    for step_idx in range(k):
        ax.annotate(str(step_idx + 1), 
                    (reshaped_path[step_idx, 1], reshaped_path[step_idx, 0]), 
                    color='black', weight='bold', fontsize=10, 
                    ha='center', va='center', zorder=6)
    
    ax.legend(loc="upper right", title=f"{title} Results")
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    # Get user inputs
    n = int(input("Enter the dimension of the EHWP grid: "))
    i = int(input("Enter the number of infected RPUs: "))
    func_string = input("Enter the function to run (e.g., (a*((b+c)*(d/e))): ")
    n_iter = int(input("Enter the number of iterations (same for BO, GA, and SA): "))
    mutation_rate = float(input("Enter the mutation rate for GA: "))
    
    # Parse the function string to count operators
    operators = re.findall(r'[\+\-\*\/]', func_string)
    k = len(operators)
    if k == 0:
        k = 1  # Fallback
    print(f"\nDetected {k} operations. Searching for path connecting {k} RPUs.")
    
    # Create EHWP grid
    EHWP = EHWP_Grid(n, i)
    
    # ============ BAYESIAN OPTIMIZATION ============
    print("\n" + "="*50)
    print("BAYESIAN OPTIMIZATION")
    print("="*50)
    
    bo_optimizer = BO(n, k, EHWP.grid, objective_function)
    train_X, train_Y = bo_optimizer.run_optimization(n_iter=n_iter, n_init=10)
    
    # Extract best BO result
    best_idx = torch.argmin(train_Y).item()
    bo_best_cost = train_Y[best_idx].item()
    best_path_cont = train_X[best_idx].detach().numpy()
    bo_best_path = np.round(best_path_cont).astype(int)
    
    print(f"\nBO Best cost: {bo_best_cost}")
    print(f"BO Best path: {bo_best_path}")
    
    # ============ GENETIC ALGORITHM ============
    print("\n" + "="*50)
    print("GENETIC ALGORITHM")
    print("="*50)
    
    ga_optimizer = GA(n, k, EHWP.grid, objective_function, 
                      pop_size=10, mutation_rate=mutation_rate)
    ga_all_costs, ga_best_path, ga_best_cost = ga_optimizer.run_optimization(n_iter=n_iter)
    
    print(f"\nGA Best cost: {ga_best_cost}")
    print(f"GA Best path: {ga_best_path}")
    
    # ============ SIMULATED ANNEALING ============
    print("\n" + "="*50)
    print("SIMULATED ANNEALING")
    print("="*50)
    
    # SA parameters - tuned for exactly n_init + n_iter evaluations
    sa_optimizer = SA(n, k, EHWP.grid, objective_function,
                      initial_temp=100.0, min_temp=0.1)
    sa_all_costs, sa_best_path, sa_best_cost = sa_optimizer.run_optimization(
        n_iter=n_iter, n_init=10
    )
    
    print(f"\nSA Best cost: {sa_best_cost}")
    print(f"SA Best path: {sa_best_path}")
    
    # ============ VERIFY EVALUATION COUNTS ============
    print("\n" + "="*50)
    print("VERIFYING EVALUATION COUNTS")
    print("="*50)
    
    bo_count = len(train_Y)
    ga_count = len(ga_all_costs)
    sa_count = len(sa_all_costs)
    
    print(f"BO evaluations: {bo_count}")
    print(f"GA evaluations: {ga_count}")
    print(f"SA evaluations: {sa_count}")
    
    expected_total = 10 + n_iter
    assert bo_count == expected_total, f"BO used {bo_count} but should use {expected_total}"
    assert ga_count == expected_total, f"GA used {ga_count} but should use {expected_total}"
    assert sa_count == expected_total, f"SA used {sa_count} but should use {expected_total}"
    
    # ============ VISUALIZATION ============
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # 1. Convergence comparison plot (all three algorithms)
    bo_costs_np, ga_costs_np, sa_costs_np = plot_convergence_comparison(
        train_Y, ga_all_costs, sa_all_costs, window_size=5
    )
    
    # 2. BO Grid Plot
    plot_grid_with_path(
        EHWP, bo_best_path, k, 
        title="BO", color='blue', linestyle='-', 
        marker_color='cyan', window_title="BO Optimization Result"
    )
    
    # 3. GA Grid Plot
    plot_grid_with_path(
        EHWP, ga_best_path, k, 
        title="GA", color='red', linestyle='--', 
        marker_color='red', window_title="GA Optimization Result"
    )
    
    # 4. SA Grid Plot (NEW)
    plot_grid_with_path(
        EHWP, sa_best_path, k, 
        title="SA", color='yellow', linestyle=':', 
        marker_color='yellow', window_title="SA Optimization Result"
    )
    
    # Print summary statistics
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"{'Algorithm':<10} {'Best Cost':<15} {'Evaluations':<15}")
    print(f"{'-'*40}")
    print(f"{'BO':<10} {bo_best_cost:<15.2f} {bo_count:<15}")
    print(f"{'GA':<10} {ga_best_cost:<15.2f} {ga_count:<15}")
    print(f"{'SA':<10} {sa_best_cost:<15.2f} {sa_count:<15}")
    
    # Determine winner
    costs = [bo_best_cost, ga_best_cost, sa_best_cost]
    algorithms = ['BO', 'GA', 'SA']
    winner_idx = np.argmin(costs)
    print(f"\n🏆 Best performing algorithm: {algorithms[winner_idx]} with cost {costs[winner_idx]:.2f}")
    
    print("\nClose plot windows to exit the program")
    plt.show()