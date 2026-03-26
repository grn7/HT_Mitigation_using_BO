# run_experiment.py - Automated Large-Scale Benchmarking Script
# Evaluates GA, SA, and RL across varying path lengths (k) and multiple random grids.

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch # Ensure torch is imported for the no_grad block

# Import your custom project modules
from init import EHWP_Grid
from function import objective_function
from GA import GA
from SA import SA  
from environment import EHWPEnv
from train import train_master_agent

def run_large_scale_experiment():
    # ==========================================
    # 1. EXPERIMENT SETUP & CONFIGURATION
    # ==========================================
    n = 16
    infected_rpus = 10
    k_values = range(20, 201, 10) # k from 20 to 200 in steps of 10
    runs_per_k = 20
    
    # Algorithm Hyperparameters
    # --- CHANGE MADE HERE: Reduced iterations from 500 to 100 ---
    n_iter = 1000
    mutation_rate = 0.4
    
    # Directory Management
    # --- CHANGE MADE HERE: Renamed folder to prevent overwriting previous 500-iter data ---
    out_dir = "experiment_results_1000_iters"
    os.makedirs(out_dir, exist_ok=True)
    csv_filename = os.path.join(out_dir, "experiment_data_16x16.csv")
    
    experiment_data = []

    print("="*70)
    print(f"INITIALIZING LARGE SCALE EXPERIMENT")
    print(f"Grid: {n}x{n} | HTs: {infected_rpus} | k: 20 -> 200 | Runs/k: {runs_per_k} | Iters: {n_iter}")
    print("="*70)

    # Load the RL Master Agent ONCE before the loops start
    print("\nLoading Pre-Trained RL Master Agent...")
    rl_agent = train_master_agent(n, infected_rpus)
    rl_agent.online_net.eval() # Set to evaluation mode
    original_epsilon = rl_agent.epsilon
    
    print(f"\nStarting 380-Run Execution Loop. Results will stream below:\n")

    # ==========================================
    # 2. THE EXECUTION LOOP
    # ==========================================
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['k', 'run_id', 'GA_cost', 'GA_time', 'SA_cost', 'SA_time', 'RL_cost', 'RL_time', 'Winner', 'RL_Trapped']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for k in k_values:
            for run in range(1, runs_per_k + 1):
                # A. Generate a completely fresh grid for this specific run
                EHWP = EHWP_Grid(n, infected_rpus)
                fresh_grid = EHWP.grid
                
                # B. Run Genetic Algorithm (GA)
                ga_optimizer = GA(n, k, fresh_grid, objective_function, pop_size=10, mutation_rate=mutation_rate)
                t0 = time.time()
                _, _, ga_cost = ga_optimizer.run_optimization(n_iter=n_iter)
                ga_time = max(time.time() - t0, 1e-6)
                
                # C. Run Simulated Annealing (SA)
                sa_optimizer = SA(n, k, fresh_grid, objective_function, initial_temp=100.0, min_temp=0.1)
                t0 = time.time()
                _, _, sa_cost = sa_optimizer.run_optimization(n_iter=n_iter, n_init=10)
                sa_time = max(time.time() - t0, 1e-6)
                
                # D. Run Reinforcement Learning (RL) Inference
                # SAFEST APPROACH: Fresh environment per run guarantees no state contamination
                rl_env = EHWPEnv(n, k, fresh_grid)
                state = rl_env.reset(fresh_grid)
                valid_mask = rl_env.get_valid_action()
                
                rl_path_coords = []
                rl_agent.epsilon = 0.0 # Force pure exploitation
                
                t0 = time.time()
                # Prevent Memory Leak during inference
                with torch.no_grad():
                    for _ in range(k):
                        action = rl_agent.act(state, valid_mask)
                        row, col = rl_env.decode_action(action)
                        rl_path_coords.extend([row, col])
                        state, _, done, _ = rl_env.step(action)
                        valid_mask = rl_env.get_valid_action()
                        if done: 
                            break
                            
                # SAFE LOG SCALE: Prevent 0.0 seconds from crashing the graph
                rl_time = max(time.time() - t0, 1e-6) 
                
                rl_best_path = np.array(rl_path_coords)
                
                # Check for incomplete/trapped paths
                rl_trapped = False
                if len(rl_best_path) < (2 * k):
                    rl_cost = 99999.0 # Massive penalty cost
                    rl_trapped = True
                    rl_status_str = "TRAPPED"
                else:
                    rl_cost = objective_function(rl_best_path, fresh_grid, k, n)
                    rl_status_str = f"{rl_cost:<7.1f}"
                
                # E. Determine the Winner for this specific run
                costs = {'GA': ga_cost, 'SA': sa_cost, 'RL': rl_cost}
                winner = min(costs, key=costs.get)
                
                # F. Terminal Output 
                print(f"[k={k:<3} | Run {run:02d}/20] "
                      f"GA: {ga_cost:<7.1f} | SA: {sa_cost:<7.1f} | RL: {rl_status_str} "
                      f"--> 🏆 {winner} Wins!")
                
                # G. Save data to CSV
                row_data = {
                    'k': k, 'run_id': run,
                    'GA_cost': ga_cost, 'GA_time': ga_time,
                    'SA_cost': sa_cost, 'SA_time': sa_time,
                    'RL_cost': rl_cost, 'RL_time': rl_time,
                    'Winner': winner,
                    'RL_Trapped': rl_trapped
                }
                writer.writerow(row_data)
                
                # Force save to hard drive immediately
                csv_file.flush() 
                
                experiment_data.append(row_data)
                
    # Restore RL agent state
    rl_agent.epsilon = original_epsilon

    print("\n" + "="*70)
    print(f"EXPERIMENT COMPLETE. Data saved to {csv_filename}")
    print("Generating Visualizations...")
    print("="*70)

    # ==========================================
    # 3. DATA PROCESSING & VISUALIZATIONS
    # ==========================================
    
    k_list = list(k_values)
    
    avg_costs = {'GA': [], 'SA': [], 'RL': []}
    avg_times = {'GA': [], 'SA': [], 'RL': []}
    wins = {'GA': defaultdict(int), 'SA': defaultdict(int), 'RL': defaultdict(int)}

    for k_val in k_list:
        k_data = [row for row in experiment_data if row['k'] == k_val]
        
        avg_costs['GA'].append(np.mean([row['GA_cost'] for row in k_data]))
        avg_costs['SA'].append(np.mean([row['SA_cost'] for row in k_data]))
        
        rl_valid_costs = [row['RL_cost'] for row in k_data if not row['RL_Trapped']]
        if rl_valid_costs:
            avg_costs['RL'].append(np.mean(rl_valid_costs))
        else:
            avg_costs['RL'].append(np.nan) 
        
        avg_times['GA'].append(np.mean([row['GA_time'] for row in k_data]))
        avg_times['SA'].append(np.mean([row['SA_time'] for row in k_data]))
        avg_times['RL'].append(np.mean([row['RL_time'] for row in k_data]))
        
        for row in k_data:
            wins[row['Winner']][k_val] += 1

    # ---- PLOT 1: Stacked Bar Chart of Wins ----
    plt.figure(figsize=(10, 7))
    algorithms = ['GA', 'SA', 'RL']
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_list)))
    bottoms = np.zeros(len(algorithms))
    
    for idx, k_val in enumerate(k_list):
        k_wins = [wins['GA'][k_val], wins['SA'][k_val], wins['RL'][k_val]]
        plt.bar(algorithms, k_wins, bottom=bottoms, color=colors[idx], 
                edgecolor='white', label=f'k={k_val}' if idx % 2 == 0 else "") 
        bottoms += np.array(k_wins)

    plt.title('Total Wins by Algorithm (Stacked by Path Length $k$)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Wins', fontsize=12)
    plt.legend(title='Path Length (k)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_1_win_tracker.png'))
    plt.close()

    # ---- PLOT 2: Average Cost vs k ----
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, avg_costs['GA'], marker='o', color='red', linewidth=2, label='GA')
    plt.plot(k_list, avg_costs['SA'], marker='s', color='yellow', linewidth=2, label='SA')
    plt.plot(k_list, avg_costs['RL'], marker='^', color='magenta', linewidth=2, label='RL (Successful Runs Only)')
    
    plt.title('Average Path Cost vs. Number of Operators ($k$)', fontsize=14, fontweight='bold')
    plt.xlabel('Path Length / Operators ($k$)', fontsize=12)
    plt.ylabel('Average Cost', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_2_cost_vs_k.png'))
    plt.close()

    # ---- PLOT 3: Average Inference Time vs k ----
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, avg_times['GA'], marker='o', color='red', linewidth=2, label='GA')
    plt.plot(k_list, avg_times['SA'], marker='s', color='yellow', linewidth=2, label='SA')
    plt.plot(k_list, avg_times['RL'], marker='^', color='magenta', linewidth=2, label='RL (Inference)')
    
    plt.title('Average Execution Time vs. Number of Operators ($k$)', fontsize=14, fontweight='bold')
    plt.xlabel('Path Length / Operators ($k$)', fontsize=12)
    plt.ylabel('Time (Seconds)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which="both", ls="--")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_3_time_vs_k.png'))
    plt.close()

    print(f"Successfully generated and saved 3 plots to the '{out_dir}' directory.")
    print("Done! You can now analyze your results in the CSV file.")

if __name__ == "__main__":
    run_large_scale_experiment()