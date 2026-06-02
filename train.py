import os 
import numpy as np
import torch
import random
import time 
from collections import deque
import matplotlib.pyplot as plt
from init import EHWP_Grid
from environment import EHWPEnv
from agent import DoubleDQNAgent, ReplayBuffer

def train_master_agent(n, infected_rpus):
    # --- CHANGE MADE HERE: Updated k bounds for 64x64 grid ---
    k_min = 20
    k_max = 200
    # ---------------------------------------------------------
    max_episodes = 10000
    batch_size = 64

    # naming weights based on parameters
    weight_filename = f"dqn_{n}x{n}_{infected_rpus}_{k_min}_{k_max}_{max_episodes//1000}k.pt"

    # initialize agent and replay buffer 
    agent = DoubleDQNAgent(n)
    
    # Increased capacity from 10000 to 50000 for deeper memory
    memory_buffer = ReplayBuffer(capacity=50000, batch_size=batch_size)

    # pre-training weight check 
    if os.path.exists(weight_filename):
        print(f"\nFound existing weights: {weight_filename}")
        print("Loading weights and skipping training entirely\n")
        # load saved state dictionary into online net
        agent.online_net.load_state_dict(torch.load(weight_filename))
        # sync target net
        agent.update_target_network()
        return agent # exit training immediately


    print(f"\nNo existing weights found for {weight_filename}.")
    print(f"Commencing {max_episodes} episode master training loop\n")

    # convergence monitoring setup
    recent_rewards = deque(maxlen=100)
    
    # Commented out early stopping setup
    # best_100_avg = -float('inf')
    # episodes_without_improvement = 0
    # convergence_patience = 500 

    global_best_cost_path = float('inf')

    # Initialize list to store the 100-episode averages for plotting
    history_100_avg = []

    # Instantiate environment once outside the loop to save compute overhead
    dummy_grid = np.zeros((n, n))
    env = EHWPEnv(n, k_min, dummy_grid)

    # Start the training stopwatch
    training_start_time = time.time()

    for episode in range(1, max_episodes+1):
        # --- CHANGE MADE HERE: Added step size of 100 ---
        # variable path length k, stepped by 100
        k = random.randrange(k_min, k_max + 1, 100)
        # ------------------------------------------------

        # randomized grid generation
        # generate new layout each time to avoid memorization
        grid_generator = EHWP_Grid(n, infected_rpus)
        fresh_grid = grid_generator.grid

        # episode execution
        env.k = k 
        state = env.reset(fresh_grid)
        valid_mask = env.get_valid_action()

        episode_reward = 0

        # inner loop: run exactly for k times
        for step_idx in range(k):
            # 1. interaction- agent performs act (with masking)
            action = agent.act(state, valid_mask)

            # 2. environment processes the step
            next_state, reward, done, _ = env.step(action)
            next_valid_mask = env.get_valid_action()

            # 3. push to memory buffer 
            memory_buffer.push(state, action, reward, next_state, done)

            # 4. learn from a batch of memories
            agent.learn(memory_buffer, batch_size=batch_size)

            # update state vars for next step
            state = next_state
            valid_mask = next_valid_mask
            episode_reward += reward

            if done:
                break

        # epsilon decay 
        agent.decay_epsilon()

        # track metrics 
        recent_rewards.append(episode_reward)
        current_cost = -(episode_reward-1000) 

        if current_cost < global_best_cost_path:
            global_best_cost_path = current_cost

        # convergence monitoring 
        if len(recent_rewards) == 100:
            current_100_avg = np.mean(recent_rewards)

            # Commented out early stopping execution
            # if current_100_avg > best_100_avg + 1.0: 
            #     best_100_avg = current_100_avg
            #     episodes_without_improvement = 0
            # else:
            #     episodes_without_improvement += 1

            # if episodes_without_improvement >= convergence_patience:
            #     print(f"\nTraining converged at episode {episode}. No improvement in 500 episodes")
            #     break

        # logging metrics
        if episode % 100 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward
            
            # Appending the average here, exactly once every 100 episodes
            history_100_avg.append(avg_reward)
            
            print(f"Episode: {episode:6d}/{max_episodes} | Average Reward (Last 100): {avg_reward:8.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | Best Global Cost: {global_best_cost_path:.1f}")

    # Stop timer and append to log file 
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours, rem = divmod(total_training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_formatted = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    print(f"\nTraining complete. Saving weights to {weight_filename}")
    print(f"Total Training Time: {time_formatted}")
    
    # Save the record to a persistent text file
    log_file_path = "training_time_log.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Weights: {weight_filename} | Episodes: {max_episodes} | Time: {time_formatted}\n")
    print(f"Training time successfully logged to {log_file_path}")
    
    torch.save(agent.online_net.state_dict(), weight_filename)

    # Generate and save the training convergence plot and data
    if len(history_100_avg) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history_100_avg, color='blue', linewidth=2)
        plt.title(f'RL Training Convergence ({n}x{n} Grid, {infected_rpus} HTs)')
        plt.xlabel('Episodes (x100)')
        plt.ylabel('100-Episode Moving Avg Reward')
        plt.grid(True, alpha=0.3)
        
        plot_filename = f'dqn_training_curve_{n}x{n}.png'
        plt.savefig(plot_filename)
        plt.close() # Closes it in the background to save memory
        print(f"Convergence plot saved as {plot_filename}")

        # Save the raw array just in case you want to format it differently later
        data_filename = f'dqn_convergence_data_{n}x{n}.npy'
        np.save(data_filename, history_100_avg)
        print(f"Raw convergence data saved as {data_filename}\n")

    return agent  

if __name__ == "__main__":
    # --- CHANGE MADE HERE: Updated main execution parameters for 64x64 ---
    trained_agent = train_master_agent(n=64, infected_rpus=40)
    # ---------------------------------------------------------------------