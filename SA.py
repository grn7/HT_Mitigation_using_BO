import numpy as np
import math

class SA:
    def __init__(self, n, k, grid, objective_fn, initial_temp=100.0, cooling_rate=0.95, min_temp=0.1):
        """
        Simulated Annealing optimizer for HT mitigation path finding
        
        Args:
            n: grid size (n x n)
            k: number of RPUs needed (operations)
            grid: EHWP grid with infected RPUs
            objective_fn: cost function to minimize
            initial_temp: starting temperature
            cooling_rate: multiplicative cooling factor (0 < cooling_rate < 1)
            min_temp: stopping temperature for inner loop
        """
        self.n = n
        self.k = k
        self.grid = grid
        self.objective_fn = objective_fn
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
        # Track all costs for convergence plotting
        self.all_sampled_costs = []
        
    def generate_random_path(self):
        """Generate a random path of k RPUs"""
        return np.random.randint(0, self.n, size=2*self.k)
    
    def generate_neighbor(self, current_path):
        """
        Generate a neighboring solution by:
        1. Randomly select one RPU to modify
        2. Randomly change either x or y coordinate of that RPU by ±1 (if within bounds)
        3. Alternative: sometimes make a larger jump to escape local minima
        """
        neighbor = current_path.copy()
        
        # Decide mutation type: small local move (80%) or large random jump (20%)
        if np.random.random() < 0.8:
            # Local move: pick one coordinate and shift by ±1
            coord_idx = np.random.randint(0, 2*self.k)
            
            # Decide direction: +1 or -1
            step = np.random.choice([-1, 1])
            
            # Apply step, ensuring within bounds [0, n-1]
            new_val = neighbor[coord_idx] + step
            if 0 <= new_val < self.n:
                neighbor[coord_idx] = new_val
        else:
            # Large jump: randomly regenerate one entire RPU (both coordinates)
            rpu_idx = np.random.randint(0, self.k)
            new_coords = np.random.randint(0, self.n, size=2)
            neighbor[2*rpu_idx:2*rpu_idx+2] = new_coords
            
        return neighbor
    
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """
        Calculate probability of accepting worse solution
        Uses Boltzmann distribution: exp(-(new_cost - current_cost) / temperature)
        """
        if new_cost < current_cost:
            return 1.0
        else:
            return math.exp(-(new_cost - current_cost) / temperature)
    
    def run_optimization(self, n_iter=None, n_init=10):
        """
        Run Simulated Annealing optimization
        
        Args:
            n_iter: number of iterations (must match BO/GA budget)
            n_init: number of random initializations (must match BO's n_init)
        
        Returns:
            all_sampled_costs: list of all costs evaluated (length = n_init + n_iter)
            best_path: best path found
            best_cost: best cost found
        """
        if n_iter is None:
            raise ValueError("n_iter must be specified to match BO/GA budget")
        
        total_evaluations = n_init + n_iter
        print(f"\nSimulated Annealing initialized with {n_init} random starts")
        print(f"Target total evaluations: {total_evaluations}")
        
        # PHASE 1: Random initialization
        initial_paths = []
        initial_costs = []
        
        for i in range(n_init):
            path = self.generate_random_path()
            cost = self.objective_fn(path, self.grid, self.k, self.n)
            initial_paths.append(path)
            initial_costs.append(cost)
            self.all_sampled_costs.append(cost)
            
        # Find best initial solution as starting point for annealing
        best_init_idx = np.argmin(initial_costs)
        current_path = initial_paths[best_init_idx].copy()
        current_cost = initial_costs[best_init_idx]
        
        best_path = current_path.copy()
        best_cost = current_cost
        
        print(f"Best initial cost: {best_cost:.1f}")
        print(f"Starting annealing with T0={self.initial_temp}")
        
        # PHASE 2: Simulated Annealing - run exactly n_iter iterations
        temperature = self.initial_temp
        
        # Calculate cooling rate to ensure we use exactly n_iter iterations
        # We want temperature to reach min_temp after n_iter iterations
        # T_n = T0 * (cooling_rate)^n_iter = min_temp
        # So cooling_rate = (min_temp/T0)^(1/n_iter)
        if n_iter > 0:
            self.cooling_rate = (self.min_temp / self.initial_temp) ** (1.0 / n_iter)
        
        for iteration in range(n_iter):
            # Generate neighbor
            new_path = self.generate_neighbor(current_path)
            new_cost = self.objective_fn(new_path, self.grid, self.k, self.n)
            
            # Track this evaluation
            self.all_sampled_costs.append(new_cost)
            
            # Decide whether to accept
            if self.acceptance_probability(current_cost, new_cost, temperature) > np.random.random():
                current_path = new_path
                current_cost = new_cost
                
                # Update global best if improved
                if current_cost < best_cost:
                    best_path = current_path.copy()
                    best_cost = current_cost
            
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Progress tracking
            window_size = 5
            if len(self.all_sampled_costs) >= window_size:
                # Calculate moving average over last 5 evaluations
                moving_avg = np.mean(self.all_sampled_costs[-window_size:])
                print(f"SA Iteration {iteration+1}/{n_iter} - T: {temperature:.2f} | "
                      f"Current: {current_cost:.1f} | 5-Iter Avg: {moving_avg:.1f} | "
                      f"Best: {best_cost:.1f}")
            else:
                print(f"SA Iteration {iteration+1}/{n_iter} - T: {temperature:.2f} | "
                      f"Current: {current_cost:.1f} | Best: {best_cost:.1f}")
        
        # Verify we used exactly the right number of evaluations
        actual_evaluations = len(self.all_sampled_costs)
        print(f"\nSA completed. Total evaluations: {actual_evaluations} (target: {total_evaluations})")
        print(f"Final best cost: {best_cost:.1f}")
        
        # Ensure we have exactly n_init + n_iter evaluations
        assert actual_evaluations == total_evaluations, \
            f"SA used {actual_evaluations} evaluations but should use {total_evaluations}"
        
        return self.all_sampled_costs, best_path, best_cost