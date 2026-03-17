import numpy as np

class GA:

    def __init__(self, n, k, grid, objective_fn, pop_size = 10, mutation_rate = 0.2):
        self.n = n
        self.k = k
        self.grid = grid
        self.objective_fn = objective_fn
        self.pop_size = pop_size # population size initially
        self.mutation_rate = mutation_rate

        self.population = np.random.randint(0, self.n, size=(self.pop_size,2*self.k))

        self.all_sampled_costs = []
        for i in range(self.pop_size):
            self.all_sampled_costs.append(self.objective_fn(self.population[i], self.grid, self.k, self.n))

    def run_optimization(self, n_iter):
        # create a local tracker for the active population's costs
        current_costs = self.all_sampled_costs.copy()

        # run the loop
        for _ in range(n_iter):

            # SELECTION
            # pick 3 random indices for the first tournament 
            t1_indices = np.random.choice(self.pop_size, 3, replace=False)
            # picks 3 nums between 0 and pop_size-1
            # find index of the winner among chosen 3
            p1_idx = t1_indices[np.argmin([current_costs[i] for i in t1_indices])]
            # if the inner code gives a list of costs say [500 100 300], argmin gives 1 as output as 100 is at index 1
            # t1[1] will grab the original number that gave this cost
            parent1 = self.population[p1_idx]

            # pick 3 for 2nd tournament
            t2_indices = np.random.choice(self.pop_size, 3, replace=False)
            p2_idx = t2_indices[np.argmin([current_costs[i] for i in t2_indices])]
            parent2 = self.population[p2_idx]

            # CROSSOVER
            # create a random binary mask of length k
            node_mask = np.random.randint(0, 2, size=self.k).astype(bool) # 1 becomes true, 0 false
            # make a mask of size k and duplicate each one so that it becomes 2k
            # this way if we pick something we pick the x,y pair whereas if we make a mask of 2k,
            # we may end up taking x from one parent and y from another
            mask = np.repeat(node_mask,2) # duplicates each item twice in a row
            child = np.where(mask, parent1, parent2) # if True grab from parent1, False then parent2


            # MUTATION
            # create an array of random floats between 0 and 1 of size 2k and check if they are lesser than mutation rate 
            mutate_mask = np.random.rand(2*self.k) < self.mutation_rate
            # False if num greater than mutation else true
            # generate random coordinates in the grid range 
            random_coords = np.random.randint(0, self.n, size=2*self.k)
            # apply the random coords wherever mutate_mask is true
            child = np.where(mutate_mask, random_coords, child)

            # EVALUATE AND TRACK
            # evaluate new child's cost
            child_cost = self.objective_fn(child, self.grid, self.k, self.n)
            # append child's cost to historical log
            self.all_sampled_costs.append(child_cost)
            # find index of individual with highest cost in population
            worst_idx = np.argmax(current_costs)
            # replace the worst individual's genes and cost with the new child's data
            self.population[worst_idx] = child
            current_costs[worst_idx] = child_cost

            # print progess tracking 
            current_best = np.min(current_costs)
            window_size = 5
            if len(self.all_sampled_costs) >= window_size:
                moving_avg = np.mean(self.all_sampled_costs[-window_size:])
                print(f"Iteration {_ + 1}/{n_iter} - Current: {child_cost:.1f} | 5-Iter Avg: {moving_avg:.1f} | Best: {current_best:.1f}")
            else:
                print(f"Iteration {_ + 1}/{n_iter} - Current: {child_cost:.1f} | Best: {current_best:.1f}") 

        # find overall winner in final population to plot 
        final_best_idx = np.argmin(current_costs)
        best_path = self.population[final_best_idx]
        best_cost = current_costs[final_best_idx]

        return self.all_sampled_costs, best_path, best_cost








