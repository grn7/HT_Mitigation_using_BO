import numpy as np
from function import manhattan_dist

class EHWPEnv:
    def __init__(self, n, k, grid):
        self.n = n
        self.k = k
        self.initial_grid = grid
        self.reset(grid)

    def reset(self, grid=None):
        self.grid = grid if grid is not None else self.initial_grid
        # initialise components 
        # maintain a n*n matrix of zeros for visited map
        self.visited_map = np.zeros((self.n, self.n))
        self.step_counter = 0
        self.path = [] # store coords to calc dist
        # set action dimension
        # every coord in the grid is a possible action
        self.action_size = self.n * self.n
        return self._get_obs()
    
    # concatenation logic 
    def _get_obs(self):
        # flatten nxn grid
        grid_flat = self.grid.flatten()
        # flatten visited map
        visited_flat = self.visited_map.flatten()
        # convert step counter into numpy array for concatenation
        step_array = np.array([self.step_counter])
        # concatenate into single array of length 2n^2+1
        obs = np.concatenate((grid_flat, visited_flat, step_array))

        return obs

    def decode_action(self, action_index):
        # translate an action int i to grid coords
        row = action_index // self.n # floor division
        col = action_index % self.n 

        return row, col

    # implement action masking
    def get_valid_action(self):
        # flatten the grid and visited map to compare with 1D action spce 
        grid_flat = self.grid.flatten()
        visited_flat = self.visited_map.flatten()
        # mark an index as invalid(false) if it is infected OR already visited 
        # ~ is like not gate
        valid_mask = ~((grid_flat == 1) | (visited_flat == 1))

        # return boolean mask of length n^2 
        return valid_mask

    def step(self, action_index):
        # executes one action and returns next_state,reward, done, info
        row, col = self.decode_action(action_index)
        current_coord = (row,col)
        reward = 0

        # penalty : already visited
        if self.visited_map[row, col] == 1:
            reward -= 5000
        # manhattan dist
        if len(self.path) > 0:
            dist = manhattan_dist(self.path[-1], current_coord)
            reward -= dist
        # penalty : infection ( hard and soft)
        if self.grid[row, col] == 1:
            reward -= self.n * self.n * 100
        else:
            adjacent_infected = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = row+dr, col+dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    if self.grid[nr, nc] == 1:
                        adjacent_infected += 1
            reward -= adjacent_infected * (self.n*5) 
        # update state  
        self.visited_map[row, col] = 1
        self.path.append(current_coord)
        self.step_counter += 1 # increment counter very time a waypoint is placed 

        # check termination 
        done = (self.step_counter >= self.k)

        # positive reward for completing a good path
        if done:
            reward +=1000

        return self._get_obs(), reward, done, {}
