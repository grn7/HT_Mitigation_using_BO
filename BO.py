import torch # used to handle and process tensors(multi dim arrays) very fast
import numpy as np
from scipy.stats import qmc # scipy.stats contains collection of statistical tools
# qmc is quasi monte carlo, for evenly spaced sampling 
# It generates quasi-random numbers, which 
# are designed to spread out and fill a space very evenly
from botorch.models import SingleTaskGP # for gaussian process
from botorch.models.transforms.outcome import Standardize # for translating raw data into a better form GP can process better
from botorch.models.transforms.input import Normalize # Normalizer to scale grid coordinates to a unit cube (0 to 1)

from gpytorch.mlls import ExactMarginalLogLikelihood # this is the loss function / scorekeeper
# it calculates how well the GP's current predictions match the real dataset train_X and train_Y
# higher the score more accurate the model is
from botorch.fit import fit_gpytorch_mll
# instead of manually training your model step by step, u give it gp model and scorekeeper
# it automatically runs it until the model fits your data well

from botorch.acquisition import LogExpectedImprovement # Replaced ExpectedImprovement with LogExpectedImprovement to fix math crashes
from botorch.optim import optimize_acqf # respects the bounds and pinpoints coords of path to try next


class BO:
    def __init__(self, n, k, grid, objective_fn):
        self.n = n
        self.k = k 
        self. grid = grid
        self.objective_fn = objective_fn
    
    def intialise_search_space(self, n_init):
        # each path has k waypoints, so dim of array is 2*k
        dimension = 2 * self.k 

        #initialise the sampler
        sampler = qmc.LatinHypercube(d = dimension)
        # qmc.LatinHy.. is how we are picking the numbers
        # d = .. tells the sampler how many variables you are working with
        # the setup is saved into sampler so we can use it 
        # this lines doesnt actually generate numbers, it is only prepping to generate numbers 

        #generate the samples
        sample = sampler.random(n = n_init)
        # gives out n_init evenly spaced points
        # all the numbers here are between 0 and 1 

        # scale the samples to work in the grid
        l_bounds = [0] * dimension
        u_bounds = [self.n - 1] * dimension
        scaled_sample = qmc.scale(sample,l_bounds,u_bounds)  
        # if ur original sample was 0.5 and ur bounds are 0 and 100, 0.5 is now 50

        # round to ints as coordinates cant be in decimal
        int_paths = np.round(scaled_sample).astype(int)
        # round takes it from 4.7 to 5.0, int takes it from 5.0 to 5
        # pure ints are needed when python tries to index the grid

        # evaluate objective function for each path
        train_X_list = []
        train_Y_list = []

        for i in range(n_init):
            continuous_path = scaled_sample[i]
            rounded_path = int_paths[i]

            cost = self.objective_fn(rounded_path, self.grid, self.k, self.n) 
            train_X_list.append(continuous_path) 

            # why append floats in the GP data not ints?
            # We store the continuous 'scaled_sample' (floats) in train_X rather than 'int_paths'.
            # 1. Gradient Information: GPs need continuous values to calculate the 'slope' 
            #    of the cost surface and predict where the next best point lies.
            # 2. Preventing Aliasing: Using only integers creates 'staircase' data where 
            #    multiple different inputs map to the same point, causing numerical 
            #    instability and singular matrices in the GP covariance math.
            # 3. Search Precision: It allows the optimizer to 'slide' between RPUs to 
            #    identify which direction (e.g., 2.1 vs 2.9) leads to a better global minimum.

            train_Y_list.append(cost)

        # convert to pytorch tensors for BO loop 
        train_X = torch.tensor(np.array(train_X_list), dtype = torch.float64) 
        train_Y = torch.tensor(train_Y_list, dtype = torch.float64).unsqueeze(-1) 
        # converts row  vector to column vector as the BO libraries require this

        return train_X, train_Y
    
    def get_optimizer_bounds(self):
        dimensions = 2 * self.k 
        low_bounds = torch.zeros(dimensions, dtype = torch.float64) 
        upper_bounds = torch.full((dimensions,), self.n -1, dtype = torch.float64) 

        bounds = torch.stack([low_bounds, upper_bounds]) # creates a 2D matrix 
        # this is what botorch needs , row 0 is mins, 1 is maxs
        return bounds
    
    def fit_GP(self, train_X, train_Y):
        # negate the costs as BoTorch's acquisiton functions are meant to find max val
        neg_train_Y = -train_Y

        # Get the absolute grid boundaries
        bounds = self.get_optimizer_bounds()

        # use the GP
        # CHANGED: Added input_transform=Normalize with bounds to fix the Unit Cube scaling crash
        model = SingleTaskGP(
            train_X, 
            neg_train_Y, 
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=train_X.shape[-1], bounds=bounds)
        )
        # our objective function outputs very large nos due the penalty, which can the cause model's internal "lengthscales" to become unstable,
        # making it hard to find good fit. so we scale costs to have mean of 0, var of 1 using standardise which keeps the math stable
        # lengthscale tells the ath how smooth or jagged it should assume the relation bw paths and costs will be
        # m=1 tells that output data has exactly 1 dimension, a single cost score per path
        # model saves the fully built, data loaded GP so we can proceed to train it

        # fit the hyperparameters
        mll = ExactMarginalLogLikelihood(model.likelihood, model) # measures how well the model's current settings explain actual data
        fit_gpytorch_mll(mll) # trains the GP to understand patters in the grid
        # adjusts internal hyperparameters of model so it can accurately predict cost of paths it hasnt seen yet

        return model, mll
    
    def get_next_path(self, model, train_Y, bounds):
        # find current best
        neg_train_Y = -train_Y
        best_f = neg_train_Y.max()

        # initialise acq fn
        EI = LogExpectedImprovement(model=model, best_f=best_f) 
        # model is our trained GP

        # optimise 
        new_path, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        # bounds tells it search within valid limits
        # q =1 gives a single best path to try next 
        # EI creates a map of our problems, which contains hills(good paths) and valleys(bad paths)
        # raw samples = 20, the algo picks 20 random points into the EI function and look at the score
        # num_restarts = 5, picks the 5 best points and from each of those points calculates the slope, and uses that slope
        # to take a step uphill, recalculates slope, takes another step uphill and repeats until it find peak of that hill

        return new_path
    
    def run_optimization(self, n_iter, n_init = 10):
        # setup
        train_X, train_Y = self.intialise_search_space(n_init)
        bounds = self.get_optimizer_bounds()
        print(f"Initialised {n_init} starting paths")
        print(f"Starting best cost: {train_Y.min().item()}") # .item() strips the extra info that it had from pytorch
        # without item u would print something like tensor(15.500) while with item u would print 15.5

        # optimization loop
        for i in range(n_iter):
            # fit gp
            model, _ = self.fit_GP(train_X, train_Y)
            # _ stores the data returned by the function which u dont need

            # next path
            new_path_tensor = self.get_next_path(model, train_Y, bounds)
            # this has additional pytorch stuff which we dont need
            # ex it has uneccessary bracket around it like [[4.2 4.4 ]] instead of [4.2 4.4]
            # format path
            continuous_new_path = new_path_tensor.squeeze(0).detach().numpy()
            # squeeze(0) looks at dimension 0, the very 1st outermost bracket and if there is only one item inside it , it deletes that bracket
            # i.e makes the makes the 2D matrix [[4.2 8.9]] into a 1D list [4.2 8.9]
            # detach() removes extra stuff that pytorch stores 
            # if u dont do this, prog will eventually crash from running out of memory
            # numpy convert from tensor to numpy array

            # evaluate 
            rounded_path_np = np.round(continuous_new_path).astype(int)
            cost = self.objective_fn(rounded_path_np, self.grid, self.k, self.n) 

            # update memory
            continuous_tensor = torch.tensor(continuous_new_path, dtype=torch.float64).unsqueeze(0) 
            # unsqueeze(0) wraps path in an extra bracket so that pytorch recognises it as a standalone row and can add it to the  2D matrix
            train_X = torch.cat([train_X, continuous_tensor])

            cost_tensor = torch.tensor([cost], dtype=torch.float64).unsqueeze(-1) 
            # turns a single number [15.5] into a proper 2D column entry [[15.5]]
            train_Y = torch.cat([train_Y, cost_tensor])

            # show progress
            best_cost = train_Y.min().item()
            
            # Print the moving average
            window_size = 5
            if len(train_Y) >= window_size:
                # Grab the last 5 costs using negative indexing and calculate the mean
                moving_avg = train_Y[-window_size:].mean().item()
                print(f"Iteration {i+1}/{n_iter} - Current: {cost:.1f} | 5-Iter Avg: {moving_avg:.1f} | Best: {best_cost:.1f}")
            else:
                print(f"Iteration {i+1}/{n_iter} - Current: {cost:.1f} | Best: {best_cost:.1f}")

        return train_X, train_Y