# this file is for making the EHWP grid with the infected RPUs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as ListedColorMap

class EHWP_Grid:
    def __init__(self, n, num_ht): # to make a square matrix of size n*n
        self.n = n
        self.num_ht = num_ht
        #make and store grid internally so its consistent
        self.grid = self.make_grid()

    def make_grid(self):
        matrix = np.zeros(self.n*self.n) # n*n matrix of zeros
        indices = np.random.choice(self.n*self.n, self.num_ht, replace = False)
        # picks infected_rpus no. of nums between 0 to n^2-1
        # imagines n*n matrix as a flat matrix of n^2 slots
        # replace = false makes sure same position isnt picked twice
        matrix[indices] = 1 #assigns 1 to picked positions
        matrix = matrix.reshape(self.n, self.n) # reshapes to n*n
        return matrix
    
    def display_grid(self):
        fig, ax = plt.subplots(figsize=(8,8)) #fig is the blank canvas
        # this is the square area where the grid is drawn

        cmap = ListedColorMap(['green', 'red'])
        #listed color map maps 0 to green and 1 to red

        ax.imshow(self.grid,cmap = cmap)
        # looks at the matrix like a picture, colors 0 to green and 1 to red
        ticks = np.arange(-0.5, self.n, 1)
        #draw lines manually
        ax.hlines(ticks, -0.5, self.n-0.5, color = 'black', linewidth=7)
        ax.vlines(ticks, -0.5, self.n-0.5, color = 'black', linewidth=7)
        # by default matplotlib put the ticks(little marks for numbers) at the centre of the cells
        # if we draw lines there it would right through the cell, so shift by 0.5
        # np.arange(..) makes array like [-0.5,0.5,1.5 and so on]

        ax.tick_params(which = 'both', bottom = False, left = False, labelbottom = False, labelleft = False)
        # both applies to both axes
        # bottom,left removes tiny little dashes that stick out from axis
        # labelbottom hides actual numbers from sides of plot

        plt.title(f"EHWP of size {self.n}x{self.n} infected with {self.num_ht} HTs", fontsize = 20, pad = 25) 
        # increase font size and move its position slightly above
        plt.show(block = False) # makes it non blocking and goes back to running script
        plt.pause(0.1) # gives matplotlib time to draw graph by not transferring control immediately to the main script

    