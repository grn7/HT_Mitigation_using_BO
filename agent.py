import torch 
import torch.nn as nn
import random
from collections import deque
import numpy as np

class QNetwork(nn.Module): # neural net module
    def __init__(self, n):
        super(QNetwork, self).__init__()
        # QNetwork is the child class and nn.Module is the parent class 
        # super tells to talk to parent class
        # init runs all the background setup needed , like creating empty 
        # dictionaries needed to track weights, biases etc

        # dimension of input layer
        input_dim = 2 * (n ** 2) + 1 # grid (n^2) + visited map (n^2) + 1
        # output layer - one q value for every possible coord
        output_dim = n ** 2
        # hidden layers : 2 layers will 128 neurons each
        hidden_dim = 128

        # the actual stuff
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # sequential automatically hands data to first layer, takes results, hands to 2nd and so on
        # nn.Linear represents the fully connected layer of artificial neurons, which contains all underlying
        # weights and biases that PyTorch will adjust during training
        # ReLU is the activation function

    def forward(self, x): # x is the input data, in our case x is the stae of our env currently
        # passes the state through network to predict Q-values
        return self.network(x) # basically giving x as input to the network we made above 
        # and the output is q values 

# Dual network
class DoubleDQNAgent:
    def __init__(self, n, learning_rate = 1e-3):
        self.n = n
        # use cuda 12.8 gpu is available , else fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # online network - active learner that picks actions
        self.online_net = QNetwork(n).to(self.device)
        # target network - stable reference for evaluating actions
        self.target_net = QNetwork(n).to(self.device)

        # initially both networks must start with the same weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        # pytorch deletes the target net's old math and overwrites with online net's math 
        # put target network in evaluation mode , i.e it doesnt train or calculate gradients
        self.target_net.eval()
        # standard optimizer to update online network 
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr = learning_rate)
        # adam is adaptive moment estimation , most popular algo to train neural nets

        # epsilon variables for exploration/exploitation
        self.epsilon = 1.0 # start with 1 i.e completely random exploration
        self.epsilon_decay = 0.995 # decay- multiply by 0.995 after every episode 
        self.epsilon_min = 0.05 # minimum - ensure some curiosity remains

        self.learn_step_counter = 0


    def update_target_network(self):
        # copy weights from online to target net once every 1000 steps or so
        # prevents agent from overestimating rewards 
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, state, valid_mask):
        # determine next action using epsilon greedy logic and action masking
        valid_indices = np.where(valid_mask)[0] # returns list of all indices that are true( safe to move)
        # where hands a tuple , but we need only the first element

        # if trapped with no valid moves return 0, environment should ideally flag 'done' before this happens
        if len(valid_indices) == 0:
            return 0

        if random.random() < self.epsilon:
            # exploratio - pick random action, only from valid list
            action = random.choice(valid_indices)
        else:
            # exploitation - pass state through online net to find highest q - value
            # convert numpy state to pytorch tensor and send to gpu
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            # convert to float, unsqueeze adds an extra bracket outside making a flat list into a 2D matrix as neural nets process this format
            # send it to device to that data and neural net on same piece of hardware

            # get predictions without tracking gradients (saves memory/time during inference)
            with torch.no_grad(): # tells pytorch to not track the math, just run data through the network as fast as possible and give answer
                q_values = self.online_net(state_tensor).squeeze(0) # removes the extra outermost bracket

            # convert back to numpy so we can easily apply mask
            q_values_np = q_values.cpu().numpy() # cpu copies data from gpu and brings to cpu

            # action masking - find all 'false' indices and set their q-value to -infinity
            # so that neural net never picks an invalid move 
            q_values_np[~valid_mask] = -float('inf')

            # pick action index with highest remaining q-val
            action = int(np.argmax(q_values_np))

        return action

    def decay_epsilon(self):
        # call this at end of every episode
        # multiplies epsilon , but use max to ensure it never drops below 0.05
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, memory_buffer, batch_size=64, gamma=0.99):
        # the core double dqn learning algo
        # 1. data prep and batching
        # only learn if we have enough experiences
        if len(memory_buffer) < batch_size:
            return

        # random sampling - draw the batch
        states, actions, rewards, next_states, dones = memory_buffer.sample(batch_size)

        # move everything to pytorch tensors on the device 
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        # unsqueeze reshapes from [64] to [64,1] so matrix math works correctly ( 64 rows, 1 column)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        # convert booleans to 1.0 or 0.0 floats 
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # get q values the online net predicted for the states we were in
        # gather() pulls out only the q values for the specific action the agent actually took
        current_q_values = self.online_net(states_t).gather(1,actions_t)
        # since we have 4 actions up,down, left and right, self.online_net(states_t)
        # looks like this - state 1 [a, b, c, d] 
        #                   state 2 [e, f, g, h] all are floats btw 
        # 1 tells it to search in the column direction, actions_t is 64x1 and looks like [[2], [1], ..]
        # for 2 it picks float at index 2, for 1 picks float at index 1 and so on

        # 2. double DQN logic 
        # we use torch.no_grad coz we only evaluating not training these specific steps
        with torch.no_grad():
            # step A (selection): pass next_state through online net to find index of highest q-value
            best_next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            # argmax grabs index of highest score. dim=1 tells pytorch to scan horizontally
            # keepdim maintains the 2D matrix instead of making it a flat list

            # step B (evaluation): pass next stae through target net but only look at the value for the action chosen in step A
            next_q_values = self.target_net(next_states_t).gather(1, best_next_actions)

        # 3. Bellman eq and loss
        # if dones_t is 1(true), 1- dones becomes 0 meaning there are no future rewards coz path ended
        target_q_values = rewards_t + (gamma*next_q_values*(1 - dones_t))

        # loss fn- mean square error(mse)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 4. optimization and sync
        # update online network
        self.optimizer.zero_grad() # clear old gradients from previous step
        loss.backward() # backpropogate the error and calc gradient
        self.optimizer.step() # nudge weights of online net in right direction

        # target sync- keep track of steps and copy weights every 100 steps
        self.learn_step_counter += 1
        if self.learn_step_counter % 100 == 0:
            self.update_target_network()


# build the replay buffer
class ReplayBuffer:
    def __init__(self, capacity = 10000, batch_size = 64):
        # make a double ended queue(deque) with a max length . when the buffer hits max lengthm adding new one 
        # automatically pushes oldest one out
        self.buffer = deque(maxlen = capacity)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        # packages current transition into single tuple.
        # it represents a single memory/ experience of the agent
        experience = (state, action, reward, next_state, done)
        # add new experience tuple 
        self.buffer.append(experience)

    def sample(self, batch_size):
        # randomly draw a batch of batch_size experiences
        # random.sample picks unique elements from the buffer without replacement
        batch = random.sample(self.buffer, batch_size)

        # zip(*batch) unzips a list of tuples. takes [(s1,a1,r1..), (s2,a2,r2...)]
        # and groups it by column like (states,actions, rewards...) making it easier to convert to pytorch tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        # helper to check how full the buffer is 
        # we need this as we cant sample a batch of 64 until we have 64 items atleast
        return len(self.buffer)