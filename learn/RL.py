import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
import matplotlib.pyplot as plt
from base.base import Actions

# TODO 1: mode ==> Carying = dict() diamond number = key | score = value
# TODO 2: function reward need updated
# TODO 3: set target 
# TODO 4:   

# Gray scale marks for cells
visited_mark = 0.9
diamond_mark = 0.65
home_mark = 0.75
agent_mark = 0.5

# Actions
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actions_dict = {
    LEFT:  Actions.LEFT,
    UP:    Actions.UP,
    RIGHT: Actions.RIGHT,
    DOWN:  Actions.DOWN,
}

diamond_dict = {
    0:2,
    1:5,
    2:3,
    3:1,
    4:10
}

mode_dict = {
    'start' :   'start',
    'valid' :   'valid',
    'invalid':  'invalid',
    'blocked' : 'blocked',
    'carying':  diamond_dict,
    'diamond':  diamond_dict,
    'home':     diamond_dict
}

num_actions = len(actions_dict)

class TdfMaze(object):
    """
    Tour De diamonds maze object
    maze: a 2d Numpy array of floats between 0 to 1:
        1.00 - a free cell
        0.65 - diamond cell
        0.75 - home cell
        0.50 - agent cell
        0.00 - an occupied cell
    agent: (row, col) initial agent position (defaults to (0,0))
    diamonds: list of cells occupied by diamonds
    """
    #! Set agent location
    #! list of diamonds
    #TODO Need update
    def __init__(self, maze, diamonds, homes, agent=(0,0), target=None):
        #* Set _maze with np object
        self._maze = np.array(maze)
        
        #* set diamond location and home location
        self._diamonds = set(diamonds)
        self._homes = set(homes)
        
        #* Set map size
        nrows, ncols = self._maze.shape
        
        #! set target ==> we Need it?
        if target is None:
            self.target = (nrows-1, ncols-1)   # default target cell where the agent to deliver the "diamonds"
        
        #* Set free cells
        self.free_cells = set((r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0)
        
        #* Remove target location from free cells 
        self.free_cells.discard(self.target)
        
        #* Remove diamonds location from free cells
        self.free_cells -= self._diamonds
                
        #? check error ==> We need it?
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not agent in self.free_cells:
            raise Exception("Invalid agent Location: must sit on a free cell")
        
        
        self.reset(agent)

    #TODO Need Update
    def reset(self, agent=(0,0)):
        #* Set agent location
        self.agent = agent
        
        #* Copy from _maze
        self.maze = np.copy(self._maze)
        
        #* Set diamonds from _diamonds and homes from _homes
        self.diamonds = set(self._diamonds)
        self.homes = set(self._homes)

        #* Set map size
        nrows, ncols = self.maze.shape
        
        #* Set agent location 
        row, col = agent
        
        #* Mark the agent location
        self.maze[row, col] = agent_mark
        
        #* Initial the state
        self.state = ((row, col),(mode_dict.get('start'),''))
        
        #! WHAT?
        self.diameter = np.sqrt(self.maze.size)
        
        #? Set visited home ==> We need it?
        self.visited = dict(((r,c),0) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0)
        
        #* Total Reward
        self.total_reward = 0
        
        #* Set for end game
        self.min_reward = -0.5 * self.maze.size
        
        #! Dict for our rewards ==> we should update it
        self.reward = {
            'blocked':  self.min_reward,
            # 'diamond':     1.0/len(self._diamonds),
            'invalid': -4.0/self.diameter,
            'valid':   -1.0/self.maze.size,
            0 : mode_dict['carying'].get(0),
            1 : mode_dict['carying'].get(1),
            2 : mode_dict['carying'].get(2),
            3 : mode_dict['carying'].get(3),
            4 : mode_dict['carying'].get(4)
        }

    def act(self, action):
        #* Update state
        self.update_state(action)
        
        #* Calculate reward
        reward = self.get_reward()
        
        #* Calculate total reward
        self.total_reward += reward
        
        #* Set game Status == win? == lose? == ongoing?
        status = self.game_status()
        
        #? draw and observe the env ==> We need it?
        env_state = self.observe()
        
        return env_state, reward, status

    #TODO Need Update
    def get_reward(self):
        
        agent, mode = self.state
        fmode , smode = mode
        #! Update it
        if agent == self.target:
            return 1.0 - len(self.diamonds) / len(self._diamonds)
        
        # elif agent in self.diamonds:
        #     return self.reward['diamond']
        
        
        #* Check for blocked :
        if fmode == mode_dict.get('blocked'):
            return self.reward['blocked']
        
        
        #* Check for valid and home cell:
        elif fmode == mode_dict.get('valid') and smode == mode_dict['home'].get(0):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['home'].get(1):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['home'].get(2):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['home'].get(3):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['home'].get(4):
            pass
        
        
        #* Check for valid and diamond cell
        elif fmode == mode_dict.get('valid') and smode == mode_dict['diamond'].get(0):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['diamond'].get(1):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['diamond'].get(2):
            pass

        elif fmode == mode_dict.get('valid') and smode == mode_dict['diamond'].get(3):
            pass

        elif fmode == mode_dict.get('valid') and smode == mode_dict['diamond'].get(4):
            pass
        
        
        #* Check for valid and carying diamond
        elif fmode == mode_dict.get('valid') and smode == mode_dict['carying'].get(0):
            pass

        elif fmode == mode_dict.get('valid') and smode == mode_dict['carying'].get(1):
            pass   
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['carying'].get(2):
            pass
        
        elif fmode == mode_dict.get('valid') and smode == mode_dict['carying'].get(3):
            pass

        elif fmode == mode_dict.get('valid') and smode == mode_dict['carying'].get(4):
            pass
        
        #* Check for invalid :
        elif fmode == mode_dict.get('invalid'):
            return self.reward['invalid']
        
        # elif mode == mode_dict.get('valid'):
        #     return self.reward['valid'] #* (1 + 0.1*self.visited[agent] ** 2)

    #TODO Need Update
    def update_state(self, action):
        
        #* Set map size
        nrows, ncols = self.maze.shape
        
        #* Set agent location and mode | mode == 'start' or 'blocked' or 'valid' or 'invalid'
        (nrow, ncol), nmode = agent, mode = self.state

        #? Set visited list ==> we Need it?
        if self.maze[agent] > 0.0:
            self.visited[agent] += 1  # mark visited cell
        
        #* Remove diamonds that we collected
        # if agent in self.diamonds:
        #     self.diamonds.remove(agent)   

        

        #* set action ==> valid_action == number of list for each move (0=left ... 3=down) | it can be empty list
        valid_actions = self.valid_actions()

        #* there is no action 
        if not valid_actions:
            nmode = mode_dict.get('blocked') 
        
        #* list of valid_action is not empty
        elif action in valid_actions:
            nmode = mode_dict.get('valid')
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in agent position
            nmode = mode_dict.get('invalid')

        #* Update state with new location for agent and mode
        agent = (nrow, ncol)
        self.state = (agent, nmode)

    def game_status(self):
        #* Check for end game
        if self.total_reward < self.min_reward:
            return 'lose'
        
        #* if agent in target cell and diamond list is empty we win != we lose
        agent, mode = self.state
        if agent == self.target:
            if len(self.diamonds) == 0:
                return 'win'
            else:
                return 'lose'

        return 'ongoing'

    def observe(self):
        canvas = self.draw_env()
        env_state = canvas.reshape((1, -1))
        return env_state

    #? Draw envirmonet ==> we need it?
    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the diamonds
        for r,c in self.diamonds:
            canvas[r,c] = diamond_mark
        # draw the agent
        agent, mode = self.state
        canvas[agent] = agent_mark
        return canvas

    #* Generate valid action ==> return list of 
    def valid_actions(self, cell=None):
        if cell is None:
            (row, col), mode = self.state
        else:
            row, col = cell
        actions = [LEFT, UP, RIGHT, DOWN]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(UP)
        elif row == nrows-1:
            actions.remove(DOWN)

        if col == 0:
            actions.remove(LEFT)
        elif col == ncols-1:
            actions.remove(RIGHT)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(UP)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(DOWN)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(LEFT)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(RIGHT)

        return actions

#------------ Experience Class --------------

class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.97):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [env_state, action, reward, next_env_state, game_over]
        # memory[i] = episode
        # env_state == flattened 1d maze cells info, including agent cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, env_state):
        return self.model.predict(env_state)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # env_state 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            env_state, action, reward, next_env_state, game_over = self.memory[j]
            inputs[i] = env_state
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep (quote by Eder Santana)
            targets[i] = self.predict(env_state)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(next_env_state))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

#------------ Q-training Class --------------

class Qtraining(object):
    def __init__(self, model, env, **opt):
        self.model = model  # Nueral Network Model
        self.env = env  # Environment (Tour De diamonds maze object)
        self.n_epoch = opt.get('n_epoch', 1000)  # Number of epochs to run
        self.max_memory = opt.get('max_memory', 4*self.env.maze.size)  # Max memory for experiences
        self.data_size = opt.get('data_size', int(0.75*self.env.maze.size))  # Data samples from experience replay
        self.agent_cells = opt.get('agent_cells', [(0,0)])  # Starting cells for the agent
        self.weights_file = opt.get('weights_file', "")  # Keras model weights file
        self.name = opt.get('name', 'model')  # Name for saving weights and json files

        self.win_count = 0
        # If you want to continue training from a previous model,
        # just supply the h5 file name to weights_file option
        if self.weights_file:
            print("loading weights from file: %s" % (self.weights_file,))
            self.model.load_weights(self.weights_file)

        if self.agent_cells == 'all':
            self.agent_cells = self.env.free_cells

        # Initialize experience replay object
        self.experience = Experience(self.model, max_memory=self.max_memory)

    def train(self):
        start_time = datetime.datetime.now()
        self.seconds = 0
        self.win_count = 0
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            self.loss = 0.0
            agent = random.choice(self.agent_cells)
            self.env.reset(agent)
            game_over = False
            # get initial env_state (1d flattened canvas)
            self.env_state = self.env.observe()
            self.n_episodes = 0
            while not game_over:
                game_over = self.play()

            dt = datetime.datetime.now() - start_time
            self.seconds = dt.total_seconds()
            t = format_time(self.seconds)
            fmt = "Epoch: {:3d}/{:d} | Loss: {:.4f} | Episodes: {:4d} | Wins: {:2d} | diamonds: {:d} | e: {:.3f} | time: {}"
            print(fmt.format(epoch, self.n_epoch-1, self.loss, self.n_episodes, self.win_count, len(self.env.diamonds), self.epsilon(), t))
            if self.win_count > 2:
                if self.completion_check():
                    print("Completed training at epoch: %d" % (epoch,))
                    break

    def play(self):
        action = self.action()
        prev_env_state = self.env_state
        self.env_state, reward, game_status = self.env.act(action)
        if game_status == 'win':
            self.win_count += 1
            game_over = True
        elif game_status == 'lose':
            game_over = True
        else:
            game_over = False

        # Store episode (experience)
        episode = [prev_env_state, action, reward, self.env_state, game_over]
        self.experience.remember(episode)
        self.n_episodes += 1

        # Train model
        inputs, targets = self.experience.get_data(data_size=self.data_size)
        epochs = int(self.env.diameter)
        h = self.model.fit(
            inputs,
            targets,
            epochs = epochs,
            batch_size=16,
            verbose=0,
        )
        self.loss = self.model.evaluate(inputs, targets, verbose=0)
        return game_over

    def run_game(self, agent):
        self.env.reset(agent)
        env_state = self.env.observe()
        while True:
            # get next action
            q = self.model.predict(env_state)
            action = np.argmax(q[0])
            prev_env_state = env_state
            # apply action, get rewards and new state
            env_state, reward, game_status = self.env.act(action)
            if game_status == 'win':
                return True
            elif game_status == 'lose':
                return False

    def action(self):
        # Get next action
        valid_actions = self.env.valid_actions()
        if not valid_actions:
            action = None
        elif np.random.rand() < self.epsilon():
            action = random.choice(valid_actions)
        else:
            q = self.experience.predict(self.env_state)
            action = np.argmax(q)
        return action

    def epsilon(self):
        n = self.win_count
        top = 0.80
        bottom = 0.08
        if n<10:
            e = bottom + (top - bottom) / (1 + 0.1 * n**0.5)
        else:
            e = bottom
        return e
    
    def completion_check(self):
        for agent in self.agent_cells:
            if not self.run_game(agent):
                return False
        return True

    def save(self, name=""):
        # Save trained model weights and architecture, this will be used by the visualization code
        if not name:
            name = self.name
        h5file = '%s.h5' % (name,)
        json_file = '%s.json' % (name,)
        self.model.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(self.model.to_json(), outfile)
        t = format_time(self.seconds)
        print('files: %s, %s' % (h5file, json_file))
        print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (self.epoch, self.max_memory, self.data_size, t))

#-----------------------------------

def build_model(env, **opt):
    loss = opt.get('loss', 'mse')
    a = opt.get('alpha', 0.24)
    model = Sequential()
    esize = env.maze.size
    model.add(Dense(esize, input_shape=(esize,)))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(esize))
    model.add(LeakyReLU(alpha=a))
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

def show_env(env, fname=None):
    plt.grid('on')
    n = env.maze.shape[0]
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, n, 1))
    ax.set_yticks(np.arange(0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(env.maze)
    for cell in env.visited:
        if env.visited[cell]:
            canvas[cell] = visited_mark
    for cell in env.diamonds:
        canvas[cell] = diamond_mark
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    if fname:
        plt.savefig(fname)
    return img

def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

