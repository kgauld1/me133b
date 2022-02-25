
import numpy as np
import random
from hw5_utilities import Visualization, Robot

#  Define the Walls
w = ['xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
     'x               x               x               x',
     'x                x             x                x',
     'x                 x           x                 x',
     'x        xxxx      x         x                  x',
     'x        x   x      x       x                   x',
     'x        x    x      x     x      xxxxx         x',
     'x        x     x      x   x     xxx   xxx       x',
     'x        x      x      x x     xx       xx      x',
     'x        x       x      x      x         x      x',
     'x        x        x           xx         xx     x',
     'x        x        x           x           x     x',
     'x        x        x           x           x     x',
     'x        x        x           x           x     x',
     'x                 xx         xx           x     x',
     'x                  x         x                  x',
     'x                  xx       xx                  x',
     'x                   xxx   xxx                   x',
     'x                     xxxxx         x           x',
     'x                                   x          xx',
     'x                                   x         xxx',
     'x            x                      x        xxxx',
     'x           xxx                     x       xxxxx',
     'x          xxxxx                    x      xxxxxx',
     'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx']


walls = np.array([[1.0*(c == 'x') for c in s] for s in w])
rows  = np.size(walls, axis=0)
cols  = np.size(walls, axis=1)

# ----- Helper Functions ------

# Normalizes a grid of probabilistic locations (sum of all locations = 1)
def normalize(grid):
    s = np.sum(grid)
    if (s == 0.0):
        print("LOST ALL BELIEF.  SHOULD NOT HAPPEN.  STARTING OVER!!!!")
        grid = 1.0 - walls
        s    = np.sum(grid)
    return grid/s

# Get the parameters for any given part a-e of the question. Given in a tuple formatted
# (visual, robot, probCmd, probProximal)
def get_init(part):
    # Initialize the figure.
    visual = Visualization(walls)
    
    # "for fun" case
    robot = robot = Robot(walls, probCmd = 0.8, probProximal = [0.9, 0.6, 0.3])
    if part == 'a':
        robot = Robot(walls)
    elif part == 'b':
        robot = Robot(walls, row=12, col=26)
    elif part == 'c':
        robot = Robot(walls, row=12, col=26, probProximal = [0.9, 0.6, 0.3])
    elif part == 'd' or part == 'e':
        robot = Robot(walls, row=15, col=47, 
                      probCmd = 0.8, probProximal = [0.9, 0.6, 0.3])



    # Pick the algorithm assumptions:
    probCmd      = 1.0   # Part (a/b), (c), (d)
    if part == 'e':
        probCmd = 0.8    # Part (e)
    
    
    probProximal = [0.9, 0.6, 0.3]      # Part (c), (d), (e)
    if part == 'a' or part == 'b':
        probProximal = [1.0]                # Part (a/b)
        
    
    return (visual, robot, probCmd, probProximal)

# Define the possible status levels for each state.
WALL      = 0
UNKNOWN   = 1
ONDECK    = 2
PROCESSED = 3
PATH      = 4

# Define the constant START and GOAL positions
START = (5,  4)
GOAL  = (5, 12)


def computePrediction(bel, drow, dcol, probCmd = 1):
    # Prepare an empty prediction grid.
    prd = np.zeros((rows,cols))

    # Iterate over/determine the probability for all (non-wall) elements.
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            ncoord = (row+drow, col+dcol)
            ccoord = (row, col)
            # If the new position is not a wall, find probability of moving 
            # there, else do not change it (can't move into wall)
            prd[ncoord] += probCmd*bel[ccoord] if walls[ncoord] != 1 else 0
            
            # If the new position is a wall, then the probability of moving into the current
            # position is heightened, else lower by the probability of moving to the open position
            prd[ccoord] += (1-probCmd)*bel[ccoord] if walls[ncoord] != 1 else bel[ccoord]
            
    # Return the normalized prediction grid
    return normalize(prd)

def updateBelief(prior, probSensor, sensor):
    # If the sensor is on, then the next value is prior * P(sensor is on)
    # if it's off, then the value is prior * (1 - P(sensor is on))
    post = probSensor*prior if sensor else prior*(1-probSensor)

    # Normalize.
    return normalize(post)


def computeSensorProbability(drow, dcol, probProximal = [1.0]):
    # Prepare an empty probability grid.
    prob = np.zeros((rows, cols))

    # Pre-compute the probability for each grid element, knowing the
    # walls and the direction of the sensor.
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            if walls[row][col] == 1: continue
            for i in range(len(probProximal)):
                crow = drow*(i+1)+row
                ccol = dcol*(i+1)+col
                if not 0<=crow<rows or not 0<=ccol<cols:
                    break
                if walls[crow][ccol] == 1:
                    prob[row][col] = probProximal[i]
                    break
    
    # Return the computed grid. Not normalized since the sensor can have
    # a 100% chance of being on for 2 points in the grid at the same time
    return prob


def build_path(goal, pnodes):
    # Start the path at the end
    path = [goal]
    current = goal
    
    # While the current node has one before it, add the prior
    # node to the path then redo the check with this node
    while pnodes[current] != None:
        path.append(pnodes[current])
        current = pnodes[current]
    
    # Once the start node is reached, reverse the path (start -> goal) and return
    path.reverse()
    return path

def astar(start, goal, state, c_path):
    # Tracks which nodes are in ondeck and when they were added to ondeck
    ondeck = [start]
    # For any node, stores the node prior to it
    prior_node = {start:None}
    
    # While there is a node that has not been fully processed, process it
    while ondeck:
        # Sort ondeck by the path cost for each node, so the first entry
        # has the lowest path cost, then get the first entry 
        ondeck.sort(key=c_path)
        current = ondeck.pop(0)

        if current == goal:
            state[current] = PROCESSED
            # Return the path used to get to the goal and the state
            return (build_path(goal, prior_node), state)
        
        # Get the next nodes in each direction (up down left right)
        for i in (-1,1): # -1 for up/left, +1 for down/right
            for axis in (0, 1): # axis 0 for right/left, 1 for up/down
                
                # Find the next node in the specified direction
                nnode = (current[0] + (i if axis==1 else 0),
                         current[1] + (i if axis==0 else 0))
                
                # If the node has not been seen yet, add to ondeck and track
                # the node used to get to this one (the current node)
                if state[nnode] == UNKNOWN:
                    state[nnode] = ONDECK
                    ondeck.append(nnode)
                    prior_node[nnode] = current
        
        # Mark the node as processed, check if we've reached the goal, then continue
        state[current] = PROCESSED
    
    # If a path could not be found
    return (None, state)


def get_max_P(bel):
    maxlist = [(0,0)]
    for row in range(len(bel)):
        for col in range(len(bel[row])):
            if bel[row, col] > bel[maxlist[0]]:
                maxlist = [(row, col)]
            elif bel[row,col] == bel[maxlist[0]] and random.randint(0,2)==1:
                maxlist.append((row,col))
    random.shuffle(maxlist)
    return maxlist[0]

# 
#
#  Main Code
#
def main():
    (visual, robot, probCmd, probProximal) = get_init('e')

    # Pre-compute the probability grids for each sensor reading.
    probUp    = computeSensorProbability(-1,  0, probProximal)
    probRight = computeSensorProbability( 0,  1, probProximal)
    probDown  = computeSensorProbability( 1,  0, probProximal)
    probLeft  = computeSensorProbability( 0, -1, probProximal)

    # Start with a uniform belief grid.
    bel = 1.0 - walls
    bel = (1/np.sum(bel)) * bel
    fails = 0

    dirs = [(0,1), (0,-1), (1,0), (-1,0)]
    # Loop continually.
    while True:
        # Show the current belief.  Also show the actual position.
        pos = get_max_P(bel)
        visual.Show(bel, robot.Position(), pos)
        # Get the command key to determine the direction.
        
        c_path_1 = lambda x: 1*(abs(x[0]- GOAL[0]) + abs(x[1]- GOAL[1])) +\
                    1*(abs(x[0]-START[0]) + abs(x[1]-START[1]))
        state = np.array([[1-1.0*(c == 'x') for c in s] for s in w])
        path, S = astar(pos, GOAL, state, c_path_1)

        if fails >= 50 and path==None:
            print("Could not find path")
            break
        if path != None and path[0] == GOAL:
            print("Goal found!")
            break

        if path != None:
            d = (path[1][0] - pos[0], path[1][1] - pos[1])
        else:
            fails += 1
            d = dirs[random.randint(0,3)]
        robot.Command(d[0], d[1])
        # Move the robot in the simulation.
        # robot.Command(drow, dcol)


        # Compute a prediction.
        prd = computePrediction(bel, d[0], d[1], probCmd)
        #visual.Show(prd)
        #input("Showing the prediction")


        # Correct the prediction/execute the measurement update.
        bel = prd
        bel = updateBelief(bel, probUp,    robot.Sensor(-1,  0))
        bel = updateBelief(bel, probRight, robot.Sensor( 0,  1))
        bel = updateBelief(bel, probDown,  robot.Sensor( 1,  0))
        bel = updateBelief(bel, probLeft,  robot.Sensor( 0, -1))


if __name__== "__main__":
    main()
