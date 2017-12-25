# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    #using helper function for setup
    path = []
    OPEN = __setup(problem, dfs)

    while ~(OPEN.isEmpty()):
        current_node = OPEN.pop()

        # finds pacman's current (x,y) position
        current_pos = current_node[-1][0]

        if problem.isGoalState(current_pos):
            # calls helper function with option dir (directions)
            path = __getElements(current_node, "dir")
            return path

        successors = problem.getSuccessors(current_pos)
        visited = __getElements(current_node, "pos")
        for successor_state in successors:
            if successor_state[0] not in visited:
                new_node = list(current_node)
                new_node.append(successor_state)
                OPEN.push(new_node)

    return path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #using helper function for setup    
    path = []
    OPEN = __setup(problem, bfs)

    visited = __setup(problem, "visited")

    while ~(OPEN.isEmpty()):

        current_node = OPEN.pop()
        # finds pacman's current (x,y) position
        current_pos = current_node[-1][0]

        if __cost(current_node) <= visited[current_pos]:
            if problem.isGoalState(current_pos):
                if problem.isGoalState(current_pos):
                    path = __getElements(current_node, "dir")
                    return path

            sucessors = problem.getSuccessors(current_pos)
            for successor_state in sucessors:
            	#cost of new successor state
                newPathCost = __cost(current_node) + successor_state[2]

                # if the pos has not been visited before, or the cost is lower; add to OPEN
                if (successor_state[0] not in visited) or (newPathCost < visited[successor_state[0]]):
                    new_node = list(current_node)
                    new_node.append(successor_state)
                    OPEN.push(new_node)
                    visited[successor_state[0]] = newPathCost
    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    #using helper function for setup
    path = []
    OPEN = __setup(problem, ucs)
    visited = __setup(problem, "visited")

    while ~(OPEN.isEmpty()):
        current_node = OPEN.pop()
        # finds pacman's current (x,y) position
        current_pos = current_node[-1][0]

        if __cost(current_node) <= visited[current_pos]:
            
            if problem.isGoalState(current_pos):
                path = __getElements(current_node, "dir")
                return path
            
            #obtain sucessors at current pos
            sucessors = problem.getSuccessors(current_pos)
            for successor_state in sucessors:
            	#cost calculated from cost of current node + cost @ successor pos
                newPathCost = __cost(current_node) + successor_state[2]
                
                #if this pos has not been visited before, or the cost is lower, we add to OPEN
                if (successor_state[0] not in visited) or (newPathCost < visited[successor_state[0]]):
                    new_node = list(current_node)
                    new_node.append(successor_state)
                    OPEN.update(new_node, newPathCost)
                    visited[successor_state[0]] = newPathCost
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    #using helper function for setup
    path = []
    OPEN = __setup(problem, astar)
    visited = __setup(problem, "visitedh", heuristic)    

    while ~(OPEN.isEmpty()):
        current_node = OPEN.pop()
        # finds pacman's current (x,y) position
        current_pos = current_node[-1][0]
        
        # calculates fcost from cost of node + heuristics function
        fcost = __cost(current_node) + heuristic(current_pos, problem)
        
        if fcost <= visited[current_pos]:
            if problem.isGoalState(current_pos):
                path = __getElements(current_node, "dir")
                return path    

            successors = problem.getSuccessors(current_pos)
            for successor_state in successors:

            	#cost at new successor state
                newPathCost = __cost(current_node) + successor_state[2] + heuristic(successor_state[0], problem)
                
                #if this pos has not been visited before, or the cost is lower, we add to OPEN    
                if (successor_state[0] not in visited) or (newPathCost < visited[successor_state[0]]):
                    new_node = list(current_node)
                    new_node.append(successor_state)
                    OPEN.update(new_node, newPathCost)
                    visited[successor_state[0]] = newPathCost
    return path

#################################################################################
#                                                                               # 
#                               Helper Functions                                # 
#                                                                               #
#################################################################################

def __setup(problem, stype, heuristic=None):

	"""	Helper function that does setup for each search type """
    
	if stype == dfs:
		OPEN = util.Stack()
		OPEN.push([(problem.getStartState(), None, 0)])
		return OPEN

	elif stype == bfs:
		OPEN = util.Queue()
		OPEN.push([(problem.getStartState(), None, 0)])
		return OPEN

	elif stype == ucs:
		OPEN = util.PriorityQueue()
		OPEN.push([(problem.getStartState(), None,0)], 0) 
		return OPEN

	elif stype == astar:
		OPEN = util.PriorityQueue()
		OPEN.push([(problem.getStartState(), None,0)], 0) 
		return OPEN

	elif stype == "visited":
		visited = {problem.getStartState(): 0}
		return visited

	elif stype == "visitedh":
		H = heuristic(problem.getStartState(), problem)
		visited = {problem.getStartState(): H}
		return visited

	else:
		print  "Incorrect stype = {dfs, bfs, ucs, astar, visited, visitedh}"
		return None


def __getElements(node, etype):

	"""	Helper function that returns all of a given type from a node """


	elements = []

	if etype == "pos":
		for state in node:
			elements.append(state[0])

	elif etype == "dir":
		for state in node:
			if state[1] is not None:
				elements.append(state[1])

	elif etype == "cost": 
		for state in node:
			elements.append(state[2])

	return elements


def __cost(node):

	"""	Finds the cost from a given node """

	return sum(__getElements(node,'cost'))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
