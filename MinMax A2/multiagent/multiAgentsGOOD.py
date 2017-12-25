# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        remainingFood = newFood.asList()
        curFood = currentGameState.getFood().asList()

        evaluation = 0
        FoodPos = []
        GhostPos = []

        for food in remainingFood:
          FoodPos.append(util.manhattanDistance(newPos, food))
        for ghost in newGhostStates:
          GhostPos.append(util.manhattanDistance(newPos, ghost.configuration.getPosition()))

        if FoodPos:
            closest = min(FoodPos)

            if newPos not in curFood:
                evaluation -= closest

        if GhostPos:
            closest = min(GhostPos)

            allScared = True

            for ghostStatus in newScaredTimes:
                if ghostStatus == 0:
                  allScared = False
                  break

            if allScared == False:
              if closest < 2:
                evaluation = -9999

        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        typeA = "minimax"
        pruning = False
        turn = 0;
        bestMove, bestVal = maxAlgo(self, gameState, turn, typeA, pruning)
        return bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        typeA = "minimax"
        pruning = True
        alpha = -float("inf") 
        beta = float("inf")
        turn = 0;
        bestMove, bestVal = maxAlgo(self, gameState, turn, typeA, pruning, alpha, beta)
        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        typeA = "expectimax"
        pruning = False
        turn = 0;
        bestMove, bestVal = maxAlgo(self, gameState, turn, typeA, pruning)
        return bestMove

def betterEvaluationFunction(currentGameState):
  """
  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
  evaluation function (question 5).

  From Piazza:

  "There is only one ghost for this question." 

  Assuming that the grading is based on one ghost

  If the state is a isLose state, return back lowest score possible
  If the ghost is near the player, the evaluation score is set to a very low score
  since we do not want to be near the ghost before we have finished eating the dots

  If the state is a isWin state, return back the highest score possible

  For all food on the game board, we give a score based on the amount of food left on the board.
  The lower the amount of food, the lower the amount of points that is taken away from the eval score.

  We also based evaluation on how far the current position is to the nearest food, however that is scaled lower
  then the amount of food left on the board. This is because we do want to move closer to the food, but we dont
  want to get stuck. Instead, we prioritize having less food on the board as a higher score, and winning the game
  as the highest score.

  """
  "*** YOUR CODE HERE ***"
  curState = currentGameState
  curPos = currentGameState.getPacmanPosition()
  curFood = currentGameState.getFood().asList()
  curGhostStates = currentGameState.getGhostStates()
  curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
  curCapsule = currentGameState.getFood()

  evaluation = 0
  if curState.isLose():
    return -float("inf")
  if curState.isWin():
    return float("inf")

  FoodPos = []
  GhostPos = []
  for food in curFood:
    FoodPos.append(util.manhattanDistance(curPos, food))
  for ghost in curGhostStates:
    GhostPos.append(util.manhattanDistance(curPos, ghost.configuration.getPosition()))

  if FoodPos:
    closestFood = min(FoodPos)
    evaluation -= closestFood * 10
    evaluation -= len(curFood) * 100
  else:
    #Last remaining food, increase scaling to quickly finish game
    evaluation -= len(curFood) * 500


  closestGhost = min(GhostPos)
  if closestGhost < 2:
    # if we are too close to the ghost we will lose, so set the eval score to a low one
    evaluation = -99999

  return evaluation 

# Abbreviation
better = betterEvaluationFunction



def maxAlgo(maxAgent, state, turn, typeA, pruning, alpha=None, beta=None):

  bestAction = Directions.STOP
  agent_num = getAgentN(turn,state.getNumAgents())
  value = initValue(agent_num, typeA)
  
  # Base case
  if finishedState(turn, state, maxAgent):
    return bestAction, maxAgent.evaluationFunction(state)

  for action in state.getLegalActions(agent_num):

    nextState = state.generateSuccessor(agent_num, action)

    # for alpha beta pruning
    if pruning:
      nextAction, nextVal = maxAlgo(maxAgent, nextState, turn+1, typeA, pruning, alpha, beta)
      if agent_num == 0:
        if value < nextVal:
          value = nextVal
          bestAction = action
        if value >= beta:
          return bestAction, value
        alpha = float(max(alpha, value))
      else:
        if value > nextVal:
          value = nextVal
          bestAction = action
        if value <= alpha:
          return bestAction, value
        beta = float(min(beta, value))
    else:
      nextAction, nextVal = maxAlgo(maxAgent, nextState, turn+1, typeA, pruning)
      if agent_num == 0:
        if value < nextVal:
          value = nextVal
          bestAction = action
      else: #agent_num != 0 
        if typeA == "minimax" and value > nextVal:
          value = nextVal
          bestAction = action
        elif typeA == "expectimax":
          value += float(nextVal / len(state.getLegalActions(agent_num)))
  return bestAction, value



def initValue(agent_num, typeA):

  if agent_num == 0:
    return -float("inf")
  else:
    if typeA == "minimax":
      return float("inf")
    else:
      return 0



def getAgentN(turn, total_agent):

  return (turn % total_agent)



def finishedState(turn, state, maxAgent):

  maxDepth = maxAgent.depth * state.getNumAgents()

  if state.isWin():
    return True
  elif state.isLose():
    return True
  elif turn >= maxDepth:
    return True
  else:
    return False