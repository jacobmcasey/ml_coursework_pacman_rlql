# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
import math

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.pacman_position = state.getPacmanPosition()
        self.ghost_positions = state.getGhostPositions()
        self.food_positions = state.getFood().asList()
        self.capsule_positions = state.getCapsules()

    def __hash__(self): 
        return hash((self.pacman_position, tuple(self.ghost_positions), tuple(self.food_positions), tuple(self.capsule_positions)))

    def __eq__(self, other):
        if isinstance(other, GameStateFeatures):
            return (
            self.pacman_position == other.pacman_position and
            self.ghost_positions == other.ghost_positions and
            self.food_positions == other.food_positions and
            self.capsule_positions == other.capsule_positions
        )
        else:
            return False

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 100):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.qTable = {}
        self.visitCountTable = {}

        # Initialize previous state and action
        self.prevState = None
        self.prevAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        if endState.isWin():
            # Pacman reached the goal state (ate all the food pellets)
            return 1
        elif endState.isLose():
            # Pacman lost the game (collided with a ghost)
            return -1
        
        return 0

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """

        # Get the state/action Q Value, if not present, return default 0 value.
        qValue = self.qTable.get((state, action), 0.0)
        return qValue

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # We need to check each action for the given state. And find which action gives the greatest Q value

        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        max = 0
        for action in legal:
            q_value = self.getQValue(state, action)
            if q_value > max:
                max = q_value
        return max

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Perform an update step of the qtable
        qValue = self.getQValue(state, action)

        # Compute the maximum Q-value max(Q(s_t+1,a) component
        maxNextStateQValue = self.maxQValue(nextState)

        # Get the count for the given state and action
        count = self.getCount(state, action)

        # Compute the exploration function for the count
        explorationFn = self.explorationFn(qValue, count)

        newQValue = qValue + self.getAlpha() * (self.computeReward(state,nextState) + self.getGamma() * maxNextStateQValue - qValue) * explorationFn

        # Update Q-value for (state, action) pair in the Q-table
        self.qTable[(state, action)] = newQValue

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # Get the count for state/action pair and add 1 to it. 
        self.visitCountTable[(state, action)] = self.visitCountTable.get((state, action), 0) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.visitCountTable.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # Choose a value for c, which controls the amount of exploration.
        # A higher value of c will lead to more exploration.
        c = 3.0

        # Add a small constant to avoid taking the logarithm of zero
        u = 1e-3

        # Compute the exploration bonus using the UCB1 formula.
        explorationBonus = c * math.sqrt(math.log(sum(self.visitCountTable.values()) + 4) / (counts + 1))

        return utility + explorationBonus

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """

        action = None
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Set the probability of exploration
        epsilon = self.epsilon

        # With probability epsilon, take a random action
        if util.flipCoin(epsilon):
            action = random.choice(legal)

        # Otherwise, choose the action with the highest Q-value
        else:
            stateFeatures = GameStateFeatures(state)
            qValues = {a: self.getQValue(stateFeatures, a) for a in legal}
            print(qValues)
            maxQValue = max(qValues.values())
            # In case multiple actions have the same maximum Q-value, choose randomly among them
            bestActions = [a for a, q in qValues.items() if q == maxQValue]
            action = random.choice(bestActions)

        # Store the current state and action as the previous state and action
        self.prevState = GameStateFeatures(state)
        self.prevAction = action

        return action

        
    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended! Win? {state.isWin()}")

        # Get the final reward
        finalReward = self.computeReward(self.prevState, state)

        # Update the Q-table with the final reward
        self.learn(self.prevState, self.prevAction, finalReward, state)

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0.0)
