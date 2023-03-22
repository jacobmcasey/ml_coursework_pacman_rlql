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

        self.legal_actions = state.getLegalActions()
        self.pacman_position = state.getPacmanPosition()
        self.ghost_positions = state.getGhostPositions()
        self.food_positions = state.getFood().asList()
        self.score = state.getScore()
        self.capsule_positions = state.getCapsules()

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.8,
                 epsilon: float = 1,
                 gamma: float = 0.8,
                 maxAttempts: int = 100,
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
        self.qTable = {}  # The Q-table
        self.visitCount = {}  # Visitation count for each state-action pair

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
            return 100.0
        elif endState.isLose():
            print ("LOOOSSSSSSSEEEEEEEEEEEEERRRRRRRRR")
            # Pacman lost the game (collided with a ghost or ran out of time)
            return -100.0
        else:
            # Compute the difference in the number of remaining food pellets
            numFoodStart = startState.getNumFood()
            numFoodEnd = endState.getNumFood()
            foodDiff = numFoodStart - numFoodEnd

            # Compute the reward based on the difference
            return foodDiff * 10.0

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
        q_value = self.qTable.get((state, action), 0.0)
        return q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Get the legal actions for the current state
        legalActions = state.legal_actions

        # Find the action with the highest Q-value
        maxQValue = float('-inf')
        for action in legalActions:
            qValue = self.qTable.get((state, action), 0.0)
            if qValue > maxQValue:
                maxQValue = qValue

        q_value = maxQValue
        return q_value

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
        # Get the Q-value for the current state-action pair
        currentQValue = self.qTable.get((state, action), 0.0)

        # Compute the maximum Q-value for the next state
        maxNextQValue = self.maxQValue(nextState)

        # Update the Q-value for the current state-action pair using the Q-learning formula
        newQValue = currentQValue + self.alpha * (reward + self.gamma * maxNextQValue - currentQValue)
        self.qTable[(state, action)] = newQValue

        print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        # Increment the visitation count for the current state-action pair
        self.visitCount[(state, action)] = self.visitCount.get((state, action), 0) + 1

        # Increment episode counter
        self.numEpisodes += 1

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

        # Increment the visitation count for the current state-action pair
        self.visitCount[(state, action)] = self.visitCount.get((state, action), 0) + 1

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
        # Get the visitation count for the current state-action pair
        return self.visitCount.get((state, action), 0)

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
        # Calculate the current epsilon value based on the number of episodes played so far
        epsilon = 1.0 / (1.0 + self.numEpisodes)

        # Determine whether to explore or exploit based on the current epsilon value
        if random.random() < epsilon:
            return random.uniform(-1.0, 1.0)
        else:
            return utility

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
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Determine whether to explore or exploit based on the current epsilon value
        if random.random() < self.epsilon:
            # Explore by choosing a random action from the legal actions
            return random.choice(legal)
        else:
            # Exploit by choosing the action with the highest Q-value
            maxQValue = float("-inf")
            bestAction = None
            for action in legal:
                qValue = self.qTable.get((stateFeatures, action), 0.0)
                if qValue > maxQValue:
                    maxQValue = qValue
                    bestAction = action
            return bestAction


    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)