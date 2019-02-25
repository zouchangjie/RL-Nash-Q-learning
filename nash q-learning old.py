import numpy as np
from itertools import permutations
import lemkeHowson
import matrix
import rational

# world height
WORLD_HEIGHT =3

# world width
WORLD_WIDTH = 3

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

gridIndexList = []
for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndexList.append(WORLD_WIDTH * i + j)

actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
statesAllOne = []
locationValidActions = {}

for i in permutations(gridIndexList, 2):
    statesAllOne.append(i)

statesAllOne.append((7, 7))

for i in gridIndexList:
    locationValidActions[i] = []

for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndexNumber = WORLD_WIDTH * i + j
        if i != WORLD_HEIGHT - 1:
            locationValidActions[gridIndexNumber].append(ACTION_UP)
        if i != 0:
            locationValidActions[gridIndexNumber].append(ACTION_DOWN)
        if j != 0:
            locationValidActions[gridIndexNumber].append(ACTION_LEFT)
        if j != WORLD_WIDTH - 1:
            locationValidActions[gridIndexNumber].append(ACTION_RIGHT)

class agent:
    def __init__(self, agentIndex = 0, startLocationIndex = 0):
        self.goalState = ()
        self.qTable = {}
        self.timeNumber = {}
        self.alpha = {}
        # self.singleQValue = {}
        # self.singleAlpha = {}
        self.currentState = ()
        self.nextState = ()
        self.strategy = {}
        self.agentIndex = agentIndex
        self.startLocationIndex = startLocationIndex
        self.locationIndex = startLocationIndex
        self.currentAction = 0
        self.currentReward = 0
        self.timeStep = 0

    def initialSelfStrategy(self):
        for i in statesAllOne:
            self.strategy[i] = {}
            for j in locationValidActions[i[self.agentIndex]]:
                self.strategy[i][j] = 0

    def initialSelfQTable(self):
        # agent0 and agnet1
        self.qTable[0] = {}
        self.qTable[1] = {}
        for i in statesAllOne:
            self.qTable[0][i] = {}
            self.qTable[1][i] = {}
            for j_1 in locationValidActions[i[0]]:
                for j_2 in locationValidActions[i[1]]:
                    self.qTable[0][i][(j_1, j_2)] = 0
                    self.qTable[1][i][(j_1, j_2)] = 0

    def initialSelfAlpha(self):
        for i in statesAllOne:
            self.alpha[i] = {}
            self.timeNumber[i] = {}
            for j_1 in locationValidActions[i[0]]:
                for j_2 in locationValidActions[i[1]]:
                    self.alpha[i][(j_1, j_2)] = 0
                    self.timeNumber[i][(j_1, j_2)] = 0

    # def initialSingleQValue(self):
    #     for i in statesAllOne:
    #         self.singleQValue[i] = {}
    #         for j in locationValidActions[i[self.agentIndex]]:
    #             self.singleQValue[i][j] = 0
    #
    # def initialSingleAlpha(self):
    #     for i in statesAllOne:
    #         self.singleAlpha[i] = {}
    #         for j in locationValidActions[i[self.agentIndex]]:
    #             self.singleAlpha[i][j] = 0

    # def chooseActionWithEpsilon(self, EPSILON, currentState):
    #     self.locationIndex = currentState[self.agentIndex]
    #     if np.random.binomial(1, self.EPSILON) == 1:
    #         self.currentAction = np.random.choice(locationValidActions[self.locationIndex])
    #     else:
    #         # it is a method to find the max value in a dict and the corresponding key
    #         self.currentAction = max(zip(self.singleQValue[self.currentState].values(), self.singleQValue[self.currentState].keys()))[1]

    def chooseActionRandomly(self, currentState):
        self.locationIndex = currentState[self.agentIndex]
        self.currentAction = np.random.choice(locationValidActions[self.locationIndex])
        return self.currentAction

    def constructPayoffTable(self, state):
        actions0 = locationValidActions[state[0]]
        actions1 = locationValidActions[state[1]]
        m0 = matrix.Matrix(len(actions0), len(actions1))
        m1 = matrix.Matrix(len(actions0), len(actions1))
        for i in range(len(actions0)):
            for j in range(len(actions1)):
                m0.setItem(i+1, j+1, self.qTable[0][state][(actions0[i], actions1[j])])
                m1.setItem(i+1, j+1, self.qTable[1][state][(actions0[i], actions1[j])])
        return (m0, m1)

    def nashQLearning(self, gamma, agent0Action, agent0Reward, currentState, nextState, agent1Action, agent1Reward):
        self.gamma = gamma
        self.currentState = currentState
        self.nextState = nextState
        self.timeNumber[self.currentState][(agent0Action, agent1Action)] += 1
        self.alpha[self.currentState][(agent0Action, agent1Action)] = 1.0 / self.timeNumber[self.currentState][(agent0Action, agent1Action)]

        (m0, m1) = self.constructPayoffTable(nextState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        prob0 = np.array(probprob[0])
        re0 = np.where(prob0 == np.max(prob0))[0][0]
        prob1 = np.array(probprob[1])
        re1 = np.where(prob1 == np.max(prob1))[0][0]
        nash0 = m0.getItem(re0 + 1, re1 + 1)
        nash1 = m1.getItem(re0 + 1, re1 + 1)
        nashQValues = [nash0, nash1]
        self.qTable[0][self.currentState][(agent0Action, agent1Action)] \
            = (1 - self.alpha[self.currentState][(agent0Action, agent1Action)]) \
                * self.qTable[0][self.currentState][(agent0Action, agent1Action)] \
                    + self.alpha[self.currentState][(agent0Action, agent1Action)] \
                        * (agent0Reward + self.gamma * nashQValues[0])
        # for keys1 in self.qTable[0]:
        #     for keys2 in self.qTable[0][keys1]:
        #         if self.qTable[0][keys1][keys2] != 0:
        #             print (keys1, keys2, self.qTable[0][keys1][keys2])
        self.qTable[1][self.currentState][(agent0Action, agent1Action)] \
            = (1 - self.alpha[self.currentState][(agent0Action, agent1Action)]) \
              * self.qTable[1][self.currentState][(agent0Action, agent1Action)] \
              + self.alpha[self.currentState][(agent0Action, agent1Action)] \
                * (agent1Reward + self.gamma * nashQValues[1])
        self.timeStep += 1

    def chooseActionBasedOnQTable(self, currentState):
        self.locationIndex = currentState[self.agentIndex]
        (m0, m1) = self.constructPayoffTable(currentState)
        probprob = lemkeHowson.lemkeHowson(m0, m1)
        prob0 = np.array(probprob[0])
        re0 = np.where(prob0 == np.max(prob0))[0][0]
        prob1 = np.array(probprob[1])
        re1 = np.where(prob1 == np.max(prob1))[0][0]
        re = [re0, re1]
        actionsAvailable = locationValidActions[currentState[self.agentIndex]]
        return actionsAvailable[re[self.agentIndex]]


def nextGridIndex (action, gridIndex):
    action = action
    index_i = int(gridIndex / 3)
    index_j = gridIndex - index_i * 3
    if (action == 0):
        index_i += 1
    elif (action == 1):
        index_i -= 1
    elif (action == 2):
        index_j -= 1
    elif (action == 3):
        index_j += 1
    nextIndex = index_i * 3 + index_j
    return nextIndex

def gridGameOne(action_0, action_1, currentState):
    action_0 = action_0
    action_1 = action_1
    currentState = currentState
    reward_0 = 0
    reward_1 = 0
    endGameFlag = 0

    currentIndex_0 = currentState[0]
    currentIndex_1 = currentState[1]
    nextIndex_0 = nextGridIndex(action_0, currentState[0])
    nextIndex_1 = nextGridIndex(action_1, currentState[1])

    if (nextIndex_0 == 8 and nextIndex_1 == 6):
        reward_0 = 100
        reward_1 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
    elif (nextIndex_0 == 8):
        reward_0 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
        if (nextIndex_1 == 8):
            reward_1 = -1
        else:
            reward_1 = 0
    elif (nextIndex_1 == 6):
        reward_1 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
        if (nextIndex_0 == 6):
            reward_0 = -1
        else:
            reward_0 = 0
    elif (nextIndex_0 == nextIndex_1):
        reward_0 = -1
        reward_1 = -1
        nextState = (currentIndex_0, currentIndex_1)
    else:
        reward_0 = 0
        reward_1 = 0
        nextState = (nextIndex_0, nextIndex_1)
    return reward_0, reward_1, nextState, endGameFlag

def resetStartState():
    agent_0LocationIndex = np.random.choice([x for x in gridIndexList if x not in [8]])
    while True:
        agent_1LocationIndex = np.random.choice([x for x in gridIndexList if x not in [6]])
        if agent_1LocationIndex != agent_0LocationIndex:
            break
    return (agent_0LocationIndex, agent_1LocationIndex)

def playGameOne(agent_0 = agent, agent_1 = agent):
    gamma = 0.99
    agent_0 = agent_0
    agent_1 = agent_1
    currentState = (agent_0.startLocationIndex, agent_1.startLocationIndex)
    timeStep = 0 # calculate the timesteps in one episode
    episodes = 0
    endGameFalg = 0
    agent_0.initialSelfQTable()
    agent_1.initialSelfQTable()
    agent_0.initialSelfAlpha()
    agent_1.initialSelfAlpha()
    while episodes < 5000:
        print (episodes)
        while True:
            agent0Action = agent_0.chooseActionRandomly(currentState)
            agent1Action = agent_1.chooseActionRandomly(currentState)
            reward_0, reward_1, nextState, endGameFlag = gridGameOne(agent0Action, agent1Action, currentState)
            agent_0.nashQLearning(gamma, agent0Action, reward_0, currentState, nextState, agent1Action, reward_1)
            agent_1.nashQLearning(gamma, agent0Action, reward_0, currentState, nextState, agent1Action, reward_1)
            if (endGameFlag == 1): # one episode of the game is end
                episodes += 1
                currentState = resetStartState()
                break
            currentState = nextState



def test (agent_0 = agent, agent_1 = agent):
    agent_0 = agent_0
    agent_1 = agent_1
    startState = (0, 2)
    endGameFlag = 0
    runs = 0
    agentActionList = []
    currentState = startState
    endGameFlag = 0
    while endGameFlag != 1:
        agent0Action = agent_0.chooseActionBasedOnQTable(currentState)
        agent1Action = agent_1.chooseActionBasedOnQTabule(currentState)
        agentActionList.append([agent0Action, agent1Action])
        reward_0, reward_1, nextState, endGameFlag = gridGameOne(agent0Action, agent1Action, currentState)
        currentState = nextState
    agentActionList.append(currentState)
    return agentActionList

runs = 0
agentActionListEveryRun = {}
for runs in range(1):
    agent_0 = agent(agentIndex=0, startLocationIndex=0)
    agent_1 = agent(agentIndex=1, startLocationIndex=2)
    playGameOne(agent_0, agent_1)
    agentActionListEveryRun[runs] = test(agent_0, agent_1)
    print (runs)
nashnum = 0
for runs in range(1):
    if agentActionListEveryRun[runs][4] == (8, 6):
        nashnum += 1
print (agentActionListEveryRun)
print (nashnum)

