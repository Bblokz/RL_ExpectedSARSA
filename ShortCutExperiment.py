from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    run_windy_Q_experiment()
    run_windy_SARSA_experiment()
    run_Q_experiment()
    run_SARSA_experiment()
    run_expected_SARSA_experiment()


def run_windy_Q_experiment():
    n_timesteps = 10000
    n_repetitions = 1
    epsilon = 0.1
    gamma = 1
    alphas = [0.01, 0.1, 0.5, 0.9]
    result = dict()
    for alpha in alphas:
        result[alpha] = run_Q(n_timesteps, n_repetitions,
                              epsilon, alpha, gamma, True)
    plotSteps(result, "Q_learning_windy", "Q-learning")


def run_windy_SARSA_experiment():
    n_timesteps = 10000
    n_repetitions = 1
    epsilon = 0.1
    gamma = 1
    alphas = [0.1]
    result = dict()
    for alpha in alphas:
        result[alpha] = run_SARSA(n_timesteps, n_repetitions,
                                  epsilon, alpha, gamma, True)
    plotSteps(result, "SARSA_windy", "SARSA")


def run_Q_experiment():
    n_timesteps = 1000
    n_repetitions = 100
    epsilon = 0.1
    gamma = 1
    alphas = [0.01, 0.1, 0.5, 0.9]
    result = dict()
    for alpha in alphas:
        result[alpha] = run_Q(n_timesteps, n_repetitions,
                              epsilon, alpha, gamma)
    plotSteps(result, "Q_learning", "Q-learning")


def run_SARSA_experiment():
    n_timesteps = 1000
    n_repetitions = 100
    epsilon = 0.1
    gamma = 1
    alphas = [0.01, 0.1, 0.5, 0.9]
    result = dict()
    for alpha in alphas:
        result[alpha] = run_SARSA(
            n_timesteps, n_repetitions, epsilon, alpha, gamma)
    plotSteps(result, "SARSA", "SARSA")


def run_expected_SARSA_experiment():
    n_timesteps = 1000
    n_repetitions = 100
    epsilon = 0.1
    gamma = 1
    alphas = [0.01, 0.1, 0.5, 0.9]
    result = dict()
    for alpha in alphas:
        result[alpha] = run_expected_SARSA(
            n_timesteps, n_repetitions, epsilon, alpha, gamma)
    plotSteps(result, "Expected_SARSA", "Expected SARSA")


def plotSteps(statesActions, fileNamePrefix, modelName):
    plotX = 0
    plotY = 0
    if (len(statesActions.keys()) == 1):
        fig, axs = plt.subplots(1, 1)
        currentAxis = axs
    else:
        fig, axs = plt.subplots(2, 2)
        currentAxis = axs[0, 0]
    fig.suptitle(modelName + " State action map", fontsize=16)
    fig.set_size_inches(12, 12)
    fig.legend(handles=[mpatches.Patch(color='black', label='Preferred action'), mpatches.Patch(color='red', label='Preferred actions'), mpatches.Patch(
        color='blue', label='Never visited'), mpatches.Patch(color='green', label='Greedy path')], loc='lower center', ncol=4)
    for key, value in statesActions.items():
        # 0 = down, 1 = up, 2 = left, 3 = right.
        maxStateValue = np.argmax(value, axis=1)
        startingStates = [26, 110]

        # print states and estimated action rewards.
        # for i in range(144):
        #     print(i, end=' ')
        #     print(value[i, :])
        # print("meaning [Up, Down, Left, Right]")

        multiAction = dict()  # used to store state with multiple actions preferred
        for i in range(len(maxStateValue)):
            # return indices equal to max cumulated reward
            indices = np.where(value[i] == maxStateValue[i])
            # states with multiple actions equal to the max action value.
            numberEqualActions = len(indices[0])
            if (numberEqualActions > 1):
                if (numberEqualActions == 4):  # never visited state
                    maxStateValue[i] = -2
                else:  # multiple actions preferred.
                    maxStateValue[i] = -1
                    multiAction[i] = indices[0]

        greedyPath = dict()
        for startingState in startingStates:
            currentState = startingState
            x = True
            while x:
                # up
                if (maxStateValue[currentState] == 0):
                    if (0 <= currentState < 12 or currentState in greedyPath):  # at the border or in loop
                        x = False
                    else:
                        greedyPath[currentState] = currentState-12
                        currentState -= 12
                # down
                elif (maxStateValue[currentState] == 1):
                    # at the border or in loop
                    if (132 <= currentState < 144 or currentState in greedyPath):
                        x = False
                    else:
                        greedyPath[currentState] = currentState+12
                        currentState += 12
                # left
                elif (maxStateValue[currentState] == 2):
                    if (currentState % 12 == 0 or currentState in greedyPath):  # aat the border or in loop
                        x = False
                    else:
                        greedyPath[currentState] = currentState-1
                        currentState -= 1
                # right
                elif (maxStateValue[currentState] == 3):
                    if (currentState % 12 == 11 or currentState in greedyPath):  # at the border or in loop
                        x = False
                    else:
                        greedyPath[currentState] = currentState+1
                        currentState += 1

                else:
                    x = False

        currentAxis.title.set_text('alpha = ' + str(key))
        for y in range(12):
            for x in range(12):
                # display cell state number
                currentAxis.text(x, 11-y, str(x + 12*y), fontsize=7)
                # up
                if maxStateValue[x + y * 12] == 0:
                    currentAxis.arrow(x+0.5, 11-y+0.25, 0, 0.5, width=0.05,
                                      length_includes_head=True, facecolor='black', edgecolor='black')
                # down
                elif maxStateValue[x + y * 12] == 1:
                    currentAxis.arrow(x+0.5, 11-y+0.75, 0, -0.5,
                                      width=0.05, length_includes_head=True, facecolor='black', edgecolor='black')
                # left
                elif maxStateValue[x + y * 12] == 2:
                    currentAxis.arrow(x+0.75, 11-y+0.5, -0.5, 0,
                                      width=0.05, length_includes_head=True, facecolor='black', edgecolor='black')
                # right
                elif maxStateValue[x + y * 12] == 3:
                    currentAxis.arrow(x+0.25, 11-y+0.5, 0.5, 0, width=0.05,
                                      length_includes_head=True, facecolor='black', edgecolor='black')
                # multiple actions preferred.
                elif maxStateValue[x + y * 12] == -1:
                    # get all preferred actions for the given state.
                    actions = multiAction[x + y * 12]
                    for action in actions:
                        # up
                        if action == 0:
                            currentAxis.arrow(x+0.5, 11-y+0.25, 0, 0.4, head_width=0.175,
                                              length_includes_head=False, facecolor='red', edgecolor='red')
                        # down
                        elif action == 1:
                            currentAxis.arrow(x+0.5, 11-y+0.75, 0, -0.4,
                                              head_width=0.175, length_includes_head=False, facecolor='red', edgecolor='red')
                        # left
                        elif action == 2:
                            currentAxis.arrow(x+0.75, 11-y+0.5, -0.4, 0,
                                              head_width=0.175, length_includes_head=False, facecolor='red', edgecolor='red')
                        # right
                        elif action == 3:
                            currentAxis.arrow(x+0.25, 11-y+0.5, 0.4, 0, head_width=0.175,
                                              length_includes_head=False, facecolor='red', edgecolor='red')
                else:  # State never visited.
                    currentAxis.scatter([x+0.5], [11-y+0.5], c='b')
        for key, value in greedyPath.items():
            x1 = key % 12 + 0.5
            y1 = 11 - (key//12) + 0.5

            x2 = value % 12 + 0.5
            y2 = 11 - (value//12) + 0.5
            currentAxis.plot(
                [x1, x2], [y1, y2], color="green", alpha=0.5, linewidth=3)
        currentAxis.grid(True)
        currentAxis.get_yaxis().set_visible(False)
        currentAxis.get_xaxis().set_visible(False)
        plotX += 1
        if (plotX == 2):
            plotX = 0
            plotY += 1
            if (plotY == 2):
                break
        if (len(statesActions.keys()) > 1):
            currentAxis = axs[plotX, plotY]
    fig.tight_layout(pad=3)
    fig.subplots_adjust(bottom=0.05)
    plt.savefig(fileNamePrefix+".png")


def run_Q(n_timesteps, n_repetitions, epsilon, alpha, gamma, windy=False):
    vectorResult = np.zeros(n_timesteps)
    testQ = np.zeros((12*12, 4))
    for j in range(n_repetitions):
        if (windy):
            env = WindyShortcutEnvironment()
        else:
            env = ShortcutEnvironment()
        pi = QLearningAgent(env.action_size(), env.state_size(),
                            epsilon, alpha, gamma)  # Initialize policy
        for i in range(1, n_timesteps+1):
            if env.done():
                break
            current_state = env.state()
            # select action for current state.
            a = pi.select_action(current_state)
            r = env.step(a)  # sample reward
            vectorResult[i-1] += r
            pi.update(current_state, a, r, env.state())  # update policy
        testQ += pi.Q.copy()
    return testQ
    return vectorResult


def run_SARSA(n_timesteps, n_repetitions, epsilon, alpha, gamma, windy=False):
    testQ = np.zeros((12*12, 4))
    for j in range(n_repetitions):
        if (windy):
            env = WindyShortcutEnvironment()
        else:
            env = ShortcutEnvironment()
        pi = SARSAAgent(env.action_size(), env.state_size(),
                        epsilon, alpha, gamma)  # Initialize policy
        for i in range(1, n_timesteps+1):
            if env.done():
                break
            current_state = env.state()
            # select action for current state.
            a = pi.select_action(current_state)
            r = env.step(a)  # sample reward
            pi.update(current_state, a, r, env.state())  # update policy
        testQ += pi.Q.copy()
    return testQ


def run_expected_SARSA(n_timesteps, n_repetitions, epsilon, alpha, gamma, windy=False):
    testQ = np.zeros((12*12, 4))
    for j in range(n_repetitions):
        if (windy):
            env = WindyShortcutEnvironment()
        else:
            env = ShortcutEnvironment()
        pi = ExpectedSARSAAgent(env.action_size(), env.state_size(),
                                epsilon, alpha, gamma)  # Initialize policy
        for i in range(1, n_timesteps+1):
            if env.done():
                break
            current_state = env.state()
            # select action for current state.
            a = pi.select_action(current_state)
            r = env.step(a)  # sample reward
            pi.update(current_state, a, r, env.state())  # update policy
        testQ += pi.Q.copy()
    return testQ


if __name__ == "__main__":
    main()
