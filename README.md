
### Introduction
This project was made at the Introduction to Reinforcement Learning course, Leiden Univsersity.
The goal is to investigate Three algorithms:
1. Q-Learning
2. SARSA
3. Expected-SARSA

To study these algorithms, a custom environment has been created called the ShortCut environment. This environment consists of a 12x12 grid, with the agent being able to move to adjacent squares. Actions that try to move the agent outside the grid do not work. The objective of the agent is to reach the goal square, which is shown as green in the provided figure called ShortCut.png. If the goal is reached, the episode ends. At the beginning of each episode, the agent starts from one of two squares, which are shown as blue in the figure, with each square having an equal probability of being chosen. There are certain squares in the environment that act as a 'cliff', similar to the Cliff Walking environment as described on page 132 of Sutton & Barto. If the agent ends up in the cliff, it is returned to the starting position, but with a negative reward of -100. All other actions have a reward of -1. The RL algorithm is supposed to maximize the cumulative reward of each episode.

The original report writen for this assignment is also included in the repository,
in there you can find my results and analysis of the algorithms. And intuitions of why each algorithm performs better or worse in certain situations.

### Setup & Run
create a virtual environment:
```bash
python3 -m venv myenv
```
install the requirements:
```bash
pip install -r requirements.txt
```

--- run the experiments on linux:
1. activate environment:
```bash
source myenv/bin/activate
```
2. run the experiments:
```bash
python3 ShortCutExperiment.py
```

--- run the experiments on windows:
1. activate environment:
```bash
myenv\Scripts\activate
```
2. run the experiments:
```bash
python ShortCutExperiment.py
```
