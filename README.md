# Snake

A simple implementation of the old video game **Snake**.

<br>

<p align="center"><img src="img/screenshot_1000.gif?raw=true" height="300"></p>
<p align="center"><em>DQN after 1000 episodes</em></p>


## How to

Install dependencies with    ```pip install -r requirements.txt```.

Run    ```main.py train <n>```    to train an agent for n episodes.

Run    ```main.py evaluate <n>```    to evaluate (i.e. average score) a trained agent over n episodes.

Run    ```main.py play True```    to play the game.

Run    ```main.py play False```    to let the RL agent to its thing.


## Deep Q-Network (DQN)

Reference: [V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller (2013) Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

The agent is based on the DQN algorithm and if fully trained gets an average score over 1000 episodes of:

| Episodes trained  | Score |
|-------------------|:-----:|
| 100               | 21.00 |
| 500               | 50.05 |
| 1000              | 59.19 |


## Dependencies

- Python v3.10.9
- Numpy v1.24.1
- PyGame v2.1.2
- PyTorch v1.13.1
- Tqdm v4.64.1
- Matplotlib v3.6.2
- Typer v0.7.0
