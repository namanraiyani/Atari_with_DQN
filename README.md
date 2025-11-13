# Atari_with_DQN
![vid9 (1)](https://github.com/user-attachments/assets/4c71ae8c-aecd-4fd8-b451-8c1a2059d031)

-----

#  DQN Atari Breakout Solver

This project is a modular implementation of a **Deep Q-Network (DQN)** agent capable of learning how to play **Atari Breakout** from scratch.

It uses **PyTorch** for the neural network, **Gymnasium** for the game environment, and **Pygame** to render a custom, real-time dashboard that displays the game feed, neural network metrics, and a live training performance graph.

##  Features

  * **Double DQN Logic:** Implements stable training targets to reduce value overestimation.
  * **Real-Time Dashboard:** A Pygame window showing the agent's view alongside a live plotting of rewards and moving averages.
  * **Experience Replay:** Stores past experiences to break correlation between consecutive learning steps.
  * **Modular Design:** Clean separation of concerns (Agent, Model, Memory, Config) for easy debugging and extension.
  * **Epsilon-Greedy Exploration:** implementation of exploration decay strategies.

##  Installation

1.  **Clone the repository**.

2.  **Install dependencies:**
    This project requires Python 3.8+. Run the following command to install all necessary libraries, including the Atari ROM license manager:

    ```bash
    pip install gymnasium[atari] gymnasium[accept-rom-license] ale-py pygame torch numpy opencv-python matplotlib
    ```

##  Usage

To start the training process, simply run the main script:

```bash
python main.py
```

### The Dashboard

Once running, a window will appear with two sections:

1.  **Left Side:** The game rendering (processed 84x84 grayscale input).
2.  **Right Side:**
      * **Episode:** Current game number.
      * **Score:** Total reward for the current game.
      * **Action:** The decision the AI just made (LEFT, RIGHT, FIRE, NOOP).
      * **Epsilon:** The current probability of taking a random action (exploration rate).
      * **Graph:** A live plot of raw Reward (Cyan) vs. 50-Episode Moving Average (White).

## 📂 Project Structure

```text
atari_breakout/
│
├── config.py          
├── main.py             
│
└── src/
    ├── __init__.py
    ├── agent.py        
    ├── memory.py       
    ├── model.py        
    └── utils.py        
```

##  How It Works

### 1\. The Input (Preprocessing)

The agent does not know what a "ball" or "paddle" is. It only sees raw pixels.

  * The game frame is converted to **Grayscale**.
  * It is resized to **84x84 pixels**.
  * These pixels are fed into a **Convolutional Neural Network (CNN)**.

### 2\. The Brain (Model)

The network (`src/model.py`) takes the image input and outputs a **Q-Value** for every possible action. A Q-Value represents: *"How much total reward do I expect to get in the future if I take this specific action right now?"*

### 3\. The Training (Double DQN)

The code in `src/agent.py` implements **Double DQN** to stabilize learning:

1.  **Selection:** The `policy_net` decides which action is the best for the *next* state.
2.  **Evaluation:** The `target_net` calculates the value of that specific action.
3.  **Loss Calculation:** The system calculates the error (Huber Loss / Smooth L1) between the predicted Q-value and the actual reward received plus the future value.
4.  **Backpropagation:** The error is backpropagated to update the weights of the `policy_net`.

### 4\. Exploration vs. Exploitation

  * **Exploration:** At the start, the agent moves randomly (High Epsilon) to discover the game mechanics.
  * **Exploitation:** Over time, Epsilon decays, and the agent relies more on its Neural Network to make strategic decisions to maximize the score.

##  Configuration

You can tweak the training behavior in `config.py`:

  * `BATCH_SIZE`: How many memories to train on at once.
  * `LR`: The learning rate (speed of adaptation).
  * `GAMMA`: The discount factor (how much the agent cares about future rewards vs. immediate rewards).
  * `EPS_DECAY`: How fast the agent stops acting randomly.
