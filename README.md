## A Learnable Similarity Metric between Tasks with Dynamics Mismatch

Readme file for the code base to help run the specified experiments 

### Prerequisites

`Python3.7` was used for the current experiments.
We recommend using an [Anaconda](https://www.anaconda.com/) environment, or a [pipenv](https://pypi.org/project/pipenv/) in order to isolate this setup from the rest of your system and preventing unintended consequences.

The necessary libraries and packages can be found in `requirements.txt` and installed using

``python3.7 -m pip install -r requirements.txt``

### Folder structure

The code base is structured as follows:

**data/**: Storage directory for source optimal Q values and policy for the discrete state space environments

**data_cartpole/**: Storage directory for source information obtained for CartPole environment

**data_acrobot/**: Storage directory for source information obtained for Acrobot environment

**data_lander/**: Storage directory for source information obtained for LunarLander environment

**results/**: Directory for results from the discrete state space environments 

**env/**: Contains all the necessary environment files that are need to create the transfer learning scenarios

**agents/**: Contains all the implemented algorithms

**main_tabular.py**: Main File needed to run experiments for environments where tabular representation is used

**main_deeprl.py**: Main File needed to run experiments for environments where neural network representations are required

**plot_results.py**: Helper File that aids in plotting and visualization

**train_source_policy.py**: Main File to train source agent for continuous state space environments


### Training the target task agents

To launch target task experiments for *Windy Gridworld* with Q Learning using Delta as Criteria for Direct Transfer, run the following 

``python3.7 main_tabular.py --env gridworld --seed 0 --agent delta``

The transfer learning scenario can be set in the *main* function of *main_tabular.py*

To launch target task experiments for tasks like *CartPole*, run the following

``python3.7 main_deeprl.py --env CartPole --seed 0 --num_episodes 4000``

The target task specific parameters like Gravity Ratio, Mass Ratio and Lander Mass Ratio can be set by modifying the variable *dynamics_factor* in *main_deeprl.py*

The tensorboard logs obtained from Ray are stored by default external to this repository under the folder name *ray_results*

The above scripts can be modified to launch experiments for the other supported environments by passing appropriate command line arguments

### Training the source agent

The necessary source task information after training the source task can be obtained for *CartPole* using 

``python3.7 train_source_policy.py --env CartPole-v0 --seed 0 --num_episodes 4000``
