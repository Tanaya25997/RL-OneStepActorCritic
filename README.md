# RL-OneStepActorCritic

Implementation of the one step actor critic on Acrobot and Mountain Car Domains

2.1 ENVIRONMENTS CHOSEN:
• Acrobot
• Mountain Car

Both the environments follow the definitions as provided in the Gym. documentation. In the code
as well, the environments are implemented using Gym.

Note that for both these environments, a reward of -1 is obtained when the action taken does not
result into reaching the goal and a reward of 0 when the goal is reached. The goal of the algorithm
should be to get a total reward that is as less negative as possible.

2.2 A BRIEF DESCRIPTION OF THE METHODS AND WHAT THEY DO:

Both Acrobot and Mountain Car are ”discrete actions continuous states” environments. As a result,
I have implemented a generic ’One-Step Actor-Critic’ algorithm.

A single ’OneStepActorCritic’ class has been implemented. A brief description of the different methods
inside this class and what they do is as follows:

1. init (args):
This methods initializes the class variables. To be exact, we initialize the Number of states
(num actions) and Number of Actions (num states) of the environment, along with the Fourier
basis order (M), the Actor Learning rate (alpha theta), the Critic Learning rate (alpha weights),
the Discount Factor (gamma), the state intervals (low and high). Another parameter is the
acrobot true which is a boolean and signifies if the algorithm is running for Acrobot or Mountain
Car.

3. fourier basis(state):
This method return the Fourier basis of the given state for a particular order. I have implemented
Fourier basis as it is a linear function approximator and is suggested to work well with One-Step
Actor-Critic which is based on semi-stochastic gradient descent.So, the output of this method are
the state featured for a given state.

5. select action(state):
This method gives the action that the environment should state based on it’s current state and policy.
Since we are using a parameterized policy, therefore, softmax function is being used to calculate the
probabilities for each action. Based on the probabilities, a random choice of action is performed.

4.gradient ln policy(args):
This method calculates the gradient of the policy with respect to the parameters theta of the policy.
This has been implemented using the definition of gradient as described in the lecture notes (Section
11.6 Pages 104,105). To recapitulate, the gradient with respect to theta is a matrix of size |A| ∗ |ϕ(s)| where |A| is the
number of actions and ϕ(s) is the state feature. The matrix is obtained by the softmax probability
for each action with the state feature (this constitutes each row of the matrix) except for the
action chosen for which the row is obtained by multiplying the state feature with (1 - the softmax
probability for that action).

7. actor critic algorithm(env, episodes):
This is the main algorithm. I’m running the code for a total of 1000 episodes. Inside each episode,
the environment is initialized and then episode is run until the termination (the goal is accomplished)
or the truncation (episode length over 500 for Acrobot and 200 for Mountain Car) is reached. Once
this is done, the episode ends and a new episode is run.

Here, the values for δ and the updates for weights(w), theta(θ) are calculated as per the definitions
given in the pseudocode of the One-Step Actor-Critic algorithm (Section 13.5 of the RL book).

2.4 A DISCUSSION OF HOW YOU TUNED THEIR HYPER-PARAMETERS

The parameters γ, αθ, αw,M were tuned using trial and error.

Some of my observations were as follows:

• With gamma < 0.99, it is difficult to find learning rates that make the algorithm learn.
And mostly, the episodes truncated (i.e., reached the maximum number of steps allowed by
the environment in each episode. A γ = 0.98 still has learning rates that give good results
but the search is more intensive and usually require a higher Fourier basis order to work.
Below 0.98 though, finding optimal learning rates and M was nearly impossible.

• Secondly, for Acrobot, it was observed that for γ = 0.99, when learning rates were at 0.02
and above 0.02, the algorithm faced variance and more spikes are seen in the learning curve.
For Mountain Car environment, however, this does not seem to be the case as learning rates
of 0.1 also work with a considerably high Fourier Basis order.

