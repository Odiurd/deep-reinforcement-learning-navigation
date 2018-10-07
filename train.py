from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
from dqn_agent import Agent
import matplotlib.pyplot as plt


ENV_PATH = "E:/Users/Megaport/Dropbox/Python/Udacity/Project 1/Banana_Windows_x86_64/Banana.exe"
CHECKPOINT_NAME = 'checkpoint.pth'
IMAGE_NAME = 'scores.png'
TARGET_SCORE = 13
GRAPHICS_OFF = True


def plot(scores, IMAGE_NAME):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('img/{}'.format(IMAGE_NAME))


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    # agent uses double DQN with dueling + experience replay
    #agent = Agent(state_size=state_size, action_size=action_size, seed=0, num_layers=2, hidden_units=64)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            action = int(action) ### FIX 
            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        if np.mean(scores_window)>=TARGET_SCORE:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), "ckpt/{}".format(CHECKPOINT_NAME))
            break
        
    return scores


env = UnityEnvironment(file_name=ENV_PATH, no_graphics=GRAPHICS_OFF)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

print('Number of agents: {}'.format(env_info.agents))
print('Number of actions: {}'.format(action_size))
print('Number of states: {}'.format(state_size))

print('States: {}'.format(state))


if torch.cuda.is_available():
    print("trainining on GPU")
else:
    print("training on CPU")


scores = dqn(n_episodes=1000)
plot(scores, IMAGE_NAME)
env.close()


