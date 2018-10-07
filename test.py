from unityagents import UnityEnvironment
from dqn_agent import Agent
import torch

ENV_PATH = "E:/Users/Megaport/Dropbox/Python/Udacity/Project 1/Banana_Windows_x86_64/Banana.exe"
CHECKPOINT_NAME = 'checkpoint.pth'
GRAPHICS_OFF = False

n_episodes = 3
max_t = 200

env = UnityEnvironment(file_name=ENV_PATH, no_graphics=GRAPHICS_OFF)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('ckpt/{}'.format(CHECKPOINT_NAME)))

for i_episode in range(1, n_episodes+1):
    print('Starting episode {}'.format(i_episode))
    env_info = env.reset(train_mode=GRAPHICS_OFF)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0
    for t in range(max_t):
        action = agent.act(state)
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