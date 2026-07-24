import os
import logging
import time
import numpy as np
import psutil
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler
from torch.utils.tensorboard import SummaryWriter
from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent
import curriculum_door  # noqa: F401 -- registers the CurriculumDoor env with robosuite

if __name__ == "__main__":

    if not os.path.exists("tmp/ppo"):
        os.makedirs("tmp/ppo")

    env_name = "CurriculumDoor"

    from robosuite.controllers import load_part_controller_config
    controller_config = load_composite_controller_config(controller="BASIC")
    controller_config["body_parts"]["right"] = load_part_controller_config(default_controller="JOINT_VELOCITY")
    controller_config["body_parts"]["right"]["gripper"] = {"type": "GRIP"}

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs = controller_config,
        has_renderer = False,
        use_camera_obs = False,
        horizon = 300,
        reward_shaping = True,
        control_freq = 20,
        hard_reset = False,
    )
    
    env = GymWrapper(env)
    
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128
    
    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005, input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, warmup=5000)
    
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'logs/{run_name}')
    n_games = 10
    
    for experiment in range(10):
        best_score = 0
        
        episode_identifier = f"{experiment} - actor_learning_rate={actor_learning_rate} critic_learning_rate{critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"
                
        # agent.load_models()
        for i in range(n_games):
            observation, info = env.reset()  #ALWAYS RESETTING THE ENVIRONMENT
            done = False
            score = 0

            while not done:
                action = agent.choose_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                score += reward
                agent.remember(observation, action, reward, next_observation, done)
                agent.learn()
                observation = next_observation

            writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

            if (i % 10) == 0:
                agent.save_models()

            print(f"Episode: {i} Score: {score}")
            
            actor_learning_rate = actor_learning_rate * 0.8


    ###
    # critic_network = CriticNetwork([8], n_actions=8)
    # actor_network = ActorNetwork([8], n_actions=8)
    
    # replay_buffer = ReplayBuffer(max_size=100000, input_shape=[8], n_actions=8)
    