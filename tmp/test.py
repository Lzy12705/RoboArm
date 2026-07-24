import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
import curriculum_door  # noqa: F401 -- registers the CurriculumDoor env with robosuite

if __name__ == '__main__':
    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")

    env_name = "CurriculumDoor"

    controller_config = load_composite_controller_config(controller="BASIC")
    controller_config["body_parts"]["right"] = load_part_controller_config(default_controller="JOINT_VELOCITY")
    controller_config["body_parts"]["right"]["gripper"] = {"type": "GRIP"}

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs = controller_config,
        has_renderer = True,
        use_camera_obs = False,
        horizon = 300,
        reward_shaping = True,
        render_camera = "frontview",
        has_offscreen_renderer = True,
        control_freq = 20,
    )
    
    env = GymWrapper(env)
    
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128
    
    agent = Agent(actor_learning_rate = actor_learning_rate, critic_learning_rate = critic_learning_rate, tau = 0.005, input_dims = env.observation_space.shape,
                  env = env, n_actions = env.action_space.shape[0], layer1_size = layer1_size, layer2_size = layer2_size, batch_size = batch_size)
    
    #  Built for testing purposes 
    n_games = 10000
    best_score = 0
    episode_identifier = f"0 -actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer_1_size={layer1_size} layer_2_size={layer2_size}"
    
    agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()  #ALWAYS RESETTING THE ENVIRONMENT
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, validation = True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            observation = next_observation
            time.sleep(0.03)
            
            
        print(f"Episode: {i} Score: {score}")