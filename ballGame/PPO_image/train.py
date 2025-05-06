import os
from stable_baselines3.common.logger import configure
import numpy as np
import random
import pygame
import gym
from gym import spaces
import gymnasium as gym  # 使用 gymnasium 替代 gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from ballEnv import PongEnv

import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
# 创建你的自定义环境
env = PongEnv()
check_env(env, warn=True)
clock=pygame.time.Clock()


print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# 配置日志记录
log_path = "./logs/"
new_logger = configure(log_path, ["stdout", "csv"])
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

class LoggerCallback(BaseCallback):
    def __init__(self, log_path: str, verbose=0):
        super(LoggerCallback, self).__init__(verbose)
        self.log_path = log_path
        self.total_reward = 0.0  # 初始化总的奖励
        self.episode_count = 0   # 初始化已完成的回合数量

    def _on_step(self) -> bool:
        reward = self.locals['rewards']
        self.total_reward += reward  # 累计奖励

        if self.locals['dones'][0]:  # 如果回合结束
            self.episode_count += 1  # 增加回合数量

        if self.n_calls % 1000 == 0:  # 每1000步记录一次
            average_reward = self.total_reward / self.episode_count if self.episode_count > 0 else 0
            print(f"Step {self.num_timesteps}: Total Reward: {self.total_reward}, Average Reward: {average_reward}")
            self.model.save("ppo_ball_game")

        return True

    def _on_training_end(self) -> None:
        # 在训练结束时，可以记录或处理最后的结果
        average_reward = self.total_reward / self.episode_count if self.episode_count > 0 else 0
        print(f"Training finished. Total Reward: {self.total_reward}, Average Reward: {average_reward}")


# 使用回调
logger_callback = LoggerCallback(log_path="reward_log.txt")

# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=256):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(1*1*64, features_dim)  # Adjust according to your output dimensions


#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.flatten(x)
#         x = F.relu(self.fc(x))
#         return x


# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=256)
# )

model = PPO(policy='CnnPolicy', env=env, verbose=1,tensorboard_log=log_path, learning_rate=1e-4, n_steps=2048, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.1)
model.learn(total_timesteps=100000,callback=logger_callback)
# 保存和加载模型
model.save("ppo_ball_game")
model = PPO.load("ppo_ball_game")

# 评估模型
obs, _ = env.reset()  # 从 `reset()` 中解包  info

while True:
    action, _states = model.predict(obs, deterministic=True)
    # step()返回 `obs, reward, done, info, truncated`
    obs, reward, done, info, truncated = env.step(action)
    env.render()
    
    if done or truncated:  # 处理 done 或 truncated 标志
        obs, _ = env.reset()  
    
    clock.tick(20)  # 控制帧率，每秒10帧


