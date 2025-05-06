import gym
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from ballEnv import PongEnv
from stable_baselines3.common.env_checker import check_env
# 初始化环境
env = PongEnv()
check_env(env, warn=True)

clock=pygame.time.Clock()
# 创建 DQN 模型
model = DQN(
    policy="CnnPolicy",  # 使用卷积神经网络作为策略
    env=env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1
    # tensorboard_log="./dqn_pong_tensorboard/"
)

# 训练模型
model.learn(total_timesteps=1000,progress_bar=True)  # 根据需要调整时间步数

# 保存模型
model.save("dqn_pong")
model = DQN.load("dqn_pong")
# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# 评估模型
obs,_= env.reset()  # 从 `reset()` 中解包  info

while True:
    action, _states = model.predict(obs, deterministic=True)
    # step()返回 `obs, reward, done, info, truncated`
    obs, reward, done,info, truncated = env.step(action)
    env.render()
    
    if done or truncated:  # 处理 done 或 truncated 标志
        obs,_= env.reset()  
    
    clock.tick(20)  # 控制帧率，每秒10帧
