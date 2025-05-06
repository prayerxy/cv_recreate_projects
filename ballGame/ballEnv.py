from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from PIL import Image
import gymnasium as gym  # 使用 gymnasium 替代 gym
from gymnasium import spaces
import numpy as np
import random
import pygame

class PongEnv(gym.Env):
    def __init__(self, grid_size=30, paddle_size=5, max_bricks=100, max_paddle_speed=4,max_ball_speed=4.0):
        super(PongEnv, self).__init__()
        self.grid_size = grid_size
        self.paddle_size = paddle_size
        self.max_bricks = max_bricks
        self.max_ball_speed = max_ball_speed #限制最大水平速度
        self.max_paddle_speed = max_paddle_speed
        self.paddle_velocity = 0
        self.previous_paddle_velocity = 0

        # self.num_frames=4
        self.screen_size = 600
        pygame.init()

        self.cell_size = self.screen_size // self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        # 动作空间：弹板的水平速度连续-4到4
        self.action_space = spaces.Discrete(9)  # 9 个离散动作：-4, -3, -2, -1, 0, 1, 2, 3, 4
        # 生成连续的四张图像
        self.ratio=3
        # 球的位置(0,0) 球的水平速度、球的竖直速度、挡板的位置、挡板的水平速度、砖块的位置
        # 使用一维数组表示
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -self.max_ball_speed, -self.max_ball_speed, 0, -self.max_paddle_speed] + [0]*self.grid_size*self.grid_size),
            high=np.array([self.grid_size-1, self.grid_size-1, self.max_ball_speed, self.max_ball_speed, self.grid_size-self.paddle_size, self.max_paddle_speed] + [1]*self.grid_size*self.grid_size),
            dtype=np.float32
        )

        self.reset()
        pygame.display.set_caption('Pong with Reinforcement Learning')
    def step(self, action):
        # action是下一帧的速度  所以这里计算弹板位置是上一帧的速度
        #把0~8的动作转换为-4~4的速度
        action = action-4
        # 更新弹板速度为当前的动作速度
        self.paddle_velocity = action        # 更新弹板的位置
        self.paddle_position += self.paddle_velocity
        self.paddle_position = int(np.clip(self.paddle_position, 0, self.grid_size - self.paddle_size))

        # 计算球的下一个位置
        next_ball_position = [self.ball_position[0] + self.ball_velocity[0],
                            self.ball_position[1] + self.ball_velocity[1]]
        reward = 0
        # 检查边界碰撞
        if next_ball_position[0] < 0 or next_ball_position[0] >= self.grid_size:
            self.ball_velocity[0] *= -1
        if next_ball_position[1] < 0:
            self.ball_velocity[1] *= -1
        elif next_ball_position[1] >= self.grid_size - 1:
            # 球碰到挡板
            if self.paddle_position <= next_ball_position[0] <= self.paddle_position + self.paddle_size:
                # 计算新速度
                self.ball_velocity[1] *= -1
                self.ball_velocity[0] += self.paddle_velocity #上一帧的速度 
                self.ball_velocity[0] = np.clip(self.ball_velocity[0], -self.max_ball_speed, self.max_ball_speed)
                reward += 2 + 1 - abs(self.ball_position[0] - (self.paddle_position + self.paddle_size / 2)) / self.grid_size  # 挡住球的奖励并且根据挡板和球的位置差距给额外奖励
            elif next_ball_position[1] >= self.grid_size:
                self.done = True
                reward += -50
                return self._get_obs(), reward, True, False, {}
        
        #对挡板的位置进行惩罚和奖励
        # if self.paddle_position<1 and self.paddle_velocity<0:
        #     #惩罚
        #     reward-=0.5
        # elif self.paddle_position+self.paddle_size>self.grid_size-3 and self.paddle_velocity>0:
        #     reward-=0.5
        #鼓励挡板始终在球的下方 
        if  self.paddle_position<=self.ball_position[0] and self.paddle_position+self.paddle_size>=self.ball_position[0]:
            reward+=0.5

        # 使用更新后的速度更新球的位置
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[0]=np.clip(self.ball_position[0],0,self.grid_size-1)
        self.ball_position[1] += self.ball_velocity[1]
        self.ball_position[1]=np.clip(self.ball_position[1],0,self.grid_size-1)

        # 检查砖块碰撞
        potential_hits = [
            (int(self.ball_position[1]), int(self.ball_position[0])),
            (int(self.ball_position[1]), int(self.ball_position[0] + 1)) if self.ball_position[0] + 1 < self.grid_size else None,
            (int(self.ball_position[1]), int(self.ball_position[0] - 1)) if self.ball_position[0] - 1 >= 0 else None,
            (int(self.ball_position[1] + 1), int(self.ball_position[0])) if self.ball_position[1] + 1 < self.grid_size else None,
            (int(self.ball_position[1] - 1), int(self.ball_position[0])) if self.ball_position[1] - 1 >= 0 else None,
            (int(self.ball_position[1] + 1), int(self.ball_position[0] + 1)) if self.ball_position[0] + 1 < self.grid_size and self.ball_position[1] + 1 < self.grid_size else None,
            (int(self.ball_position[1] + 1), int(self.ball_position[0] - 1)) if self.ball_position[0] - 1 >= 0 and self.ball_position[1] + 1 < self.grid_size else None,
            (int(self.ball_position[1] - 1), int(self.ball_position[0] + 1)) if self.ball_position[0] + 1 < self.grid_size and self.ball_position[1] - 1 >= 0 else None,
            (int(self.ball_position[1] - 1), int(self.ball_position[0] - 1)) if self.ball_position[0] - 1 >= 0 and self.ball_position[1] - 1 >= 0 else None,
        ]

        # 遍历可能的碰撞点并检查
        for hit in potential_hits:
            if hit is not None and self.bricks[hit] == 1:   
                self.bricks[hit] = 0  # 将砖块设置为0
                self.ball_velocity[1] *= -1
                reward += 5 + (self.grid_size - int(self.ball_position[1])) / self.grid_size * 5  # 根据砖块的位置给予额外奖励
       

        # 检查是否赢得游戏
        if np.sum(self.bricks) == 0:
            self.done = True
            reward += 100
       
        # 返回观察、奖励、是否终止、是否截断和信息字典
        return self._get_obs(), reward, self.done, False, {}
    def reset(self,seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.ball_position = [self.grid_size//2, self.grid_size-2]
        self.ball_velocity = [random.choice([-1, 1]), -1]

        self.paddle_position = (self.grid_size - self.paddle_size) // 2
        self.paddle_velocity = 0
        self.previous_paddle_velocity = 0
        self.bricks = np.zeros((self.grid_size, self.grid_size))
        brick_positions = random.sample(range(self.grid_size * (self.grid_size//2)), self.max_bricks)
        for pos in brick_positions:
            x, y = pos % self.grid_size, pos // self.grid_size
            self.bricks[y, x] = 1

        self.done = False
        
        return self._get_obs(),{}

    def _get_obs(self):
        # 生成数值观测
        brick_state = self.bricks.flatten().astype(np.float32)
        obs = np.array(
            [self.ball_position[0], self.ball_position[1], 
             self.ball_velocity[0], self.ball_velocity[1], 
             self.paddle_position, self.paddle_velocity] + 
            list(brick_state),
            dtype=np.float32
        )
        return obs

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.bricks[y, x] == 1:
                    pygame.draw.rect(self.screen, (255, 255, 255), 
                                     pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        pygame.draw.rect(self.screen, (0, 255, 0), 
                         pygame.Rect(self.paddle_position * self.cell_size, (self.grid_size - 1) * self.cell_size, self.paddle_size * self.cell_size, self.cell_size))

        pygame.draw.rect(self.screen, (255, 0, 0), 
                         pygame.Rect(self.ball_position[0] * self.cell_size, self.ball_position[1] * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

    def _generate_frame(self):
        # Calculate the size of the color image based on the grid size and cell size
        ratio=3
        image_size = self.grid_size * ratio
        color_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)  # Black background

        # Draw paddle
        paddle_color = [128, 128, 128]  # Gray paddle
        paddle_y = image_size - ratio  # Paddle is at the bottom of the image
        paddle_x_start = self.paddle_position * ratio
        paddle_x_end = paddle_x_start + self.paddle_size *ratio
        color_image[paddle_y:paddle_y + ratio, paddle_x_start:paddle_x_end] = paddle_color

        # Draw ball
        ball_color = [64, 64, 64]  # Dark gray ball
        ball_x = int(self.ball_position[0] * ratio)
        ball_y = int(self.ball_position[1] * ratio)
        color_image[ball_y:ball_y + ratio, ball_x:ball_x +ratio] = ball_color

        # Draw bricks
        brick_color = [255, 255, 255]  # White bricks
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.bricks[y, x] == 1:
                    brick_x_start = x * ratio
                    brick_y_start = y * ratio
                    brick_x_end = brick_x_start + ratio
                    brick_y_end = brick_y_start + ratio
                    color_image[brick_y_start:brick_y_end, brick_x_start:brick_x_end] = brick_color

        #单通道灰度图
        gray_image = np.mean(color_image, axis=2).astype(np.uint8)
        return gray_image  



    def close(self):
        pygame.quit()
# env = PongEnv()
# obs,_ = env.reset()
# clock=pygame.time.Clock()

# # Generate 
# for i in range(10000):
#     action = 4  # No movement
#     obs, reward, done, _, _ = env.step(action)
#     #画出frames最新的一帧
#     env.render()
#     if done:
#         obs, _ = env.reset()  
#     clock.tick(20)  # 控制帧率，每秒10帧



# env.close()