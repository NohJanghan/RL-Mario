import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.optim as optim
import torchvision.transforms as T
from gym.spaces import Box
from torch.distributions.categorical import Categorical
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
import numpy as np
from pathlib import Path
import time
from PIL import Image

# 환경 래퍼 클래스들
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float32)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Actor 네트워크 정의
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

def create_mario_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env

def load_model(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state_dim = (4, 84, 84)
    action_dim = 3  # SIMPLE_MOVEMENT의 액션 수
    
    model = ActorNetwork(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.to(device)
    model.eval()
    return model

def play_episode(model, env, max_steps=1000):
    state, info = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:
        # 상태 전처리
        if isinstance(state, tuple):
            state = state[0]
        state_tensor = torch.FloatTensor(state.__array__()).unsqueeze(0).to(model.device)
        
        # 행동 선택
        action = model.get_action(state_tensor)
        
        # 환경과 상호작용
        next_state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        
        # 화면 업데이트
        env.render()
        
        state = next_state
        step += 1
        
        # 골 도착 시 종료
        if info.get("flag_get", False):
            print("🎉 골 도착!")
            break
    
    return total_reward, step

def main():
    # 모델 경로 설정
    models_path = Path("./checkpoints")  # 실제 저장된 모델 경로로 수정

    for model_path in sorted(models_path.glob("*.chkpt")):

        # 환경 생성
        env = create_mario_env()
        
        # 모델 로드
        model = load_model(model_path)
    
        # 에피소드 실행
        print("게임 시작!")
        reward, steps = play_episode(model, env)
        print(f"에피소드 종료 - 총 보상: {reward:.2f}, 스텝 수: {steps}")
    
        env.close()

if __name__ == "__main__":
    main()
