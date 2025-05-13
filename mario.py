import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, time, os
import matplotlib.pyplot as plt

# Gym은 강화학습을 위한 OpenAI 툴킷입니다.
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# OpenAI Gym을 위한 NES 에뮬레이터
from nes_py.wrappers import JoypadSpace

# OpenAI Gym에서의 슈퍼 마리오 환경 세팅
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import gc

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """모든 `skip` 프레임만 반환합니다."""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """행동을 반복하고 포상을 더합니다."""
        total_reward = 0.0
        for i in range(self._skip):
            # 포상을 누적하고 동일한 작업을 반복합니다.
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
        # [H, W, C] 배열을 [C, H, W] 텐서로 바꿉니다.
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
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


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # 마리오의 DNN은 최적의 행동을 예측합니다 - 이는 학습하기 섹션에서 구현합니다.
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999
        #self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e4  # Mario Net 저장 사이의 경험 횟수

    def act(self, state):
        """
    주어진 상태에서, 입실론-그리디 행동(epsilon-greedy action)을 선택하고, 스텝의 값을 업데이트 합니다.

    입력값:
    state (``LazyFrame``): 현재 상태에서의 단일 상태(observation)값을 말합니다. 차원은 (state_dim)입니다.
    출력값:
    ``action_idx`` (int): Mario가 수행할 행동을 나타내는 정수 값입니다.
    """
        # 임의의 행동을 선택하기
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 최적의 행동을 이용하기
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # exploration_rate 감소하기
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # 스텝 수 증가하기
        self.curr_step += 1
        return action_idx

#===================
#User defined reward function
def calculate_reward(reward, info, done, prev_info=None):
    total_reward = reward * 0.1  # 기본 보상 스케일 조정
    
    # 목표 달성 보너스
    if info["flag_get"]:
        total_reward += 1000  # 깃발 획득 보너스
    
    # 코인 획득 보너스
    if prev_info and info["coins"] > prev_info["coins"]:
        total_reward += 50  # 코인당 50점 보너스
    
    # 생존 보너스
    if not done:
        total_reward += 1.0  # 매 프레임마다 1점 보너스
    
    # 진행도 보너스
    if prev_info and info["x_pos"] > prev_info["x_pos"]:
        total_reward += 10  # 오른쪽으로 진행할 때마다 10점 보너스
    
    # 생명 감소 패널티
    if prev_info and info["life"] < prev_info["life"]:
        total_reward -= 200  # 생명 감소시 200점 패널티
    
    # 추가 보상
    if prev_info:
        # 높이 증가 보너스
        if info["y_pos"] < prev_info["y_pos"]:  # 위로 올라갈 때
            total_reward += 5
        
        # 속도 보너스
        if info["x_pos"] - prev_info["x_pos"] > 5:  # 빠른 이동
            total_reward += 15
    
    # 보상 클리핑
    total_reward = np.clip(total_reward, -1000, 1000)
    
    return total_reward


class Mario(Mario):  # 연속성을 위한 하위 클래스입니다.
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(50000, device=torch.device("cpu")))
        self.batch_size = 32
        self.prev_info = None

    
    def cache(self, state, next_state, action, reward, done, info):
        """
        Store the experience to self.memory (replay buffer)

        입력값:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        # 리워드 계산
        reward = calculate_reward(reward, info, done, self.prev_info)
        self.prev_info = info.copy()
        
        # 기존 캐시 로직
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({
            "state": state, 
            "next_state": next_state, 
            "action": action, 
            "reward": reward, 
            "done": done
        }, batch_size=[]))


    def recall(self):
        """
        메모리에서 일련의 경험들을 검색합니다.
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

class MarioNet(nn.Module):
    """작은 CNN 구조
  입력 -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> 출력
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target 매개변수 값은 고정시킵니다.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        # discount_factor <- greedy 하게 해도 충분히 좋지 않을까? 당장 안 죽는게 더 중요하잖아.
        self.gamma = 0.7

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

class Mario(Mario):
    def save(self):
        if self.curr_step % self.save_every == 0:
            # 이전 체크포인트 삭제
            for old_checkpoint in self.save_dir.glob("mario_net_*.chkpt"):
                if old_checkpoint != self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt":
                    old_checkpoint.unlink()
        
            # 새로운 체크포인트 저장
            save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
            torch.save(
                dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
                save_path,
            )
            print(f"MarioNet saved to {save_path} at step {self.curr_step}")

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # 학습을 진행하기 전 최소한의 경험값.
        self.learn_every = 3  # Q_online 업데이트 사이의 경험 횟수.
        self.sync_every = 1e4  # Q_target과 Q_online sync 사이의 경험 수

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            gc.collect()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # 메모리로부터 샘플링을 합니다.
        state, next_state, action, reward, done = self.recall()

        # TD 추정값을 가져옵니다.
        td_est = self.td_estimate(state, action)

        # TD 목표값을 가져옵니다.
        td_tgt = self.td_target(reward, next_state, done)

        # 실시간 Q(Q_online)을 통해 역전파 손실을 계산합니다.
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)



class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

         # 텐서보드 writer 초기화
        self.writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
        
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # 지표(Metric)와 관련된 리스트입니다.
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # 모든 record() 함수를 호출한 후 이동 평균(Moving average)을 계산합니다.
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # 메모리 관리 설정
        self.max_history_length = 1000  # 최대 저장할 에피소드 수
        self.cleanup_interval = 100     # 몇 에피소드마다 정리할지
        self.last_cleanup = 0           # 마지막 정리 시점

        # 현재 에피스드에 대한 지표를 기록합니다.
        self.init_episode()

        # 시간에 대한 기록입니다.
        self.record_time = time.time()

    def cleanup_old_data(self, current_episode):
        """오래된 데이터 정리"""
        if current_episode - self.last_cleanup >= self.cleanup_interval:
            # 이동 평균 데이터는 유지하고 나머지 데이터 정리
            if len(self.ep_rewards) > self.max_history_length:
                self.ep_rewards = self.ep_rewards[-self.max_history_length:]
            if len(self.ep_lengths) > self.max_history_length:
                self.ep_lengths = self.ep_lengths[-self.max_history_length:]
            if len(self.ep_avg_losses) > self.max_history_length:
                self.ep_avg_losses = self.ep_avg_losses[-self.max_history_length:]
            if len(self.ep_avg_qs) > self.max_history_length:
                self.ep_avg_qs = self.ep_avg_qs[-self.max_history_length:]
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.last_cleanup = current_episode
            print(f"Memory cleanup performed at episode {current_episode}")


    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "에피스드의 끝을 표시합니다."
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        # 메모리 정리 수행
        self.cleanup_old_data(episode)

        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)

        # 텐서보드에 기록
        self.writer.add_scalar('Metrics/Mean Reward', mean_ep_reward, episode)
        self.writer.add_scalar('Metrics/Mean Length', mean_ep_length, episode)
        self.writer.add_scalar('Metrics/Mean Loss', mean_ep_loss, episode)
        self.writer.add_scalar('Metrics/Mean Q Value', mean_ep_q, episode)
        self.writer.add_scalar('Metrics/Epsilon', epsilon, episode)
        
        # 이동 평균
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))
    def close(self):
        self.writer.close()
        # 마지막 메모리 정리
        gc.collect()
        if torch.backends.mps.is_available():
            torch.backends.mps.empty_cache()



class ParallelEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.envs = []
        self.queues = []
        
        for _ in range(num_envs):
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
            env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])
            env = SkipFrame(env, skip=4)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, shape=84)
            env = FrameStack(env, num_stack=4)
            self.envs.append(env)
            self.queues.append(Queue())
    
    def reset(self):
        states = []
        for env in self.envs:
            state = env.reset()
            states.append(state)
        return states
    
    def step(self, actions):
        states = []
        rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, trunc, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            if done:
                state = env.reset()
                states[i] = state
        
        return states, rewards, dones, infos


def train_worker(worker_id, shared_memory, num_episodes):
    # 각 워커별 환경 초기화
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    # 에이전트 초기화
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=Path("checkpoints"))
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        prev_info = info
        
        while True:
            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)

            # 리워드 계산
            reward = calculate_reward(reward, info, done, prev_info)
            
            
            # 수정된 캐시 호출
            mario.cache(state, next_state, action, reward, done, info)
            
            q, loss = mario.learn()
            
            episode_reward += reward
            state = next_state
            prev_info = info
            
            if done or info["flag_get"]:
                break
        
        # 공유 메모리에 결과 저장
        shared_memory.put({
            'worker_id': worker_id,
            'episode': episode,
            'reward': episode_reward,
            'epsilon': mario.exploration_rate
        })

def main():

     # 메인 환경 초기화
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    num_workers = 4  # 병렬로 실행할 워커 수
    num_episodes = 10000  # 각 워커당 에피소드 수
    
    # 공유 메모리 초기화
    shared_memory = Queue(maxsize=1000)
    
    # 워커 프로세스 생성
    processes = []
    for i in range(num_workers):
        p = Process(target=train_worker, args=(i, shared_memory, num_episodes))
        p.start()
        processes.append(p)
    
    # 메인 에이전트 초기화
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=Path("checkpoints"))
    logger = MetricLogger(save_dir=Path("checkpoints"))
    
    # 결과 수집 및 로깅
    total_episodes = 0
    while total_episodes < num_workers * num_episodes:
        result = shared_memory.get()
        logger.log_episode()
        logger.record(
            episode=result['episode'],
            epsilon=result['epsilon'],
            step=mario.curr_step
        )
        total_episodes += 1
    
    # 프로세스 종료
    for p in processes:
        p.join()
    
    logger.close()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정되어 있는 경우 무시
    main()