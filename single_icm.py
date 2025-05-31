"""
Super Mario Bros 강화학습 with ICM (Intrinsic Curiosity Module)

이 파일은 ICM을 활용한 Mario 강화학습 구현입니다.

주요 개선사항:
1. ICM 논문에 따른 손실 함수 가중치 적용: (1-β)*L_I + β*L_F
2. 내재적 보상 정규화 옵션 추가로 안정성 향상
3. 하이퍼파라미터 세분화로 조정 가능성 증대
4. 더 나은 가독성을 위한 코드 구조 개선

ICM 구성 요소:
- FeatureExtractor: 상태를 저차원 특징으로 압축
- InverseModel: 연속된 두 상태 특징으로부터 행동 예측
- ForwardModel: 현재 상태 특징과 행동으로부터 다음 상태 특징 예측
"""

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import numpy as np
from pathlib import Path
import datetime, time

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import matplotlib.pyplot as plt

# Constants and hyperparameters
RENDER_MODE = 'rgb_array'  # Options: 'human', 'rgb_array'
MARIO_LEVEL = "SuperMarioBros-1-1-v0"
FRAME_SKIP = 4
RESIZE_SHAPE = 84

# Training parameters
EXPLORATION_RATE_INITIAL = 1.0
EXPLORATION_RATE_DECAY = 0.99999
EXPLORATION_RATE_MIN = 0.1
GAMMA = 0.9  # Discount factor
BATCH_SIZE = 128
BURNIN = 5000  # Min experiences before training
LEARN_EVERY = 5  # Experiences between Q_online updates
SYNC_EVERY = 10000  # Experiences between Q_target & Q_online sync
SAVE_EVERY = 10000  # Experiences between saving model
MEMORY_SIZE = 100000
MAX_STEPS = 4000000  # Maximum training steps

# Learning parameters
LEARNING_RATE = 0.00025
USE_ICM = True # ICM 사용 여부 결정
ICM_LR_SCALE = 1.0  # ICM 모델 학습률 스케일 (기존 LR에 곱해짐)
BETA_ICM = 0.2  # 내재적 보상 가중치
ICM_FEATURE_SIZE = 256 # Feature extractor의 출력 크기

ICM_BETA_LOSS = 0.2  # Forward Model 손실 가중치 (논문 기준)
ICM_NORMALIZE_INTRINSIC_REWARD = True  # 내재적 보상 정규화 여부

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
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

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
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_size):
        super().__init__()
        c, h, w = input_dim

        if h != RESIZE_SHAPE:
            raise ValueError(f"Expecting input height: {RESIZE_SHAPE}, got: {h}")
        if w != RESIZE_SHAPE:
            raise ValueError(f"Expecting input width: {RESIZE_SHAPE}, got: {w}")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # CNN 출력 크기 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
        
        self.fc = nn.Linear(cnn_out_dim, feature_size)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class InverseModel(nn.Module):
    def __init__(self, feature_size, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size * 2, 256), # phi_t와 phi_t+1을 concat
            nn.ReLU(),
            nn.Linear(256, action_dim) # 행동 개수만큼 출력
        )

    def forward(self, phi_t, phi_t_plus_1):
        x = torch.cat([phi_t, phi_t_plus_1], dim=1)
        return self.fc(x)

class ForwardModel(nn.Module):
    def __init__(self, feature_size, action_dim):
        super().__init__()
        # 행동을 embedding하거나 one-hot으로 변환 후 feature와 결합
        # 여기서는 action_dim을 직접 사용 (나중에 one-hot으로 변환하여 concat할 수 있음)
        self.fc = nn.Sequential(
            nn.Linear(feature_size + action_dim, 256), # phi_t와 action을 concat
            nn.ReLU(),
            nn.Linear(256, feature_size) # 다음 상태의 feature 예측
        )

    def forward(self, phi_t, action_one_hot):
        x = torch.cat([phi_t, action_one_hot], dim=1)
        return self.fc(x)

class ICM(nn.Module):
    def __init__(self, input_dim, action_dim, feature_size=ICM_FEATURE_SIZE, lr_scale=ICM_LR_SCALE, device="cpu"):
        super().__init__()
        self.action_dim = action_dim
        self.device = device

        self.feature_extractor = FeatureExtractor(input_dim, feature_size).to(device)
        self.inverse_model = InverseModel(feature_size, action_dim).to(device)
        self.forward_model = ForwardModel(feature_size, action_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.inverse_model.parameters()) +
            list(self.forward_model.parameters()),
            lr=LEARNING_RATE * lr_scale
        )

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def _to_one_hot(self, actions):
        """
        actions: (batch_size, 1) or (batch_size,) 형태의 텐서
        """
        actions = actions.long()
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        one_hot_actions = torch.zeros(actions.size(0), self.action_dim, device=self.device)
        one_hot_actions.scatter_(1, actions, 1)
        return one_hot_actions

    def compute_intrinsic_reward(self, state_t, state_t_plus_1, action_t):
        """
        내재적 보상을 계산합니다.
        state_t, state_t_plus_1: (N, C, H, W)
        action_t: (N,) or (N, 1), 실제 수행된 액션 인덱스
        """
        state_t = state_t.to(self.device)
        state_t_plus_1 = state_t_plus_1.to(self.device)
        action_t = action_t.to(self.device)

        with torch.no_grad(): # 내재적 보상 계산 시에는 그래디언트 흐름 X
            phi_t = self.feature_extractor(state_t)
            phi_t_plus_1_actual = self.feature_extractor(state_t_plus_1)
            
            action_t_one_hot = self._to_one_hot(action_t)
            
            phi_t_plus_1_pred = self.forward_model(phi_t, action_t_one_hot)
            
            # 내재적 보상: 예측된 다음 상태 특징과 실제 다음 상태 특징 간의 MSE
            intrinsic_reward = 0.5 * torch.mean((phi_t_plus_1_pred - phi_t_plus_1_actual)**2, dim=1)
            
            # 정규화 옵션 적용
            if ICM_NORMALIZE_INTRINSIC_REWARD and intrinsic_reward.numel() > 1:
                # 배치 내에서 정규화 (평균 0, 표준편차 1)
                reward_mean = intrinsic_reward.mean()
                reward_std = intrinsic_reward.std() + 1e-8  # 수치 안정성을 위한 작은 값 추가
                intrinsic_reward = (intrinsic_reward - reward_mean) / reward_std
                # 양수로 변환 (내재적 보상은 항상 양수여야 함)
                intrinsic_reward = torch.exp(intrinsic_reward)
                
        return intrinsic_reward # (N,) 형태의 텐서

    def train_batch(self, state_t, state_t_plus_1, action_t):
        """
        ICM 모듈을 학습합니다.
        ICM 논문에 따라 손실 함수는 (1-β)*L_I + β*L_F 형태로 구성됩니다.
        state_t, state_t_plus_1: (N, C, H, W)
        action_t: (N,) or (N, 1), 실제 수행된 액션 인덱스
        """
        state_t = state_t.to(self.device)
        state_t_plus_1 = state_t_plus_1.to(self.device)
        action_t = action_t.to(self.device) 

        phi_t = self.feature_extractor(state_t)
        phi_t_plus_1_actual = self.feature_extractor(state_t_plus_1)

        # Forward Model 손실
        action_t_one_hot = self._to_one_hot(action_t)
        phi_t_plus_1_pred = self.forward_model(phi_t.detach(), action_t_one_hot)
        loss_fwd = self.mse_loss(phi_t_plus_1_pred, phi_t_plus_1_actual.detach())

        # Inverse Model 손실
        pred_action_logits = self.inverse_model(phi_t, phi_t_plus_1_actual)
        loss_inv = self.ce_loss(pred_action_logits, action_t.long().squeeze())
        
        # ICM 논문에 따른 총 손실: (1-β)*L_I + β*L_F
        total_loss = (1 - ICM_BETA_LOSS) * loss_inv + ICM_BETA_LOSS * loss_fwd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss_fwd.item(), loss_inv.item()


class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self._build_net(c, output_dim)
        self.target = self._build_net(c, output_dim)

        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def _build_net(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=64, kernel_size=8, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # Windows에서 CUDA 최적화
        if torch.cuda.is_available():
            self.device = "cuda"
            # CUDA 스트림 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # DDQN Network
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        
        # ICM Module
        self.icm = None
        if USE_ICM:
            self.icm = ICM(state_dim, action_dim, device=self.device)

        self.exploration_rate = EXPLORATION_RATE_INITIAL
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min = EXPLORATION_RATE_MIN
        self.curr_step = 0

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(MEMORY_SIZE, device=torch.device("cpu")))
        self.batch_size = BATCH_SIZE

        self.gamma = GAMMA
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = BURNIN
        self.learn_every = LEARN_EVERY
        self.sync_every = SYNC_EVERY
        self.save_every = SAVE_EVERY


    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_np = state.__array__()
            state_tensor = torch.tensor(state_np, device=self.device).unsqueeze(0)
            action_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state_np = state.__array__()
        next_state_np = next_state.__array__()

        state_tensor = torch.tensor(state_np)
        next_state_tensor = torch.tensor(next_state_np)
        action_tensor = torch.tensor([action])
        reward_tensor = torch.tensor([reward]) 
        done_tensor = torch.tensor([done])
        
        intrinsic_reward_value = 0.0 # 기본값
        if USE_ICM and self.icm:
            intrinsic_reward_value = self.icm.compute_intrinsic_reward(
                state_tensor.unsqueeze(0).to(self.icm.device), 
                next_state_tensor.unsqueeze(0).to(self.icm.device), 
                action_tensor.to(self.icm.device)
            ).cpu().item() # 스칼라 값으로 변환

        intrinsic_reward_tensor = torch.tensor([intrinsic_reward_value])

        self.memory.add(TensorDict({
            "state": state_tensor, 
            "next_state": next_state_tensor, 
            "action": action_tensor, 
            "reward": reward_tensor, 
            "done": done_tensor,
            "intrinsic_reward": intrinsic_reward_tensor 
        }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done, intrinsic_reward = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done", "intrinsic_reward")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), intrinsic_reward.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action.long()
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done, intrinsic_reward):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        
        total_reward = reward
        if USE_ICM:
            total_reward = reward + BETA_ICM * intrinsic_reward 
        
        return (total_reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0 and self.curr_step > 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None, None, None 

        if self.curr_step % self.learn_every != 0:
            return None, None, None, None 

        state, next_state, action, reward, done, intrinsic_reward = self.recall()
        
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done, intrinsic_reward)
        ddqn_loss = self.update_Q_online(td_est, td_tgt)

        icm_fwd_loss, icm_inv_loss = None, None
        if USE_ICM and self.icm:
            icm_fwd_loss, icm_inv_loss = self.icm.train_batch(state, next_state, action)

        return (td_est.mean().item(), ddqn_loss, icm_fwd_loss, icm_inv_loss)


class MetricLogger:
    def __init__(self, save_dir):
        # TensorBoard 디렉토리 이름 설정
        tb_log_dir_name = "tensorboard_icm" if USE_ICM else "tensorboard_no_icm"
        self.writer = SummaryWriter(log_dir=str(save_dir / tb_log_dir_name))
        
        self.save_log = save_dir / "log.txt"
        log_header = (
            f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
            f"{'MeanLength':>15}{'MeanDDQNLoss':>15}{'MeanQValue':>15}"
        )
        if USE_ICM:
            log_header += f"{'MeanICMFwdLoss':>18}{'MeanICMInvLoss':>18}"
        log_header += f"{'TimeDelta':>15}{'Time':>20}\n"
        
        with open(self.save_log, "w") as f:
            f.write(log_header)

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg" # DDQN Loss
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        
        if USE_ICM:
            self.icm_fwd_losses_plot = save_dir / "icm_fwd_loss_plot.jpg"
            self.icm_inv_losses_plot = save_dir / "icm_inv_loss_plot.jpg"
            self.ep_avg_icm_fwd_losses = []
            self.ep_avg_icm_inv_losses = []
            self.moving_avg_ep_avg_icm_fwd_losses = []
            self.moving_avg_ep_avg_icm_inv_losses = []
            self.curr_ep_icm_fwd_loss = 0.0
            self.curr_ep_icm_inv_loss = 0.0
            self.curr_ep_icm_loss_length = 0.0


        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        
        self.init_episode() # 여기서 curr_ep_icm 관련 변수들도 초기화 (USE_ICM 조건부)
        self.record_time = time.time()

    def log_model_visualization(self, model, device, input_shape=(4, 84, 84), model_name="DDQN_Online"):
        try:
            dummy_input = torch.randn(1, *input_shape).to(device)
            self.writer.add_graph(model, dummy_input)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            layer_info = [f"{name}: {param.numel():,} parameters" for name, param in model.named_parameters()]

            param_summary = f"""
{model_name} Model Parameter Summary:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Non-trainable Parameters: {total_params - trainable_params:,}

Layer-wise Parameters:
{chr(10).join(layer_info)}
            """
            self.writer.add_text(f"Model/{model_name}_Parameter_Summary", param_summary)
            print(f"{model_name} model visualization logged to TensorBoard")

        except Exception as e:
            print(f"Warning: Failed to log {model_name} model visualization: {e}")

    def log_step(self, reward, loss, q, icm_fwd_loss=None, icm_inv_loss=None):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None and q is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1
        
        if USE_ICM and icm_fwd_loss is not None and icm_inv_loss is not None:
            self.curr_ep_icm_fwd_loss += icm_fwd_loss
            self.curr_ep_icm_inv_loss += icm_inv_loss
            self.curr_ep_icm_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0.0
            ep_avg_q = 0.0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        if USE_ICM:
            if self.curr_ep_icm_loss_length == 0:
                ep_avg_icm_fwd_loss = 0.0
                ep_avg_icm_inv_loss = 0.0
            else:
                ep_avg_icm_fwd_loss = np.round(self.curr_ep_icm_fwd_loss / self.curr_ep_icm_loss_length, 5)
                ep_avg_icm_inv_loss = np.round(self.curr_ep_icm_inv_loss / self.curr_ep_icm_loss_length, 5)
            self.ep_avg_icm_fwd_losses.append(ep_avg_icm_fwd_loss)
            self.ep_avg_icm_inv_losses.append(ep_avg_icm_inv_loss)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0.0
        if USE_ICM:
            self.curr_ep_icm_fwd_loss = 0.0
            self.curr_ep_icm_inv_loss = 0.0
            self.curr_ep_icm_loss_length = 0.0

    def _get_mean_metric(self, metric_list, window=100):
        if not metric_list:
            return 0.0
        return np.round(np.mean(metric_list[-window:]), 3)

    def _plot_metric(self, data, metric_name_base, plot_save_path, episode_label="Episode"):
        plt.figure()
        label_metric_name = metric_name_base.replace("avg_", "avg ").replace("_", " ").title()
        plt.plot(data, label=f"Moving Avg of {label_metric_name}")
        plt.xlabel(episode_label)
        plt.ylabel(label_metric_name)
        plt.title(f"Moving Average of {label_metric_name} over Episodes")
        plt.legend()
        plt.savefig(plot_save_path)
        plt.close()

    def record(self, episode, epsilon, step):
        mean_ep_reward = self._get_mean_metric(self.ep_rewards)
        mean_ep_length = self._get_mean_metric(self.ep_lengths)
        mean_ep_loss = self._get_mean_metric(self.ep_avg_losses) 
        mean_ep_q = self._get_mean_metric(self.ep_avg_qs)
        
        self.writer.add_scalar('Metrics/Mean Reward (Extrinsic)', mean_ep_reward, episode)
        self.writer.add_scalar('Metrics/Mean Length', mean_ep_length, episode)
        self.writer.add_scalar('Metrics/Mean DDQN Loss', mean_ep_loss, episode)
        self.writer.add_scalar('Metrics/Mean Q Value', mean_ep_q, episode)
        self.writer.add_scalar('Metrics/Epsilon', epsilon, episode)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss) 
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        log_print_str = (
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean DDQN Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
        )
        
        log_file_str = (
            f"{episode:8d}{step:8d}{epsilon:10.3f}"
            f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
        )

        if USE_ICM:
            mean_ep_icm_fwd_loss = self._get_mean_metric(self.ep_avg_icm_fwd_losses)
            mean_ep_icm_inv_loss = self._get_mean_metric(self.ep_avg_icm_inv_losses)
            self.writer.add_scalar('Metrics/Mean ICM Forward Loss', mean_ep_icm_fwd_loss, episode)
            self.writer.add_scalar('Metrics/Mean ICM Inverse Loss', mean_ep_icm_inv_loss, episode)
            self.moving_avg_ep_avg_icm_fwd_losses.append(mean_ep_icm_fwd_loss)
            self.moving_avg_ep_avg_icm_inv_losses.append(mean_ep_icm_inv_loss)
            log_print_str += (
                f"Mean ICM Fwd Loss {mean_ep_icm_fwd_loss} - "
                f"Mean ICM Inv Loss {mean_ep_icm_inv_loss} - "
            )
            log_file_str += f"{mean_ep_icm_fwd_loss:18.5f}{mean_ep_icm_inv_loss:18.5f}"

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        log_print_str += (
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        print(log_print_str)
        
        log_file_str += (
            f"{time_since_last_record:15.3f}"
            f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
        )
        with open(self.save_log, "a") as f:
            f.write(log_file_str)

        plot_metrics = ["rewards", "lengths", "avg_losses", "avg_qs"]
        if USE_ICM:
            plot_metrics.extend(["avg_icm_fwd_losses", "avg_icm_inv_losses"])

        for metric_name_base in plot_metrics:
            if "icm" in metric_name_base:
                plot_label_base = metric_name_base.replace("avg_", "").replace("_", " ").title()
                data_list_attr = f"moving_avg_ep_{metric_name_base}"
                plot_path_attr = f"{metric_name_base.replace('avg_', '')}_plot" 
            else: 
                plot_label_base = metric_name_base
                data_list_attr = f"moving_avg_ep_{metric_name_base}"
                plot_path_attr = f"ep_{metric_name_base}_plot"

            data_list = getattr(self, data_list_attr)
            plot_path = getattr(self, plot_path_attr)
            self._plot_metric(data_list, plot_label_base, plot_path)

    def close(self):
        self.writer.close()


def main():
    # Initialize Super Mario environment
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(MARIO_LEVEL, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(MARIO_LEVEL, render_mode=RENDER_MODE, apply_api_compatibility=True)

    # Limit the action-space to 3 discrete actions: ["right"], ["right", "A"], ["A"]
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])

    # Apply Wrappers
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=RESIZE_SHAPE)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    # 저장 디렉토리 이름에 ICM 사용 여부 반영
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    icm_suffix = "_icm" if USE_ICM else "_no_icm"
    save_dir_name = f"{current_time_str}{icm_suffix}"
    save_dir = Path("checkpoints") / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    mario = Mario(state_dim=(4, RESIZE_SHAPE, RESIZE_SHAPE), action_dim=env.action_space.n, save_dir=save_dir)
    logger = MetricLogger(save_dir) # 수정된 save_dir 전달

    logger.log_model_visualization(mario.net.online, mario.device, model_name="DDQN_Online")
    if USE_ICM and mario.icm:
        logger.log_model_visualization(mario.icm.feature_extractor, mario.device, model_name="ICM_FeatureExtractor")
        # ForwardModel과 InverseModel은 입력 형태가 단순 상태가 아니므로, 필요시 더미 입력 조정 필요
        # dummy_phi = torch.randn(1, ICM_FEATURE_SIZE).to(mario.device)
        # dummy_action_one_hot = mario.icm._to_one_hot(torch.zeros(1, dtype=torch.long).to(mario.device))
        # logger.log_model_visualization(mario.icm.forward_model, mario.device, input_shape=(dummy_phi, dummy_action_one_hot), model_name="ICM_ForwardModel")
        # logger.log_model_visualization(mario.icm.inverse_model, mario.device, input_shape=(dummy_phi, dummy_phi), model_name="ICM_InverseModel")


    print(f"Starting training for {MAX_STEPS:,} steps...")

    def _get_obs(obs_or_tuple):
        """환경에서 반환된 관측값 또는 (관측값, 정보) 튜플에서 실제 관측값을 추출합니다."""
        if isinstance(obs_or_tuple, tuple):
            return obs_or_tuple[0]
        return obs_or_tuple

    episode = 0
    while mario.curr_step < MAX_STEPS:
        state_tuple = env.reset()
        state = _get_obs(state_tuple)

        env.render()
        while True:
            action = mario.act(state)
            step_result = env.step(action)
            next_state = _get_obs(step_result[0])
            reward = step_result[1]
            done = step_result[2]
            trunc = step_result[3]
            info = step_result[4]

            mario.cache(state, next_state, action, reward, done)
            
            q, ddqn_loss, icm_fwd_loss, icm_inv_loss = mario.learn() 
            
            logger.log_step(reward, ddqn_loss, q, icm_fwd_loss, icm_inv_loss)
            state = next_state

            if done or trunc or info.get("flag_get", False):
                break

        logger.log_episode()

        if (episode % 20 == 0) or (mario.curr_step >= MAX_STEPS):
            logger.record(episode=episode, epsilon=mario.exploration_rate, step=mario.curr_step)

        episode += 1

    logger.close()
    print("Training finished.")

if __name__ == "__main__":
    main()