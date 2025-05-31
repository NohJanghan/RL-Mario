import torch
import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from pathlib import Path
import time
import random
import numpy as np

# single.py에서 필요한 클래스들 import
from single import (
    Mario,
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    # MARIO_LEVEL,
    FRAME_SKIP,
    RESIZE_SHAPE
)
MARIO_LEVEL = "SuperMarioBros-1-1-v0"
CHECKPOINT_PATH = "test_single/mario_net_400.chkpt"

def set_seeds(seed=42):
    """재현성을 위해 모든 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_mario_model(checkpoint_path, state_dim, action_dim):
    """체크포인트에서 Mario 모델을 로드합니다."""
    # 임시 save_dir (추론에서는 사용하지 않음)

    # Mario 인스턴스 생성
    mario = Mario(state_dim=state_dim, action_dim=action_dim, save_dir=None)

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=mario.device)
    mario.net.load_state_dict(checkpoint['model'])
    mario.exploration_rate = 0.0  # 추론 모드에서는 탐험하지 않음

    print(f"Model loaded from {checkpoint_path}")
    print(f"Using device: {mario.device}")

    return mario

def test_mario(checkpoint_path, num_episodes=5, render=True, seed=42):
    """로드된 모델로 Mario를 테스트합니다."""

    # 시드 고정
    set_seeds(seed)
    print(f"Seeds set to {seed} for reproducibility")

    # 환경 설정 (single.py와 동일)
    render_mode = 'human' if render else 'rgb_array'

    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(MARIO_LEVEL, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(MARIO_LEVEL, render_mode=render_mode, apply_api_compatibility=True)

    # 액션 공간 제한
    env = JoypadSpace(env, [["right"], ["right", "A"], ["A"]])

    # 래퍼 적용
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=RESIZE_SHAPE)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    # 모델 로드
    mario = load_mario_model(
        checkpoint_path=checkpoint_path,
        state_dim=(4, RESIZE_SHAPE, RESIZE_SHAPE),
        action_dim=env.action_space.n
    )

    def _get_obs(obs_or_tuple):
        """환경에서 반환된 관측값 또는 (관측값, 정보) 튜플에서 실제 관측값을 추출합니다."""
        if isinstance(obs_or_tuple, tuple):
            return obs_or_tuple[0]
        return obs_or_tuple

    # 테스트 실행
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state_tuple = env.reset()
        state = _get_obs(state_tuple)

        total_reward = 0
        total_length = 0

        print(f"\nEpisode {episode + 1}/{num_episodes} started...")
        env.render()
        while True:

            # 모델로 액션 예측 (탐험 없이)
            action = mario.act(state)

            # 환경에서 스텝 실행
            step_result = env.step(action)
            next_state = _get_obs(step_result[0])
            reward = step_result[1]
            done = step_result[2]
            trunc = step_result[3]
            info = step_result[4]

            total_reward += reward
            total_length += 1

            state = next_state

            # 에피소드 종료 조건
            if done or trunc or info.get("flag_get", False):
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(total_length)

        print(f"Episode {episode + 1} finished!")
        print(f"  Total Reward: {total_reward}")
        print(f"  Total Length: {total_length}")
        print(f"  Flag Get: {info.get('flag_get', False)}")
        print(f"  X Position: {info.get('x_pos', 'N/A')}")

    env.close()

    # 결과 요약
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_length = sum(episode_lengths) / len(episode_lengths)

    print(f"\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.2f}")
    print(f"Episode Rewards: {episode_rewards}")
    print(f"Episode Lengths: {episode_lengths}")

def main():
    # 체크포인트 파일 경로 설정
    # 실제 체크포인트 파일 경로로 수정해주세요
    checkpoint_path = CHECKPOINT_PATH

    # 체크포인트 파일이 존재하는지 확인
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please check the path and make sure the file exists.")

        # checkpoints 폴더에서 사용 가능한 체크포인트 찾기
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            print(f"\nAvailable checkpoint folders in {checkpoints_dir}:")
            for folder in checkpoints_dir.iterdir():
                if folder.is_dir():
                    print(f"  {folder.name}")
                    for chkpt_file in folder.glob("*.chkpt"):
                        print(f"    - {chkpt_file.name}")
        return

    # 테스트 실행
    test_mario(
        checkpoint_path=checkpoint_path,
        num_episodes=3,  # 테스트할 에피소드 수
        render=True,     # 화면 렌더링 여부
        seed=42          # 재현성을 위한 시드
    )

if __name__ == "__main__":
    main()