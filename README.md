# 강화학습 기반 슈퍼 마리오 게임

이 프로젝트는 강화학습을 사용하여 슈퍼 마리오 게임을 자동으로 플레이하는 AI를 구현합니다. 
강화학습에 사용되는 다양한 알고리즘의 학습 및 적용을 시도해 보려고 합니다.

아래와 같은 개념을 공부하고 적용하고 있습니다.
1. epsilon-greedy
2. Q-learning
3. Actor-Critic (공부 중,,)
4. PPO (공부 중,,)


## 프로젝트 구조

### 1. 핵심 파일
- `mario.py`: 마리오 환경 및 에이전트 구현
- `PPOmario.py`: PPO 알고리즘 구현 및 학습 중,,,
- `main.ipynb`: 메인 학습 스크립트
- `inference.py`: 학습된 모델 추론

### 2. 환경 설정
- `pyproject.toml`: 프로젝트 의존성
- `.python-version`: Python 버전
- `.venv/`: 가상 환경

### 3. 체크포인트
- `checkpoints/`: 학습된 모델 저장

## 필요 조건

- Python 3.8 이상
- PyTorch
- Gym
- NES-py
- TensorBoard
- Matplotlib

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
```

2. 가상 환경을 생성하고 활성화합니다:
```bash
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 필요한 패키지를 설치합니다:
```bash
uv sync
```

## 사용 방법

### 학습 실행
```bash
uv run main.py
```

### 추론 실행
```bash
uv run inference.py
```

## 주요 기능

1. **환경 래퍼**
   - 프레임 스킵
   - 그레이스케일 변환
   - 관측 리사이징
   - 프레임 스택

2. **에이전트 구현**
   - DQN 네트워크
   - 경험 리플레이
   - 입실론-그리디 탐험
   - 타겟 네트워크

3. **보상 함수**
   - 목표 달성 보너스
   - 코인 획득 보너스
   - 생존 보너스
   - 진행도 보너스
   - 패널티 시스템

4. **학습 최적화**
   - 배치 정규화
   - 경험 리플레이
   - 타겟 네트워크 동기화
   - 학습률 스케줄링

## 구현 세부사항

### 환경 래퍼
```python
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
```

### 보상 함수
```python
def calculate_reward(reward, info, done, prev_info=None):
    total_reward = reward * 0.1
    
    if info["flag_get"]:
        total_reward += 1000
    
    if prev_info and info["coins"] > prev_info["coins"]:
        total_reward += 50
    
    if not done:
        total_reward += 1.0
    
    if prev_info and info["x_pos"] > prev_info["x_pos"]:
        total_reward += 10
    
    return np.clip(total_reward, -1000, 1000)
```

### 에이전트 네트워크
```python
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.online = self.__build_cnn(input_dim, output_dim)
        self.target = self.__build_cnn(input_dim, output_dim)
        self.target.load_state_dict(self.online.state_dict())
```

## 참고 자료
- [혁펜하임 강화학습 이론](https://youtu.be/cvctS4xWSaU?si=PpxgYrUgy-XQykK-)
- [PPO 논문](https://arxiv.org/abs/1707.06347)
- [Gym 문서](https://gym.openai.com/)
- [PyTorch 문서](https://pytorch.org/docs/stable/index.html)
- [NES-py 문서](https://github.com/Kautenja/gym-super-mario-bros)
