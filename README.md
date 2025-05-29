# 강화학습 기반 슈퍼 마리오 게임

이 프로젝트는 강화학습을 사용하여 슈퍼 마리오 게임을 자동으로 플레이하는 AI를 구현합니다. 
강화학습에 사용되는 다양한 알고리즘의 학습 및 적용을 시도해 보려고 합니다.

아래와 같은 개념을 공부하고 적용하고 있습니다.
1. epsilon-greedy
2. Q-learning


## 프로젝트 구조

### 핵심 파일
- `single.py`: DQN 기반 마리오 AI 학습 (단일 프로세스)
- `parallel.py`: DQN 학습의 병렬 처리 가속화 버전
- `test_single.py`: 학습된 모델 시연 및 테스트
- `checkpoints/`: 학습된 모델 체크포인트 저장소
- `test_single/`: 시연을 위해서 모델을 저장하는 디렉토리

### 환경 설정
- `pyproject.toml`: 프로젝트 의존성 관리 (uv 패키지 매니저 사용)
- `.python-version`: Python 버전전
- `uv.lock`: 의존성 잠금 파일
- `.venv/`: 가상 환경



## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
cd RL-Mario
```

2. UV 패키지 매니저를 사용하여 의존성을 설치합니다:
```bash
uv sync
```

## 사용 방법

### 1. 모델 학습 (단일 프로세스)
```bash
uv run single.py
```


### 3. 학습된 모델 테스트
```bash
uv run test_single.py
```

## 주요 특징

### 1. DQN 아키텍처
- **네트워크 구조**: CNN 기반 (Conv2D + BatchNorm + ReLU)
- **입력**: 84x84 그레이스케일 프레임 4개 스택
- **출력**: 3개 액션 ([오른쪽], [오른쪽+점프], [점프])
- **더블 DQN**: 온라인/타겟 네트워크 분리



## 모니터링 및 시각화

### TensorBoard 로그
```bash
tensorboard --logdir checkpoints
```

### 메트릭
- **평균 보상**: 최근 100 에피소드 평균
- **에피소드 길이**: 생존 시간
- **Q-값**: 예상 가치 함수 값
- **손실**: 학습 손실
- **탐험률**: 현재 epsilon 값
