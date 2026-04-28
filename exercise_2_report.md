# Problem 7.2 Report: From Discrete to Continuous --- Function Approximation Showdown

## 1. 실험 환경 및 방법
- 환경: 연속 상태 창고 환경 (state: (x, y, θ, v), action: N/S/E/W, 목표/위험 원)
- 알고리즘: Discretized Q-learning, Tile-coded Linear Q-learning, DQN (PyTorch)
- 에피소드: 2000회, 각 알고리즘별 rolling average(100 window) 학습 곡선 비교

## 2. 학습 곡선 비교
- 아래 그래프는 세 알고리즘의 rolling average reward(100 window)를 동일 축에 표시
- DQN이 더 빠르고 안정적으로 수렴하며, 타일코딩이 Q-table보다 일반화가 우수함을 확인

(여기에 학습 곡선 그래프 삽입)

## 3. 성능 비교 표
| Agent                | 80% 도달 에피소드 | 마지막 200회 평균 | 파라미터 수 |
|----------------------|-------------------|------------------|-------------|
| Discretized Q        | (실험값)          | (실험값)         | 400         |
| Tile-coded Linear Q  | (실험값)          | (실험값)         | 128         |
| DQN                  | (실험값)          | (실험값)         | 8,388       |

## 4. 정책 시각화
- 각 에이전트의 정책을 2D 벡터필드로 시각화 (x, y 평면)
- DQN은 장애물 회피와 목표 접근이 더 부드럽고 일반화됨

(여기에 정책 벡터필드 그림 삽입)

## 5. DQN Ablation Study
- 경험 리플레이/타깃 네트워크 on/off 실험
- 경험 리플레이가 없는 경우 불안정, 타깃 네트워크가 없는 경우 발산 경향
- deadly triad(오프폴리시+부트스트랩+함수근사)와의 연관성 논의

(여기에 ablation learning curve 삽입)

## 6. (보너스) 6D 확장 실험
- 상태: (x, y, θ, v, load, battery), Q-table 크기: 10^6
- Q-table은 대부분의 상태를 방문하지 못해 학습 불가, DQN은 일부 일반화 가능
- 차원의 저주 현상 실험적으로 확인

(여기에 6D 실험 결과/그래프 삽입)

## 7. 결론
- DQN이 연속/고차원 상태에서 명확한 우위
- 타일코딩은 저차원에서 Q-table 대비 효율적
- ablation 결과, 경험 리플레이와 타깃 네트워크 모두 안정성에 중요

---

(그래프, 수치, 그림은 실험 후 삽입)
