# CSI 기반 낙상 감지 시스템 🚨

Channel State Information (CSI) 데이터를 활용한 LSTM 기반 실시간 낙상 감지 시스템입니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [설치 방법](#설치-방법)
- [데이터 준비](#데이터-준비)
- [사용 방법](#사용-방법)
- [모델 아키텍처](#모델-아키텍처)
- [성능 평가](#성능-평가)
- [문제 해결](#문제-해결)
- [기여 방법](#기여-방법)

## 프로젝트 개요

### 📖 배경

WiFi CSI (Channel State Information)는 무선 신호의 채널 상태 정보를 담고 있으며, 환경 내 움직임과 활동을 감지할 수 있습니다. 이 프로젝트는 CSI 데이터를 분석하여 낙상 사고를 실시간으로 감지하는 시스템을 구현합니다.

### 🎯 주요 기능

- **실시간 낙상 감지**: LSTM을 활용한 시계열 패턴 분석
- **다양한 특징 지원**: 64개~256개 특징 자동 처리
- **유연한 모델**: 3가지 모델 아키텍처 제공
- **자동 최적화**: 데이터에 따른 자동 하이퍼파라미터 조정
- **상세한 평가**: 성능 지표 및 특징 중요도 분석

## 기술 스택

### 🔧 핵심 라이브러리

- **TensorFlow/Keras**: 딥러닝 모델 구현
- **scikit-learn**: 전처리 및 평가
- **pandas**: 데이터 처리
- **numpy**: 수치 연산
- **matplotlib**: 시각화

### 🧠 모델 아키텍처

- **LSTM**: 시계열 패턴 학습
- **CNN + LSTM**: 지역적 + 시간적 패턴 결합
- **Dense Layer**: 특징 차원 축소

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd csi-fall-detection
```

### 2. 의존성 설치

```bash
pip install tensorflow>=2.8.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
```

또는 requirements.txt 사용:

```bash
pip install -r requirements.txt
```

### 3. 폴더 구조 생성

```bash
mkdir csi_data
mkdir models
```

## 데이터 준비

### 📊 데이터 형식

CSI 데이터는 다음 형식의 CSV 파일이어야 합니다:

```csv
timestamp,label,feat_0,feat_1,feat_2,...,feat_255
2025-05-29 15:47:27,0,0.1234,0.5678,...,0.9012
2025-05-29 15:47:28,1,0.2345,0.6789,...,0.0123
```

### 📁 필수 컬럼

- **timestamp**: 데이터 수집 시간 (YYYY-MM-DD HH:MM:SS)
- **label**: 낙상 여부 (0: 정상, 1: 낙상)
- **feat_0 ~ feat_N**: CSI 특징 데이터 (64개 또는 256개)

### 📂 데이터 배치

모든 CSV 파일을 `csi_data/` 폴더에 배치하세요:

```
프로젝트_폴더/
├── csi_data/
│   ├── fall_event_1.csv      # 낙상 데이터
│   ├── fall_event_2.csv
│   ├── normal_walk_1.csv     # 정상 활동 데이터
│   ├── normal_sit_1.csv
│   └── ...
```

## 사용 방법

### 🚀 기본 실행

```bash
python run_training.py
```

### 📈 실행 과정

1. **데이터 로드**: CSV 파일들을 자동으로 검색하고 로드
2. **전처리**: 특징 선택, 정규화, 슬라이딩 윈도우 적용
3. **모델 구축**: 특징 수에 따른 최적 모델 자동 선택
4. **학습**: Early Stopping과 Learning Rate 조정으로 학습
5. **평가**: 성능 지표 출력 및 혼동 행렬 표시
6. **저장**: 학습된 모델을 `models/` 폴더에 저장

### ⚙️ 주요 파라미터 조정

`run_training.py`에서 다음 파라미터를 수정할 수 있습니다:

```python
# CSIFallDetection 초기화
detector = CSIFallDetection(
    window_size=50,    # 시계열 윈도우 크기
    stride=1          # 슬라이딩 윈도우 간격
)

# 모델 타입 선택
model_type = 'lightweight'  # 'standard', 'lightweight', 'cnn_lstm'
```

## 모델 아키텍처

### 🏗️ 1. Standard Model

```
LSTM(128) → Dropout(0.4) → LSTM(64) → Dense(32) → Dense(16) → Dense(1)
```

- **용도**: 일반적인 CSI 데이터 (64-128 특징)
- **특징**: 높은 정확도, 충분한 표현력

### 🪶 2. Lightweight Model

```
Dense(64) → LSTM(32) → LSTM(16) → Dense(8) → Dense(1)
```

- **용도**: 고차원 데이터 (256 특징) 또는 빠른 추론
- **특징**: 빠른 학습, 적은 메모리 사용

### 🔀 3. CNN-LSTM Hybrid

```
Conv1D(64) → Conv1D(32) → MaxPool1D → LSTM(32) → Dense(16) → Dense(1)
```

- **용도**: 지역적 + 시간적 패턴 모두 중요한 경우
- **특징**: 복합적 패턴 학습

## 성능 평가

### 📊 평가 지표

- **Accuracy**: 전체 정확도
- **Precision**: 양성 예측값 중 실제 양성 비율
- **Recall**: 실제 양성 중 올바르게 예측한 비율
- **F1-Score**: Precision과 Recall의 조화 평균

### 🎯 낙상 감지 중요 지표

- **Recall**: 실제 낙상을 놓치지 않는 것이 중요
- **False Positive Rate**: 거짓 알람을 최소화

### 📈 성능 최적화 팁

1. **데이터 균형**: 낙상/정상 데이터 비율 1:1 ~ 1:3 유지
2. **Window Size**: 낙상 지속 시간에 맞춰 조정 (보통 30-100)
3. **Feature Selection**: 256개 특징 시 자동 선택 활성화
4. **Ensemble**: 여러 모델 조합으로 성능 향상

## 문제 해결

### ❗ 자주 발생하는 오류

#### 1. 폴더/파일 없음

```
❌ 오류: ./csi_data 폴더가 없습니다!
```

**해결**: `mkdir csi_data` 명령으로 폴더 생성

#### 2. CSV 형식 오류

```
❌ 경고: 필수 컬럼이 없습니다.
```

**해결**: CSV 파일에 timestamp, label, feat\_\* 컬럼 확인

#### 3. 메모리 부족

```
❌ ResourceExhaustedError: OOM when allocating tensor
```

**해결**:

- window_size 줄이기 (50 → 30)
- stride 늘리기 (1 → 2)
- batch_size 줄이기 (32 → 16)

#### 4. 라벨 불균형

```
⚠️ 경고: 낙상 데이터가 없습니다!
```

**해결**: 일부 CSV 파일의 label을 1로 설정

### 🔧 성능 개선 방법

#### 1. 데이터 증강

```python
# 노이즈 추가
noise = np.random.normal(0, 0.01, X.shape)
X_augmented = X + noise

# 시간 왜곡
from scipy.interpolate import interp1d
# 시간축 변형 코드
```

#### 2. 하이퍼파라미터 튜닝

```python
# 그리드 서치
window_sizes = [30, 50, 70, 100]
learning_rates = [0.001, 0.0005, 0.0001]
# 최적 조합 탐색
```

#### 3. 앙상블 모델

```python
# 여러 모델 조합
models = ['standard', 'lightweight', 'cnn_lstm']
predictions = []
for model_type in models:
    pred = train_and_predict(model_type)
    predictions.append(pred)
final_pred = np.mean(predictions, axis=0)
```

## 파일 구조

```
csi-fall-detection/
├── README.md                    # 프로젝트 설명서
├── requirements.txt             # 의존성 목록
├── csi_fall_detection.py        # 핵심 클래스 정의
├── run_training.py              # 실행 스크립트
├── csi_data/                    # 학습 데이터
│   ├── fall_event_1.csv
│   ├── normal_activity_1.csv
│   └── ...
├── models/                      # 학습된 모델
│   └── csi_fall_detection_*.h5
└── docs/                        # 추가 문서
    ├── data_format.md
    └── model_architecture.md
```

## 기여 방법

### 🤝 기여 가이드라인

1. **이슈 생성**: 버그 리포트나 기능 요청
2. **Fork & Clone**: 저장소를 포크하고 로컬에 클론
3. **브랜치 생성**: `git checkout -b feature/new-feature`
4. **코드 작성**: PEP 8 스타일 가이드 준수
5. **테스트**: 새로운 기능에 대한 테스트 작성
6. **Pull Request**: 상세한 설명과 함께 PR 생성

### 📝 개발 로드맵

- [ ] 실시간 CSI 데이터 수집 모듈
- [ ] 웹 기반 모니터링 대시보드
- [ ] 모바일 앱 알림 시스템
- [ ] 다중 사용자 환경 지원
- [ ] 클라우드 배포 지원

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

- **개발자**: [Your Name]
- **이메일**: [your.email@example.com]
- **프로젝트 URL**: [https://github.com/username/csi-fall-detection]

---

## 🙏 감사의 글

이 프로젝트는 CSI 기반 인간 활동 인식 연구에 기반하여 개발되었습니다. 오픈소스 커뮤니티와 관련 연구자들에게 감사드립니다.

**⚠️ 주의사항**: 이 시스템은 연구 및 교육 목적으로 개발되었습니다. 실제 의료용 또는 안전 시스템으로 사용하기 전에 충분한 검증이 필요합니다.
