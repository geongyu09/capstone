# CSI 기반 낙상 감지 시스템 v2.0

> 288Hz 고주파 CSI 데이터에 최적화된 딥러닝 기반 실시간 낙상 감지 시스템

## 🎯 주요 특징

- **고성능 CNN+LSTM 하이브리드 모델**: 지역 패턴과 시계열 학습 결합
- **메모리 효율적 대용량 처리**: 100개+ 파일 동시 학습 가능
- **실시간 타임라인 분석**: 낙상 이벤트 자동 감지 및 시각화
- **실제 데이터 최적화**: 288Hz 샘플링, feat_6~250 활성 특성 반영
- **모듈화 구조**: 각 기능별 독립적 클래스 설계

## 📋 시스템 요구사항

- Python 3.8+
- TensorFlow 2.10+
- 메모리 8GB+ (대용량 학습 시)
- GPU 권장 (CPU도 지원)

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론 또는 파일 다운로드
git clone <repository-url>
cd csi-fall-detection

# 필요 패키지 설치
pip install -r requirements.txt
```

### 2. 빠른 데모

```bash
# 35.csv 파일이 있다면 즉시 데모 실행
python main.py

# 또는 직접 명령
python main.py test
```

### 3. 모델 훈련

```bash
# 전체 학습 (100개+ 파일)
python main.py train --data-dir ./csi_data --epochs 50

# 빠른 테스트 (단일 파일)
python main.py train --quick --csv-file 35.csv --epochs 10
```

### 4. 데이터 분석

```bash
# 낙상 분석 + 시각화
python main.py analyze 35.csv --confidence 0.5 --visualize

# 다른 모델로 분석
python main.py analyze 35.csv --model-path custom_model.keras
```

## 📂 파일 구조

```
csi-fall-detection/
├── config.py              # 전역 설정 관리
├── data_generator.py       # 메모리 효율적 데이터 로더
├── model_builder.py        # CNN+LSTM 모델 구축기
├── trainer.py             # 메인 훈련 클래스
├── analyzer.py            # 낙상 타임라인 분석기
├── main.py               # 통합 실행 스크립트
├── requirements.txt       # 필요 패키지
├── README.md             # 이 파일
└── 데이터 디렉토리/
    ├── models/           # 학습된 모델 저장
    ├── logs/            # 로그 파일
    ├── results/         # 분석 결과
    └── csi_data/        # CSV 데이터 파일들
```

## 🎓 사용법 상세

### 모델 훈련

#### 전체 학습
```bash
# 기본 CNN+LSTM 하이브리드 모델
python main.py train --data-dir ./csi_data --epochs 100

# 다른 모델 타입 사용
python main.py train --model-type attention --epochs 50

# 경량화 모델 (실시간 처리용)
python main.py train --model-type lightweight --epochs 30
```

#### 빠른 테스트 학습
```bash
# 단일 파일로 빠른 검증
python main.py train --quick --csv-file 35.csv --epochs 5

# 여러 설정 테스트
python trainer.py quick 35.csv 10
```

### 데이터 분석

#### 기본 분석
```bash
# 낙상 이벤트 감지
python main.py analyze 35.csv

# 신뢰도 조정
python main.py analyze 35.csv --confidence 0.3

# 시각화 포함
python main.py analyze 35.csv --visualize
```

#### 고급 분석
```bash
# 특정 모델 사용
python main.py analyze data.csv --model-path models/best_model.keras

# 결과 저장 없이 분석만
python main.py analyze data.csv --no-save-results
```

### 시스템 관리

#### 설정 확인
```bash
# 전체 설정 정보
python main.py config

# 시스템 상태 확인
python main.py info
```

#### 테스트
```bash
# 기본 시스템 테스트
python main.py test

# 훈련 포함 전체 테스트
python main.py test --include-training
```

## 🔧 고급 설정

### config.py 주요 설정값

```python
# 데이터 특성 (실제 측정 기반)
SAMPLING_RATE = 288  # Hz
ACTIVE_FEATURE_RANGE = (6, 250)  # feat_6 ~ feat_250
WINDOW_SIZE = 144  # 0.5초 @ 288Hz
STRIDE = 14  # 50ms 간격

# 학습 설정
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
CONFIDENCE_THRESHOLD = 0.5
```

### 모델 타입별 특징

| 모델 타입 | 특징 | 용도 |
|-----------|------|------|
| `basic_lstm` | 기본 LSTM | 간단한 테스트 |
| `cnn_lstm_hybrid` | CNN+LSTM 결합 | **권장 (일반용)** |
| `attention` | Attention 메커니즘 | 고정확도 연구용 |
| `multi_scale` | 다중 스케일 특성 | 복잡한 패턴 분석 |
| `lightweight` | 경량화 모델 | 실시간 처리용 |

## 📊 데이터 형식

### 입력 CSV 형식
```
timestamp,label,feat_0,feat_1,...,feat_255
2025-06-01 17:33:35.538,0,0,0,...,0
2025-06-01 17:33:35.541,0,226.718,885.931,...,0
...
```

### 필수 컬럼
- `timestamp`: 시간 정보 (다양한 형식 지원)
- `label`: 낙상 라벨 (0=정상, 1=낙상)
- `feat_0` ~ `feat_255`: CSI 특성값 (feat_6~250만 활성)

## 🎯 성능 최적화

### 메모리 사용량 줄이기
```python
# config.py에서 배치 크기 조정
BATCH_SIZE = 16  # 기본 32에서 16으로

# 윈도우 크기 줄이기 (정확도 trade-off)
WINDOW_SIZE = 72  # 기본 144에서 72로
```

### 훈련 속도 향상
```python
# 경량화 모델 사용
python main.py train --model-type lightweight

# 적은 에포크로 빠른 테스트
python main.py train --epochs 10
```

## 🔍 분석 결과 해석

### 낙상 이벤트 신뢰도
- 🔴 **높음** (80%+): 확실한 낙상
- 🟡 **중간** (60-80%): 의심되는 낙상
- 🟢 **낮음** (50-60%): 낮은 확률 감지

### 시각화 그래프
1. **확률 타임라인**: 시간별 낙상 확률 변화
2. **감지 상태**: 이진 낙상 감지 결과
3. **라벨 비교**: 실제 vs 예측 비교 (라벨 있는 경우)

## 🚨 문제 해결

### 일반적인 문제

#### 모델 로드 실패
```bash
# 모델 파일 확인
ls models/*.keras

# 권한 확인
chmod +r models/*.keras
```

#### 메모리 부족
```python
# config.py에서 배치 크기 줄이기
BATCH_SIZE = 8

# 또는 경량화 모델 사용
python main.py train --model-type lightweight
```

#### GPU 관련 문제
```python
# CPU 모드로 강제 실행
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### 로그 확인
```bash
# 최신 로그 확인
tail -f logs/csi_training_*.log

# 에러 로그 검색
grep "ERROR" logs/*.log
```

## 📈 성능 벤치마크

### 실제 데이터 테스트 결과 (35.csv 기준)
- **데이터**: 800 샘플, 288Hz, 2.8초
- **특성**: 242개 활성 특성 (feat_6~250)
- **정확도**: CNN+LSTM 하이브리드 모델 기준 85%+
- **처리 속도**: 실시간 처리 가능 (GPU 기준)

### 대용량 처리 성능
- **100개 파일**: 약 4,700 시퀀스 생성
- **메모리 사용량**: 약 0.6GB
- **훈련 시간**: GPU 기준 30-60분

## 🤝 기여 방법

1. 이슈 리포트: 버그나 개선사항 제안
2. 코드 기여: Pull Request 환영
3. 데이터 제공: 다양한 CSI 데이터셋 공유
4. 문서화: 사용법이나 팁 추가

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 📞 지원

- 📧 이메일: [이메일 주소]
- 💬 이슈: GitHub Issues
- 📖 문서: 이 README 파일

---

**🎉 CSI 기반 낙상 감지 시스템으로 안전한 환경을 만들어보세요!**