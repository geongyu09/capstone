
# quick_fix_config.py
# Recall 문제 해결을 위한 수정된 설정

# 시퀀스 라벨링 완화
OVERLAP_THRESHOLD = 0.1  # 0.3 → 0.1로 낮춤

# 강화된 클래스 가중치
CLASS_WEIGHTS = {
    0: 1.0,   # 정상
    1: 15.0   # 낙상 (15배 가중치)
}

# Focal Loss 파라미터
FOCAL_LOSS_ALPHA = 0.75  # 낙상 클래스에 더 집중
FOCAL_LOSS_GAMMA = 3.0   # 어려운 샘플에 더 집중

# 학습 설정
LEARNING_RATE = 0.0005   # 낮은 학습률
PATIENCE = 20           # 더 긴 patience
EPOCHS = 60            # 더 많은 에포크

# 배치 크기 (필요시 줄이기)
BATCH_SIZE = 16        # 32 → 16
