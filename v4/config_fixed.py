"""
Recall 문제 해결을 위한 수정된 config.py
"""

import os

class Config:
    """기본 설정 클래스 (Recall 문제 수정)"""
    
    # 데이터 관련 설정
    DATA_DIR = "../csi_data"
    PROCESSED_DATA_DIR = "./processed_data"
    MODEL_DIR = "./models"
    LOG_DIR = "./logs"
    
    # CSI 데이터 특성
    AMPLITUDE_START_COL = 8
    AMPLITUDE_END_COL = 253
    TOTAL_FEATURES = AMPLITUDE_END_COL - AMPLITUDE_START_COL
    
    # 전처리 설정
    MOVING_AVERAGE_WINDOW = 5
    OUTLIER_THRESHOLD = 3.0
    SCALER_TYPE = 'minmax'
    
    # 모델 학습 설정
    WINDOW_SIZE = 50
    STRIDE = 10
    BATCH_SIZE = 16          # 32 → 16 (메모리 절약)
    EPOCHS = 60             # 100 → 60 (충분한 학습)
    LEARNING_RATE = 0.0005   # 0.001 → 0.0005 (안정적 학습)
    
    # 데이터 분할
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ⭐ 핵심 수정: 낙상 감지 임계값들
    FALL_THRESHOLD = 0.3      # 0.5 → 0.3 (더 민감하게)
    OVERLAP_THRESHOLD = 0.1   # 0.3 → 0.1 (⭐ 가장 중요한 수정!)
    
    # 로깅 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ⭐ 새로 추가: 클래스 가중치
    CLASS_WEIGHTS = {
        0: 1.0,   # 정상
        1: 15.0   # 낙상 (15배 가중치)
    }
    
    # ⭐ 새로 추가: Focal Loss 설정
    FOCAL_LOSS_ALPHA = 0.75   # 낙상 클래스에 더 집중
    FOCAL_LOSS_GAMMA = 3.0    # 어려운 샘플에 더 집중
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리 생성"""
        dirs = [
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_data_paths(cls):
        """데이터 경로 리스트 반환"""
        return [
            os.path.join(cls.DATA_DIR, "낙상 2/case1/warped"),
            os.path.join(cls.DATA_DIR, "낙상 2/case3/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 3/case1/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 3/case1/warped"),
            os.path.join(cls.DATA_DIR, "낙상 3/case2/warped"),
            os.path.join(cls.DATA_DIR, "낙상 3/case2/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 3/case3/warped"),
            os.path.join(cls.DATA_DIR, "낙상 4/case1/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 4/case2/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 4/case2/warped"),
            os.path.join(cls.DATA_DIR, "낙상 5/case1/warped"),
            os.path.join(cls.DATA_DIR, "낙상 5/case3/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 5/case3/warped"),
            os.path.join(cls.DATA_DIR, "낙상 6/case1/original"),
            os.path.join(cls.DATA_DIR, "낙상 6/case1/reversed"),
            os.path.join(cls.DATA_DIR, "낙상 6/case1/warped"),
            os.path.join(cls.DATA_DIR, "낙상 6/case2/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 6/case2/original"),
            os.path.join(cls.DATA_DIR, "낙상 6/case2/reversed"),
            os.path.join(cls.DATA_DIR, "낙상 6/case3/jitter"),
            os.path.join(cls.DATA_DIR, "낙상 6/case3/original"),
            os.path.join(cls.DATA_DIR, "낙상 6/case3/reversed"),
        ]
    
    @classmethod
    def print_config(cls):
        """설정 정보 출력 (수정사항 강조)"""
        print("⚙️ CSI 낙상 감지 v4 설정 (Recall 개선)")
        print("=" * 50)
        print(f"📁 데이터 디렉토리: {cls.DATA_DIR}")
        print(f"📁 처리된 데이터: {cls.PROCESSED_DATA_DIR}")
        print(f"📁 모델 저장: {cls.MODEL_DIR}")
        print(f"📁 로그: {cls.LOG_DIR}")
        print()
        print(f"📊 데이터 특성:")
        print(f"   - Amplitude 컬럼: {cls.AMPLITUDE_START_COL}:{cls.AMPLITUDE_END_COL}")
        print(f"   - 총 특성 수: {cls.TOTAL_FEATURES}")
        print()
        print(f"🔧 전처리 설정:")
        print(f"   - 이동 평균 창: {cls.MOVING_AVERAGE_WINDOW}")
        print(f"   - 이상치 임계값: {cls.OUTLIER_THRESHOLD}")
        print(f"   - 정규화: {cls.SCALER_TYPE}")
        print()
        print(f"🤖 모델 설정:")
        print(f"   - 윈도우 크기: {cls.WINDOW_SIZE}")
        print(f"   - 스트라이드: {cls.STRIDE}")
        print(f"   - 배치 크기: {cls.BATCH_SIZE} ⭐")
        print(f"   - 에포크: {cls.EPOCHS}")
        print(f"   - 학습률: {cls.LEARNING_RATE} ⭐")
        print()
        print(f"🎯 낙상 감지 설정 (⭐ 수정됨):")
        print(f"   - 감지 임계값: {cls.FALL_THRESHOLD} ⭐")
        print(f"   - 시퀀스 라벨링 임계값: {cls.OVERLAP_THRESHOLD} ⭐")
        print(f"   - 클래스 가중치: {cls.CLASS_WEIGHTS} ⭐")
        print(f"   - Focal Loss Alpha: {cls.FOCAL_LOSS_ALPHA} ⭐")
        print(f"   - Focal Loss Gamma: {cls.FOCAL_LOSS_GAMMA} ⭐")


class ModelConfig:
    """모델 아키텍처 설정"""
    
    # CNN 설정
    CNN_FILTERS = [64, 128, 256]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.2          # 0.3 → 0.2 (약간 줄임)
    
    # LSTM 설정
    LSTM_UNITS = [128, 64]
    LSTM_DROPOUT = 0.2         # 0.3 → 0.2
    LSTM_RECURRENT_DROPOUT = 0.2  # 0.3 → 0.2
    
    # Dense 설정
    DENSE_UNITS = [64, 32]
    DENSE_DROPOUT = 0.3        # 0.5 → 0.3 (오버피팅 방지 완화)
    
    # 출력 설정
    OUTPUT_ACTIVATION = 'sigmoid'


if __name__ == "__main__":
    Config.print_config()
    Config.ensure_directories()
    print("\n✅ Recall 개선 설정이 적용되었습니다.")
