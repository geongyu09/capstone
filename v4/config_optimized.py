"""
실제 문제에 맞춘 Recall 개선 설정
디버깅 결과 데이터는 충분하므로 모델/학습 설정에 집중
"""

import os

class Config:
    """Recall 개선에 특화된 설정 (데이터는 충분함)"""
    
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
    
    # 모델 학습 설정 (데이터가 충분하므로 안정적 학습에 집중)
    WINDOW_SIZE = 50
    STRIDE = 10
    BATCH_SIZE = 32          # 원래대로 복원 (데이터 충분)
    EPOCHS = 80             # 충분한 에포크
    LEARNING_RATE = 0.0003   # 낮은 학습률로 안정적 학습
    
    # 데이터 분할
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 낙상 감지 임계값 (데이터 충분하므로 기존 설정 유지)
    FALL_THRESHOLD = 0.4      # 적당히 민감하게
    OVERLAP_THRESHOLD = 0.3   # 원래 설정 유지 (데이터 충분)
    
    # 로깅 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ⭐ 핵심: 낙상에 집중하는 가중치 (과도하지 않게)
    CLASS_WEIGHTS = {
        0: 1.0,   # 정상
        1: 5.0    # 낙상 (5배 가중치 - 적당히)
    }
    
    # ⭐ Focal Loss 설정 (적당히)
    FOCAL_LOSS_ALPHA = 0.6    # 낙상에 약간 더 집중
    FOCAL_LOSS_GAMMA = 2.5    # 어려운 샘플에 집중
    
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
            # os.path.join(cls.DATA_DIR, "낙상 2/case1/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 2/case3/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 3/case1/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 3/case1/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 3/case2/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 3/case2/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 3/case3/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 4/case1/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 4/case2/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 4/case2/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 5/case1/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 5/case3/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 5/case3/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case1/original"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case1/reversed"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case1/warped"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case2/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case2/original"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case2/reversed"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case3/jitter"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case3/original"),
            # os.path.join(cls.DATA_DIR, "낙상 6/case3/reversed"),
            os.path.join(cls.DATA_DIR, "case1/"),
            os.path.join(cls.DATA_DIR, "case2/"),
            os.path.join(cls.DATA_DIR, "case3/"),
        ]
    
    @classmethod
    def print_config(cls):
        """설정 정보 출력"""
        print("⚙️ CSI 낙상 감지 v4 설정 (Recall 최적화)")
        print("=" * 50)
        print("📊 디버깅 결과:")
        print("   ✅ 낙상 시퀀스: 64.38% (충분함!)")
        print("   ✅ 데이터 분포: 양호")
        print("   🎯 문제: 모델/학습 설정")
        print()
        print(f"🔧 수정된 핵심 설정:")
        print(f"   - 학습률: {cls.LEARNING_RATE} (낮춤)")
        print(f"   - 클래스 가중치: {cls.CLASS_WEIGHTS} (적당히)")
        print(f"   - Focal Loss: α={cls.FOCAL_LOSS_ALPHA}, γ={cls.FOCAL_LOSS_GAMMA}")
        print(f"   - 에포크: {cls.EPOCHS} (충분히)")
        print()
        print(f"📈 기대 효과:")
        print("   - 안정적인 학습으로 Recall 개선")
        print("   - 과적합 방지")
        print("   - 점진적 성능 향상")


class ModelConfig:
    """모델 아키텍처 설정 (Recall 개선)"""
    
    # CNN 설정 (약간 완화)
    CNN_FILTERS = [64, 128, 256]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.1          # 0.3 → 0.1 (드롭아웃 줄임)
    
    # LSTM 설정 (약간 완화)
    LSTM_UNITS = [128, 64]
    LSTM_DROPOUT = 0.1         # 0.3 → 0.1
    LSTM_RECURRENT_DROPOUT = 0.1  # 0.3 → 0.1
    
    # Dense 설정 (약간 완화)
    DENSE_UNITS = [64, 32]
    DENSE_DROPOUT = 0.2        # 0.5 → 0.2 (과적합 방지 완화)
    
    # 출력 설정
    OUTPUT_ACTIVATION = 'sigmoid'


if __name__ == "__main__":
    Config.print_config()
    Config.ensure_directories()
    print("\n✅ 실제 문제에 맞춘 Recall 최적화 설정이 적용되었습니다.")
    print("\n📝 이제 다음을 실행하세요:")
    print("python train_recall_optimized.py")
