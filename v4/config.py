"""
CSI 낙상 감지 v4 설정 파일
"""

import os

class Config:
    """기본 설정 클래스"""
    
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
    SCALER_TYPE = 'minmax'  # 'minmax', 'standard', 'robust'
    
    # 모델 학습 설정
    WINDOW_SIZE = 50        # 시퀀스 길이
    STRIDE = 10             # 슬라이딩 윈도우 스트라이드
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # 데이터 분할
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 낙상 감지 임계값
    FALL_THRESHOLD = 0.5
    OVERLAP_THRESHOLD = 0.3  # 시퀀스에서 낙상 라벨 비율
    
    # 로깅 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
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
            os.path.join(cls.DATA_DIR, "case1"),
            os.path.join(cls.DATA_DIR, "case2"),
            os.path.join(cls.DATA_DIR, "case3"),
            "../labeled"
        ]
    
    @classmethod
    def print_config(cls):
        """설정 정보 출력"""
        print("⚙️ CSI 낙상 감지 v4 설정")
        print("=" * 40)
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
        print(f"   - 배치 크기: {cls.BATCH_SIZE}")
        print(f"   - 에포크: {cls.EPOCHS}")
        print(f"   - 학습률: {cls.LEARNING_RATE}")
        print()
        print(f"📈 데이터 분할:")
        print(f"   - 훈련: {cls.TRAIN_RATIO*100}%")
        print(f"   - 검증: {cls.VAL_RATIO*100}%")
        print(f"   - 테스트: {cls.TEST_RATIO*100}%")


class ModelConfig:
    """모델 아키텍처 설정"""
    
    # CNN 설정
    CNN_FILTERS = [64, 128, 256]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.3
    
    # LSTM 설정
    LSTM_UNITS = [128, 64]
    LSTM_DROPOUT = 0.3
    LSTM_RECURRENT_DROPOUT = 0.3
    
    # Dense 설정
    DENSE_UNITS = [64, 32]
    DENSE_DROPOUT = 0.5
    
    # 출력 설정
    OUTPUT_ACTIVATION = 'sigmoid'
    
    @classmethod
    def print_model_config(cls):
        """모델 설정 출력"""
        print("🧠 모델 아키텍처 설정")
        print("=" * 30)
        print(f"🔍 CNN 레이어:")
        print(f"   - 필터: {cls.CNN_FILTERS}")
        print(f"   - 커널 크기: {cls.CNN_KERNEL_SIZE}")
        print(f"   - 드롭아웃: {cls.CNN_DROPOUT}")
        print()
        print(f"🔄 LSTM 레이어:")
        print(f"   - 유닛: {cls.LSTM_UNITS}")
        print(f"   - 드롭아웃: {cls.LSTM_DROPOUT}")
        print(f"   - 순환 드롭아웃: {cls.LSTM_RECURRENT_DROPOUT}")
        print()
        print(f"🔗 Dense 레이어:")
        print(f"   - 유닛: {cls.DENSE_UNITS}")
        print(f"   - 드롭아웃: {cls.DENSE_DROPOUT}")
        print(f"   - 출력 활성화: {cls.OUTPUT_ACTIVATION}")


if __name__ == "__main__":
    # 설정 정보 출력
    Config.print_config()
    print()
    ModelConfig.print_model_config()
    
    # 디렉토리 생성
    Config.ensure_directories()
    print("\n✅ 필요한 디렉토리가 생성되었습니다.")
