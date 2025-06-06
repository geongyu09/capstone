# config.py
"""
CSI 낙상 감지 시스템 전역 설정
실제 데이터 분석 결과를 반영한 최적화된 설정
"""

import os
import platform
import matplotlib.pyplot as plt

class CSIConfig:
    """CSI 시스템 전역 설정 클래스"""
    
    # ========== 데이터 특성 설정 (실제 분석 결과 기반) ==========
    SAMPLING_RATE = 288  # Hz - 실제 측정된 샘플링 주파수
    ACTIVE_FEATURE_RANGE = (6, 250)  # feat_6 ~ feat_250만 활성
    TOTAL_FEATURES = 256
    ACTIVE_FEATURE_COUNT = 242  # 실제 활성 특성 수
    
    # ========== 윈도우 설정 (고주파 데이터 최적화) ==========
    WINDOW_SIZE = 72  # 0.5초 @ 288Hz - 낙상 감지에 최적
    STRIDE = 14        # 50ms 간격 - 실시간 처리 가능
    OVERLAP_THRESHOLD = 0.2  # 더 민감한 낙상 감지
    
    # ========== 학습 설정 ==========
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.0005  # 고주파 데이터용 낮은 학습률
    PATIENCE = 15
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # ========== 파일 경로 설정 ==========
    DEFAULT_DATA_DIR = '../csi_data'
    MODEL_SAVE_DIR = './models'
    LOG_DIR = './logs'
    RESULTS_DIR = './results'
    
    # ========== 모델 설정 ==========
    MODEL_TYPE = 'CNN_LSTM_HYBRID'
    CNN_FILTERS = [64, 32]
    CNN_KERNEL_SIZES = [5, 3]
    LSTM_UNITS = [64, 32]
    DENSE_UNITS = [16]
    DROPOUT_RATES = [0.25, 0.4, 0.3]
    
    # ========== 분석 설정 ==========
    CONFIDENCE_THRESHOLD = 0.5
    FALL_DURATION_THRESHOLD = 3  # 최소 연속 감지 횟수
    VISUALIZATION_DPI = 300
    
    # ========== 실시간 처리 설정 ==========
    REALTIME_BUFFER_SIZE = 1000
    REALTIME_UPDATE_INTERVAL = 0.1  # 초
    
    @classmethod
    def setup_matplotlib(cls):
        """Matplotlib 한글 폰트 설정"""
        if platform.system() == "Windows":
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
        elif platform.system() == "Darwin":  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.rcParams['axes.unicode_minus'] = False
        else:  # Linux
            plt.rcParams['font.family'] = 'DejaVu Sans'
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        directories = [
            cls.MODEL_SAVE_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_window_settings(cls, sampling_rate=None):
        """샘플링 주파수에 따른 윈도우 설정 자동 계산"""
        if sampling_rate is None:
            sampling_rate = cls.SAMPLING_RATE
        
        # 낙상 감지에 최적화된 시간 윈도우
        detection_time = 0.5  # 0.5초
        stride_time = 0.05    # 50ms
        
        window_size = int(detection_time * sampling_rate)
        stride = int(stride_time * sampling_rate)
        
        return window_size, stride
    
    @classmethod
    def get_model_config(cls):
        """모델 설정 딕셔너리 반환"""
        return {
            'model_type': cls.MODEL_TYPE,
            'window_size': cls.WINDOW_SIZE,
            'feature_count': cls.ACTIVE_FEATURE_COUNT,
            'cnn_filters': cls.CNN_FILTERS,
            'cnn_kernel_sizes': cls.CNN_KERNEL_SIZES,
            'lstm_units': cls.LSTM_UNITS,
            'dense_units': cls.DENSE_UNITS,
            'dropout_rates': cls.DROPOUT_RATES,
            'learning_rate': cls.LEARNING_RATE
        }
    
    @classmethod
    def get_data_config(cls):
        """데이터 설정 딕셔너리 반환"""
        return {
            'sampling_rate': cls.SAMPLING_RATE,
            'active_range': cls.ACTIVE_FEATURE_RANGE,
            'window_size': cls.WINDOW_SIZE,
            'stride': cls.STRIDE,
            'overlap_threshold': cls.OVERLAP_THRESHOLD,
            'batch_size': cls.BATCH_SIZE
        }
    
    @classmethod
    def print_config(cls):
        """설정 정보 출력"""
        print("⚙️ CSI 시스템 설정 정보")
        print("=" * 50)
        
        print(f"📊 데이터 설정:")
        print(f"   샘플링 주파수: {cls.SAMPLING_RATE}Hz")
        print(f"   활성 특성: feat_{cls.ACTIVE_FEATURE_RANGE[0]} ~ feat_{cls.ACTIVE_FEATURE_RANGE[1]} ({cls.ACTIVE_FEATURE_COUNT}개)")
        print(f"   윈도우 크기: {cls.WINDOW_SIZE}개 ({cls.WINDOW_SIZE/cls.SAMPLING_RATE:.1f}초)")
        print(f"   스트라이드: {cls.STRIDE}개 ({cls.STRIDE/cls.SAMPLING_RATE*1000:.0f}ms)")
        print(f"   겹침 임계값: {cls.OVERLAP_THRESHOLD}")
        
        print(f"\n🏗️ 모델 설정:")
        print(f"   모델 타입: {cls.MODEL_TYPE}")
        print(f"   CNN 필터: {cls.CNN_FILTERS}")
        print(f"   LSTM 유닛: {cls.LSTM_UNITS}")
        print(f"   학습률: {cls.LEARNING_RATE}")
        print(f"   배치 크기: {cls.BATCH_SIZE}")
        
        print(f"\n📁 경로 설정:")
        print(f"   데이터 디렉토리: {cls.DEFAULT_DATA_DIR}")
        print(f"   모델 저장: {cls.MODEL_SAVE_DIR}")
        print(f"   로그 저장: {cls.LOG_DIR}")
        print(f"   결과 저장: {cls.RESULTS_DIR}")
        
        print(f"\n🔍 분석 설정:")
        print(f"   신뢰도 임계값: {cls.CONFIDENCE_THRESHOLD}")
        print(f"   낙상 지속 임계값: {cls.FALL_DURATION_THRESHOLD}")

# 설정 초기화
CSIConfig.setup_matplotlib()
CSIConfig.create_directories()

if __name__ == "__main__":
    # 설정 정보 출력
    CSIConfig.print_config()