# 실시간 낙상 감지 시스템
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time
import threading
from collections import deque
import joblib

class RealTimeFallDetector:
    def __init__(self, model_path, scaler_path=None, window_size=50, threshold=0.5):
        """
        실시간 낙상 감지 시스템 초기화
        
        Args:
            model_path: 학습된 모델 파일 경로 (.h5)
            scaler_path: 스케일러 파일 경로 (.pkl)
            window_size: 윈도우 크기 (학습 시와 동일해야 함)
            threshold: 낙상 판단 임계값 (0.5 기본)
        """
        print("🚀 실시간 낙상 감지 시스템 초기화 중...")
        
        # 모델 로드
        try:
            self.model = load_model(model_path)
            print(f"✅ 모델 로드 성공: {model_path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return
        
        # 스케일러 로드 (있는 경우)
        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✅ 스케일러 로드 성공: {scaler_path}")
            except:
                print("⚠️  스케일러 로드 실패, 기본 정규화 사용")
                self.scaler = StandardScaler()
        else:
            self.scaler = StandardScaler()
        
        self.window_size = window_size
        self.threshold = threshold
        self.data_buffer = deque(maxlen=window_size)
        self.is_running = False
        
        print(f"📊 설정:")
        print(f"   - 윈도우 크기: {window_size}")
        print(f"   - 임계값: {threshold}")
        print(f"   - 모델 입력 형태: {self.model.input_shape}")
    
    def preprocess_data(self, csi_data):
        """
        실시간 데이터 전처리
        
        Args:
            csi_data: CSI 특징 데이터 (1D array)
        
        Returns:
            preprocessed_data: 전처리된 데이터
        """
        # 데이터 형태 변환
        if len(csi_data.shape) == 1:
            csi_data = csi_data.reshape(1, -1)
        
        # 정규화 (스케일러가 fit되어 있다면)
        try:
            normalized_data = self.scaler.transform(csi_data)
        except:
            # 스케일러가 fit되지 않은 경우 간단한 정규화
            normalized_data = (csi_data - np.mean(csi_data)) / (np.std(csi_data) + 1e-7)
        
        return normalized_data.flatten()
    
    def add_data_point(self, csi_features):
        """
        새로운 CSI 데이터 포인트 추가
        
        Args:
            csi_features: CSI 특징 벡터 (feat_0 ~ feat_N)
        
        Returns:
            prediction_result: 예측 결과 딕셔너리
        """
        # 전처리
        processed_data = self.preprocess_data(csi_features)
        
        # 버퍼에 추가
        self.data_buffer.append(processed_data)
        
        # 충분한 데이터가 쌓이면 예측
        if len(self.data_buffer) == self.window_size:
            return self.predict()
        
        return None
    
    def predict(self):
        """
        현재 윈도우에 대해 낙상 예측 수행
        
        Returns:
            result: 예측 결과 딕셔너리
        """
        if len(self.data_buffer) < self.window_size:
            return None
        
        # 윈도우 데이터 준비
        window_data = np.array(list(self.data_buffer))
        
        # 모델 입력 형태로 변환 (1, window_size, features)
        input_data = window_data.reshape(1, self.window_size, -1)
        
        # 예측 수행
        try:
            prediction_prob = self.model.predict(input_data, verbose=0)[0][0]
            is_fall = prediction_prob > self.threshold
            
            result = {
                'timestamp': time.time(),
                'probability': float(prediction_prob),
                'is_fall': bool(is_fall),
                'confidence': 'HIGH' if prediction_prob > 0.8 or prediction_prob < 0.2 else 'MEDIUM',
                'status': '🚨 낙상 감지!' if is_fall else '✅ 정상'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return None
    
    def simulate_real_time(self, csv_file_path, delay=0.1):
        """
        CSV 파일을 사용하여 실시간 감지 시뮬레이션
        
        Args:
            csv_file_path: 테스트할 CSV 파일 경로
            delay: 데이터 포인트 간 지연 시간 (초)
        """
        print(f"📁 시뮬레이션 시작: {csv_file_path}")
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file_path)
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            
            print(f"📊 데이터 정보:")
            print(f"   - 총 데이터 포인트: {len(df)}")
            print(f"   - 특징 수: {len(feature_cols)}")
            
            fall_detected_count = 0
            
            # 실시간 시뮬레이션
            for i, row in df.iterrows():
                # CSI 특징 추출
                csi_features = row[feature_cols].values
                
                # 예측 수행
                result = self.add_data_point(csi_features)
                
                if result:
                    # 결과 출력
                    print(f"[{i:4d}] {result['status']} (확률: {result['probability']:.3f}, 신뢰도: {result['confidence']})")
                    
                    if result['is_fall']:
                        fall_detected_count += 1
                        print(f"      🚨 알림: {time.strftime('%H:%M:%S')}에 낙상이 감지되었습니다!")
                
                # 지연 시간
                time.sleep(delay)
            
            print(f"\n📊 시뮬레이션 완료:")
            print(f"   - 처리된 데이터 포인트: {len(df)}")
            print(f"   - 낙상 감지 횟수: {fall_detected_count}")
            
        except Exception as e:
            print(f"❌ 시뮬레이션 오류: {e}")
    
    def start_monitoring(self):
        """백그라운드에서 실시간 모니터링 시작"""
        self.is_running = True
        print("🔄 실시간 모니터링 시작...")
        
        # 실제 구현에서는 여기에 CSI 데이터 수집 코드 추가
        # 예: WiFi 어댑터에서 실시간 CSI 데이터 읽기
        
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_running = False
        print("⏹️  모니터링 중지")

# 사용 예시
def main():
    """메인 실행 함수"""
    print("🏠 스마트 홈 낙상 감지 시스템")
    print("=" * 50)
    
    # 모델 파일 경로 (학습 완료 후 생성된 파일)
    model_path = "./models/csi_fall_detection_128features.h5"
    
    try:
        # 감지 시스템 초기화
        detector = RealTimeFallDetector(
            model_path=model_path,
            window_size=50,  # 학습 시와 동일하게 설정
            threshold=0.5    # 필요에 따라 조정
        )
        
        # 실시간 시뮬레이션 (테스트용)
        print("\n🧪 테스트 시뮬레이션:")
        test_csv = "./csi_data/case1/5_labeled.csv"  # 테스트 파일 경로
        detector.simulate_real_time(test_csv, delay=0.05)
        
        # 실제 환경에서는 이렇게 사용:
        # detector.start_monitoring()
        
    except FileNotFoundError:
        print("❌ 모델 파일을 찾을 수 없습니다!")
        print("   먼저 run_training.py를 실행하여 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()