"""
CSI 낙상 감지 v4 - 실시간 감지 시스템
"""

import os
import time
import queue
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import Config
from utils import load_model_artifacts, setup_logging
from data_preprocessing import CSIPreprocessor


class RealTimeCSIDetector:
    """실시간 CSI 낙상 감지기"""
    
    def __init__(self, model_name: str, window_size: int = None, threshold: float = 0.5):
        """
        Args:
            model_name: 사용할 모델 이름
            window_size: 슬라이딩 윈도우 크기
            threshold: 낙상 감지 임계값
        """
        self.model_name = model_name
        self.window_size = window_size or Config.WINDOW_SIZE
        self.threshold = threshold
        
        # 로깅 설정
        self.logger = setup_logging()
        
        # 모델 및 전처리기
        self.model = None
        self.scaler = None
        self.preprocessor = None
        
        # 데이터 버퍼 (슬라이딩 윈도우)
        self.data_buffer = deque(maxlen=self.window_size)
        self.prediction_buffer = deque(maxlen=100)  # 최근 100개 예측 저장
        
        # 실시간 데이터 큐
        self.data_queue = queue.Queue()
        
        # 상태 변수
        self.is_running = False
        self.fall_detected = False
        self.last_prediction = 0.0
        self.detection_count = 0
        
        # 콜백 함수들
        self.on_fall_detected: Optional[Callable] = None
        self.on_prediction_updated: Optional[Callable] = None
        
        self.logger.info("🚀 실시간 CSI 낙상 감지기 초기화 완료")
    
    def load_model(self) -> None:
        """학습된 모델 로드"""
        self.logger.info(f"📂 모델 로딩: {self.model_name}")
        
        try:
            # 모델 아티팩트 로드
            self.model, self.scaler, metadata = load_model_artifacts(
                Config.MODEL_DIR, self.model_name
            )
            
            if self.model is None:
                raise ValueError("모델 로딩 실패")
            
            # 전처리기 초기화
            self.preprocessor = CSIPreprocessor(
                amplitude_start_col=Config.AMPLITUDE_START_COL,
                amplitude_end_col=Config.AMPLITUDE_END_COL,
                scaler_type=Config.SCALER_TYPE,
                logger=self.logger
            )
            
            # 기존 스케일러 사용
            if self.scaler:
                self.preprocessor.scaler = self.scaler
            
            self.logger.info("✅ 모델 로딩 완료")
            self.logger.info(f"   입력 형태: {metadata.get('input_shape', 'Unknown')}")
            self.logger.info(f"   임계값: {self.threshold}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def add_data_point(self, csi_data: np.ndarray) -> None:
        """새로운 CSI 데이터 포인트 추가"""
        if len(csi_data) != (Config.AMPLITUDE_END_COL - Config.AMPLITUDE_START_COL):
            raise ValueError(f"CSI 데이터 크기가 맞지 않습니다: {len(csi_data)}")
        
        # 데이터 큐에 추가
        self.data_queue.put(csi_data)
    
    def process_data_stream(self) -> None:
        """데이터 스트림 처리 (별도 스레드에서 실행)"""
        while self.is_running:
            try:
                # 큐에서 데이터 가져오기 (타임아웃 0.1초)
                csi_data = self.data_queue.get(timeout=0.1)
                
                # 버퍼에 추가
                self.data_buffer.append(csi_data)
                
                # 윈도우가 가득 차면 예측 수행
                if len(self.data_buffer) == self.window_size:
                    self._predict_current_window()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"데이터 처리 오류: {e}")
    
    def _predict_current_window(self) -> None:
        """현재 윈도우에 대해 예측 수행"""
        try:
            # 버퍼 데이터를 배열로 변환
            window_data = np.array(list(self.data_buffer))
            
            # 데이터 전처리
            if self.preprocessor.scaler:
                # 스케일링
                scaled_data = self.preprocessor.scaler.transform(window_data)
                
                # 이동평균 필터링
                if hasattr(self.preprocessor, 'moving_avg_window'):
                    scaled_data = self.preprocessor._apply_moving_average(
                        scaled_data, self.preprocessor.moving_avg_window
                    )
            else:
                scaled_data = window_data
            
            # 모델 입력 형태로 변환 [1, window_size, features]
            model_input = scaled_data.reshape(1, self.window_size, -1)
            
            # 예측 수행
            prediction = self.model.predict(model_input, verbose=0)[0][0]
            self.last_prediction = float(prediction)
            
            # 예측 결과 저장
            self.prediction_buffer.append({
                'timestamp': datetime.now(),
                'prediction': self.last_prediction,
                'is_fall': self.last_prediction > self.threshold
            })
            
            # 낙상 감지 확인
            if self.last_prediction > self.threshold:
                self._handle_fall_detection()
            
            # 콜백 호출
            if self.on_prediction_updated:
                self.on_prediction_updated(self.last_prediction, self.threshold)
                
        except Exception as e:
            self.logger.error(f"예측 오류: {e}")
    
    def _handle_fall_detection(self) -> None:
        """낙상 감지 처리"""
        current_time = datetime.now()
        
        # 연속된 낙상 감지를 방지 (3초 쿨다운)
        if hasattr(self, 'last_fall_time'):
            if (current_time - self.last_fall_time).seconds < 3:
                return
        
        self.fall_detected = True
        self.detection_count += 1
        self.last_fall_time = current_time
        
        self.logger.warning(f"🚨 낙상 감지! 예측값: {self.last_prediction:.3f}")
        
        # 콜백 호출
        if self.on_fall_detected:
            self.on_fall_detected(self.last_prediction, current_time)
    
    def start_detection(self) -> None:
        """실시간 감지 시작"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        self.is_running = True
        self.detection_count = 0
        
        # 데이터 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self.process_data_stream)
        self.processing_thread.start()
        
        self.logger.info("🟢 실시간 낙상 감지 시작")
    
    def stop_detection(self) -> None:
        """실시간 감지 중지"""
        self.is_running = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        self.logger.info("🔴 실시간 낙상 감지 중지")
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            'is_running': self.is_running,
            'buffer_size': len(self.data_buffer),
            'last_prediction': self.last_prediction,
            'threshold': self.threshold,
            'detection_count': self.detection_count,
            'total_predictions': len(self.prediction_buffer)
        }
    
    def get_recent_predictions(self, count: int = 20) -> list:
        """최근 예측 결과 반환"""
        return list(self.prediction_buffer)[-count:]
    
    def set_threshold(self, new_threshold: float) -> None:
        """감지 임계값 변경"""
        self.threshold = new_threshold
        self.logger.info(f"임계값 변경: {new_threshold}")
    
    def reset_buffer(self) -> None:
        """데이터 버퍼 초기화"""
        self.data_buffer.clear()
        self.prediction_buffer.clear()
        self.fall_detected = False
        self.logger.info("버퍼 초기화 완료")


class CSIDataSimulator:
    """CSI 데이터 시뮬레이터 (테스트용)"""
    
    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: 테스트용 CSI 데이터 디렉토리
        """
        self.data_dir = data_dir or Config.PROCESSED_DATA_DIR
        self.test_files = []
        self.current_file_idx = 0
        self.current_row_idx = 0
        self.current_data = None
        
        self._load_test_files()
    
    def _load_test_files(self) -> None:
        """테스트 파일들 로드"""
        import glob
        
        npz_files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        
        for file_path in npz_files[:5]:  # 처음 5개 파일만 사용
            try:
                data = np.load(file_path)
                if 'X' in data:
                    self.test_files.append({
                        'path': file_path,
                        'X': data['X'],
                        'y': data.get('y', np.zeros(len(data['X'])))
                    })
            except Exception as e:
                print(f"파일 로드 실패: {file_path} - {e}")
        
        print(f"📁 테스트 파일 {len(self.test_files)}개 로드됨")
    
    def get_next_sample(self) -> Optional[np.ndarray]:
        """다음 CSI 샘플 반환"""
        if not self.test_files:
            return None
        
        # 현재 파일의 다음 샘플
        current_file = self.test_files[self.current_file_idx]
        
        if self.current_row_idx >= len(current_file['X']):
            # 다음 파일로 이동
            self.current_file_idx = (self.current_file_idx + 1) % len(self.test_files)
            self.current_row_idx = 0
            current_file = self.test_files[self.current_file_idx]
        
        # 하나의 타임스텝 데이터 반환
        sample = current_file['X'][self.current_row_idx]
        
        # 첫 번째 타임스텝의 amplitude 데이터만 반환
        if len(sample.shape) > 1:
            amplitude_data = sample[0, :]  # 첫 번째 타임스텝
        else:
            amplitude_data = sample
        
        self.current_row_idx += 1
        
        return amplitude_data
    
    def get_sample_with_label(self) -> tuple:
        """라벨과 함께 샘플 반환"""
        sample = self.get_next_sample()
        if sample is None:
            return None, None
        
        # 해당 라벨 (근사치)
        current_file = self.test_files[self.current_file_idx]
        label = current_file['y'][min(self.current_row_idx-1, len(current_file['y'])-1)]
        
        return sample, label


def create_realtime_demo():
    """실시간 감지 데모 생성"""
    
    # 사용 가능한 모델 확인
    from evaluator import list_available_models
    
    available_models = list_available_models()
    if not available_models:
        print("❌ 사용 가능한 모델이 없습니다.")
        return None
    
    # 가장 최근 모델 선택
    model_name = available_models[-1]
    print(f"🎯 사용할 모델: {model_name}")
    
    # 실시간 감지기 생성
    detector = RealTimeCSIDetector(
        model_name=model_name,
        threshold=0.5
    )
    
    # 콜백 함수들 설정
    def on_fall_detected(prediction, timestamp):
        print(f"🚨 [낙상 감지] 시간: {timestamp.strftime('%H:%M:%S')}, "
              f"확률: {prediction:.1%}")
    
    def on_prediction_updated(prediction, threshold):
        status = "🚨 위험" if prediction > threshold else "✅ 안전"
        print(f"📊 {status} | 예측: {prediction:.3f} | 임계값: {threshold}")
    
    detector.on_fall_detected = on_fall_detected
    detector.on_prediction_updated = on_prediction_updated
    
    return detector


if __name__ == "__main__":
    print("🚀 CSI 실시간 낙상 감지 시스템")
    print("=" * 50)
    
    try:
        # 데모 생성
        detector = create_realtime_demo()
        if detector is None:
            exit(1)
        
        # 모델 로드
        print("📂 모델 로딩 중...")
        detector.load_model()
        
        # 데이터 시뮬레이터 생성
        print("📊 테스트 데이터 준비 중...")
        simulator = CSIDataSimulator()
        
        # 실시간 감지 시작
        print("🟢 실시간 감지 시작...")
        detector.start_detection()
        
        # 시뮬레이션 실행
        print("⚡ 데이터 스트리밍 시작 (Ctrl+C로 중지)")
        
        try:
            for i in range(1000):  # 1000개 샘플 테스트
                sample = simulator.get_next_sample()
                if sample is None:
                    break
                
                # 실시간 감지기에 데이터 추가
                detector.add_data_point(sample)
                
                # 실제 시간 간격 시뮬레이션 (100ms)
                time.sleep(0.1)
                
                # 상태 출력 (10개마다)
                if (i + 1) % 10 == 0:
                    status = detector.get_status()
                    print(f"📈 진행: {i+1}/1000, "
                          f"버퍼: {status['buffer_size']}/{detector.window_size}, "
                          f"감지: {status['detection_count']}회")
        
        except KeyboardInterrupt:
            print("\n⏹️ 사용자 중단")
        
        # 정리
        detector.stop_detection()
        
        # 최종 결과
        final_status = detector.get_status()
        print(f"\n📊 최종 결과:")
        print(f"   총 감지 횟수: {final_status['detection_count']}회")
        print(f"   총 예측 횟수: {final_status['total_predictions']}회")
        
        recent_predictions = detector.get_recent_predictions(10)
        if recent_predictions:
            print(f"\n📈 최근 예측 결과:")
            for pred in recent_predictions[-5:]:
                status = "🚨" if pred['is_fall'] else "✅"
                print(f"   {status} {pred['timestamp'].strftime('%H:%M:%S')} | "
                      f"{pred['prediction']:.3f}")
        
        print("\n✅ 실시간 감지 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
