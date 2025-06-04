# Python 코드 실행 전에 추가
import matplotlib.pyplot as plt
import platform

# Windows
if platform.system() == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False




# fall_timeline_analyzer.py
import pandas as pd
import numpy as np
import os
import glob
import pickle
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class FallTimelineAnalyzer:
    """낙상 시간대 분석 및 구간 검출기"""
    
    def __init__(self, confidence_threshold=0.5, fall_duration_threshold=3):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.metadata = None
        
        # 분석 설정
        self.confidence_threshold = confidence_threshold
        self.fall_duration_threshold = fall_duration_threshold  # 최소 연속 감지 횟수
        
        # 모델 정보
        self.window_size = None
        self.stride = None
        
        # 분석 결과
        self.fall_events = []
        self.timeline_data = None
    
    def load_model_and_preprocessors(self, model_path=None):
        """모델과 전처리기 로드"""
        if model_path is None:
            # 최신 모델 자동 탐지
            model_files = glob.glob("*model*.keras") + glob.glob("*model*.h5")
            if not model_files:
                print("❌ 모델 파일을 찾을 수 없습니다!")
                return False
            model_path = max(model_files, key=os.path.getctime)
        
        print(f"📥 모델 로딩: {os.path.basename(model_path)}")
        
        try:
            # 모델 로드
            self.model = load_model(model_path)
            print(f"   ✅ 모델 로드 완료")
            
            # 전처리기 로드
            base_path = model_path.replace('.keras', '').replace('.h5', '')
            
            # 스케일러
            scaler_path = base_path + '_scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"   ✅ 스케일러 로드 완료")
            
            # 특징 선택기
            selector_path = base_path + '_selector.pkl'
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                print(f"   ✅ 특징 선택기 로드 완료")
            
            # 메타데이터
            metadata_path = base_path + '_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                self.window_size = self.metadata.get('window_size', 50)
                self.stride = self.metadata.get('stride', 5)
                print(f"   📋 설정: 윈도우={self.window_size}, 스트라이드={self.stride}")
            else:
                self.window_size = 50
                self.stride = 5
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def preprocess_data(self, X):
        """데이터 전처리"""
        if self.feature_selector:
            if 'variance_selector' in self.feature_selector:
                X_var = self.feature_selector['variance_selector'].transform(X)
                if self.feature_selector.get('k_selector'):
                    X_selected = self.feature_selector['k_selector'].transform(X_var)
                else:
                    X_selected = X_var
            else:
                X_selected = X[:, 10:246] if X.shape[1] > 246 else X
        else:
            X_selected = X
        
        if self.scaler:
            X_normalized = self.scaler.transform(X_selected)
        else:
            X_normalized = X_selected
        
        return X_normalized
    
    def create_sequences_with_timestamps(self, X, timestamps):
        """타임스탬프와 함께 시퀀스 생성"""
        sequences = []
        sequence_timestamps = []
        sequence_start_times = []
        sequence_end_times = []
        
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            window_X = X[i:i + self.window_size]
            window_timestamps = timestamps[i:i + self.window_size]
            
            sequences.append(window_X)
            sequence_timestamps.append(window_timestamps)
            sequence_start_times.append(window_timestamps[0])
            sequence_end_times.append(window_timestamps[-1])
        
        return (np.array(sequences), 
                sequence_timestamps, 
                sequence_start_times, 
                sequence_end_times)
    
    def detect_fall_events(self, probabilities, start_times, end_times):
        """낙상 이벤트 감지 및 구간 분석"""
        print("🔍 낙상 이벤트 구간 분석...")
        
        # 임계값 이상인 시퀀스들 찾기
        fall_mask = probabilities >= self.confidence_threshold
        fall_indices = np.where(fall_mask)[0]
        
        if len(fall_indices) == 0:
            print("   ✅ 낙상 이벤트가 감지되지 않았습니다.")
            return []
        
        # 연속된 구간들로 그룹화
        fall_events = []
        current_event = None
        
        for i, idx in enumerate(fall_indices):
            if current_event is None:
                # 새로운 이벤트 시작
                current_event = {
                    'start_index': idx,
                    'end_index': idx,
                    'start_time': start_times[idx],
                    'end_time': end_times[idx],
                    'max_probability': probabilities[idx],
                    'avg_probability': probabilities[idx],
                    'sequence_count': 1,
                    'probabilities': [probabilities[idx]]
                }
            else:
                # 연속성 확인 (인덱스 차이가 5 이하면 연속으로 간주)
                if idx - current_event['end_index'] <= 5:
                    # 기존 이벤트 확장
                    current_event['end_index'] = idx
                    current_event['end_time'] = end_times[idx]
                    current_event['max_probability'] = max(current_event['max_probability'], probabilities[idx])
                    current_event['probabilities'].append(probabilities[idx])
                    current_event['sequence_count'] += 1
                else:
                    # 이전 이벤트 완료 및 새 이벤트 시작
                    if current_event['sequence_count'] >= self.fall_duration_threshold:
                        current_event['avg_probability'] = np.mean(current_event['probabilities'])
                        current_event['duration_seconds'] = self._calculate_duration(
                            current_event['start_time'], current_event['end_time']
                        )
                        fall_events.append(current_event)
                    
                    current_event = {
                        'start_index': idx,
                        'end_index': idx,
                        'start_time': start_times[idx],
                        'end_time': end_times[idx],
                        'max_probability': probabilities[idx],
                        'avg_probability': probabilities[idx],
                        'sequence_count': 1,
                        'probabilities': [probabilities[idx]]
                    }
        
        # 마지막 이벤트 처리
        if current_event and current_event['sequence_count'] >= self.fall_duration_threshold:
            current_event['avg_probability'] = np.mean(current_event['probabilities'])
            current_event['duration_seconds'] = self._calculate_duration(
                current_event['start_time'], current_event['end_time']
            )
            fall_events.append(current_event)
        
        # 결과 정리
        for i, event in enumerate(fall_events):
            event['event_id'] = i + 1
        
        print(f"   📊 감지된 낙상 이벤트: {len(fall_events)}개")
        
        return fall_events
    
    def _calculate_duration(self, start_time, end_time):
        """시간 차이 계산 (초 단위)"""
        try:
            if isinstance(start_time, str):
                start_dt = pd.to_datetime(start_time)
                end_dt = pd.to_datetime(end_time)
            else:
                start_dt = start_time
                end_dt = end_time
            
            duration = (end_dt - start_dt).total_seconds()
            return max(duration, 0)  # 음수 방지
        except:
            return 0
    
    def analyze_csv_timeline(self, csv_path):
        """CSV 파일의 전체 타임라인 분석"""
        print(f"📊 타임라인 분석 시작: {os.path.basename(csv_path)}")
        
        try:
            # 1. 데이터 로드
            df = pd.read_csv(csv_path)
            
            print(f"   📄 데이터 크기: {df.shape}")
            
            # 2. 타임스탬프 형식 확인 및 처리
            print(f"   🔍 타임스탬프 형식 확인...")
            
            # 타임스탬프 샘플 확인
            timestamp_samples = df['timestamp'].head(10).tolist()
            print(f"   📋 타임스탬프 샘플: {timestamp_samples[:3]}")
            
            # 다양한 타임스탬프 형식 시도
            timestamp_processed = False
            
            # 형식 1: 표준 datetime 형식
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"   ✅ 표준 datetime 형식으로 파싱 성공")
                timestamp_processed = True
            except:
                pass
            
            # 형식 2: 숫자 형태 (초 단위)
            if not timestamp_processed:
                try:
                    # 숫자로 변환 시도
                    timestamp_numeric = pd.to_numeric(df['timestamp'], errors='coerce')
                    if not timestamp_numeric.isna().all():
                        # 기준 시점 설정 (현재 날짜 00:00:00)
                        base_time = pd.Timestamp.now().normalize()
                        df['timestamp'] = base_time + pd.to_timedelta(timestamp_numeric, unit='s')
                        print(f"   ✅ 숫자(초) 형식으로 파싱 성공")
                        timestamp_processed = True
                except:
                    pass
            
            # 형식 3: MM:SS.f 형태 (분:초.소수)
            if not timestamp_processed:
                try:
                    def parse_mmss(time_str):
                        """MM:SS.f 형식을 초 단위로 변환"""
                        if isinstance(time_str, str) and ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 2:
                                minutes = float(parts[0])
                                seconds = float(parts[1])
                                return minutes * 60 + seconds
                        return None
                    
                    # MM:SS 형식 파싱 시도
                    timestamp_seconds = df['timestamp'].apply(parse_mmss)
                    
                    if not timestamp_seconds.isna().all():
                        # 유효한 값이 있으면 변환
                        base_time = pd.Timestamp.now().normalize()
                        df['timestamp'] = base_time + pd.to_timedelta(timestamp_seconds, unit='s')
                        print(f"   ✅ MM:SS 형식으로 파싱 성공")
                        timestamp_processed = True
                except Exception as e:
                    print(f"   ⚠️ MM:SS 파싱 실패: {e}")
            
            # 형식 4: 인덱스 기반 (파싱 실패 시)
            if not timestamp_processed:
                print(f"   ⚠️ 타임스탬프 파싱 실패, 인덱스 기반으로 생성")
                # 0.1초 간격으로 인덱스 기반 타임스탬프 생성
                base_time = pd.Timestamp.now().normalize()
                time_intervals = pd.to_timedelta(df.index * 0.1, unit='s')
                df['timestamp'] = base_time + time_intervals
                timestamp_processed = True
            
            print(f"   ⏰ 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            # 3. 특징 추출
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            X = df[feature_cols].values
            timestamps = df['timestamp'].values
            
            print(f"   📈 특징 컬럼: {len(feature_cols)}개")
            
            # 4. 전처리
            X_processed = self.preprocess_data(X)
            
            # 5. 시퀀스 생성
            X_seq, seq_timestamps, start_times, end_times = self.create_sequences_with_timestamps(
                X_processed, timestamps
            )
            
            print(f"   🔄 생성된 시퀀스: {len(X_seq)}개")
            
            # 6. 예측 수행
            print("   🔮 예측 수행 중...")
            probabilities = self.model.predict(X_seq, verbose=0).flatten()
            
            print(f"   📊 예측 완료 - 확률 범위: {probabilities.min():.3f} ~ {probabilities.max():.3f}")
            
            # 7. 낙상 이벤트 감지
            fall_events = self.detect_fall_events(probabilities, start_times, end_times)
            
            # 8. 타임라인 데이터 생성
            self.timeline_data = {
                'timestamps': start_times,
                'end_times': end_times,
                'probabilities': probabilities,
                'fall_mask': probabilities >= self.confidence_threshold,
                'original_timestamps': timestamps,
                'original_labels': df['label'].values if 'label' in df.columns else None
            }
            
            self.fall_events = fall_events
            
            # 9. 결과 출력
            self.print_fall_summary()
            
            return fall_events
            
        except Exception as e:
            print(f"❌ 타임라인 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def print_fall_summary(self):
        """낙상 이벤트 요약 출력"""
        print(f"\n📋 낙상 감지 결과 요약")
        print("=" * 80)
        
        if not self.fall_events:
            print("✅ 낙상 이벤트가 감지되지 않았습니다.")
            
            # 전체 통계
            if self.timeline_data:
                max_prob = np.max(self.timeline_data['probabilities'])
                avg_prob = np.mean(self.timeline_data['probabilities'])
                print(f"📊 전체 통계:")
                print(f"   최대 확률: {max_prob:.1%}")
                print(f"   평균 확률: {avg_prob:.1%}")
            return
        
        print(f"🚨 총 {len(self.fall_events)}개의 낙상 이벤트가 감지되었습니다!\n")
        
        for i, event in enumerate(self.fall_events):
            print(f"📅 낙상 이벤트 #{event['event_id']}")
            print(f"   ⏰ 시작 시간: {event['start_time']}")
            print(f"   ⏰ 종료 시간: {event['end_time']}")
            print(f"   ⏱️  지속 시간: {event['duration_seconds']:.1f}초")
            print(f"   📊 최대 확률: {event['max_probability']:.1%}")
            print(f"   📊 평균 확률: {event['avg_probability']:.1%}")
            print(f"   🔢 시퀀스 수: {event['sequence_count']}개")
            
            # 신뢰도 평가
            confidence_level = "높음" if event['max_probability'] > 0.8 else \
                             "중간" if event['max_probability'] > 0.6 else "낮음"
            print(f"   🎯 신뢰도: {confidence_level}")
            print()
        
        # 전체 통계
        total_fall_time = sum(event['duration_seconds'] for event in self.fall_events)
        max_prob_overall = max(event['max_probability'] for event in self.fall_events)
        avg_prob_overall = np.mean([event['avg_probability'] for event in self.fall_events])
        
        print(f"📊 전체 통계:")
        print(f"   총 낙상 시간: {total_fall_time:.1f}초")
        print(f"   최고 확률: {max_prob_overall:.1%}")
        print(f"   평균 확률: {avg_prob_overall:.1%}")
    
    def visualize_timeline(self, save_plot=True):
        """타임라인 시각화"""
        if not self.timeline_data:
            print("❌ 타임라인 데이터가 없습니다!")
            return
        
        print("📊 타임라인 시각화 중...")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        timestamps = pd.to_datetime(self.timeline_data['timestamps'])
        probabilities = self.timeline_data['probabilities']
        
        # 1. 전체 확률 타임라인
        axes[0].plot(timestamps, probabilities, linewidth=1.5, color='blue', alpha=0.7)
        axes[0].axhline(y=self.confidence_threshold, color='red', linestyle='--', 
                       alpha=0.8, label=f'임계값 ({self.confidence_threshold:.1%})')
        
        # 낙상 구간 강조
        if self.fall_events:
            for event in self.fall_events:
                start_time = pd.to_datetime(event['start_time'])
                end_time = pd.to_datetime(event['end_time'])
                axes[0].axvspan(start_time, end_time, alpha=0.3, color='red',
                               label='낙상 구간' if event == self.fall_events[0] else "")
        
        axes[0].set_title('낙상 확률 타임라인')
        axes[0].set_ylabel('낙상 확률')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 2. 낙상 감지 상태 (이진)
        fall_binary = (probabilities >= self.confidence_threshold).astype(int)
        axes[1].fill_between(timestamps, fall_binary, alpha=0.6, color='red', 
                            step='pre', label='낙상 감지')
        axes[1].set_title('낙상 감지 상태')
        axes[1].set_ylabel('낙상 감지 (1=감지, 0=정상)')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 3. 실제 라벨과 비교 (있는 경우)
        if self.timeline_data['original_labels'] is not None:
            original_timestamps = pd.to_datetime(self.timeline_data['original_timestamps'])
            original_labels = self.timeline_data['original_labels']
            
            axes[2].fill_between(original_timestamps, original_labels, alpha=0.6, 
                               color='green', step='pre', label='실제 라벨')
            
            # 예측 결과도 함께 표시
            axes[2].fill_between(timestamps, fall_binary*0.5, alpha=0.6, 
                               color='red', step='pre', label='예측 결과')
            
            axes[2].set_title('실제 라벨 vs 예측 결과')
            axes[2].set_ylabel('라벨 (1=낙상, 0=정상)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        else:
            # 낙상 이벤트 상세 정보 표시
            axes[2].bar(range(len(self.fall_events)), 
                       [event['max_probability'] for event in self.fall_events],
                       alpha=0.7, color='red')
            axes[2].set_title('낙상 이벤트별 최대 확률')
            axes[2].set_ylabel('최대 확률')
            axes[2].set_xlabel('이벤트 번호')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fall_timeline_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 타임라인 저장: {filename}")
        
        plt.show()
    
    def export_fall_events(self, output_file=None):
        """낙상 이벤트를 파일로 내보내기"""
        if not self.fall_events:
            print("📋 내보낼 낙상 이벤트가 없습니다.")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'fall_events_{timestamp}.json'
        
        # JSON 형태로 저장
        export_data = {
            'analysis_time': datetime.now().isoformat(),
            'settings': {
                'confidence_threshold': self.confidence_threshold,
                'fall_duration_threshold': self.fall_duration_threshold,
                'window_size': self.window_size,
                'stride': self.stride
            },
            'summary': {
                'total_events': len(self.fall_events),
                'total_fall_time': sum(event['duration_seconds'] for event in self.fall_events)
            },
            'events': []
        }
        
        for event in self.fall_events:
            event_data = {
                'event_id': event['event_id'],
                'start_time': event['start_time'].isoformat() if hasattr(event['start_time'], 'isoformat') else str(event['start_time']),
                'end_time': event['end_time'].isoformat() if hasattr(event['end_time'], 'isoformat') else str(event['end_time']),
                'duration_seconds': event['duration_seconds'],
                'max_probability': float(event['max_probability']),
                'avg_probability': float(event['avg_probability']),
                'sequence_count': int(event['sequence_count'])
            }
            export_data['events'].append(event_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 낙상 이벤트 내보내기 완료: {output_file}")
        return output_file

def analyze_fall_timeline(csv_path, confidence_threshold=0.5):
    """빠른 낙상 타임라인 분석"""
    analyzer = FallTimelineAnalyzer(confidence_threshold=confidence_threshold)
    
    if not analyzer.load_model_and_preprocessors():
        return None
    
    fall_events = analyzer.analyze_csv_timeline(csv_path)
    
    if fall_events:
        analyzer.visualize_timeline()
        analyzer.export_fall_events()
    
    return analyzer

if __name__ == "__main__":
    import sys
    
    print("📊 낙상 타임라인 분석기")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    else:
        # 테스트 파일들
        test_files = [
            "../csi_data/case1/12_labeled.csv",
        ]
        
        csv_file = None
        for file_path in test_files:
            if os.path.exists(file_path):
                csv_file = file_path
                break
        
        if not csv_file:
            print("❌ 테스트할 CSV 파일이 없습니다!")
            print("사용법: python fall_timeline_analyzer.py <csv_file> [confidence_threshold]")
            exit(1)
        
        confidence = 0.5
    
    print(f"📄 분석 파일: {csv_file}")
    print(f"🎯 신뢰도 임계값: {confidence}")
    
    analyzer = analyze_fall_timeline(csv_file, confidence)
    
    if analyzer and analyzer.fall_events:
        print(f"\n🎉 분석 완료! {len(analyzer.fall_events)}개의 낙상 이벤트가 발견되었습니다.")
    else:
        print("✅ 낙상 이벤트가 감지되지 않았습니다.")