# analyzer.py
"""
낙상 타임라인 분석기
학습된 모델로 CSI 데이터 분석 및 낙상 이벤트 감지
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model

from config import CSIConfig

class FallTimelineAnalyzer:
    """낙상 타임라인 분석기 클래스"""
    
    def __init__(self, model_path=None, confidence_threshold=None):
        """
        Args:
            model_path: 모델 파일 경로 (None이면 자동 탐지)
            confidence_threshold: 낙상 감지 신뢰도 임계값
        """
        self.model = None
        self.scaler = None
        self.metadata = None
        
        # 설정값
        self.confidence_threshold = confidence_threshold or CSIConfig.CONFIDENCE_THRESHOLD
        self.window_size = CSIConfig.WINDOW_SIZE
        self.stride = CSIConfig.STRIDE
        self.active_range = CSIConfig.ACTIVE_FEATURE_RANGE
        
        # 분석 결과
        self.fall_events = []
        self.timeline_data = None
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 모델 로드
        if model_path or self._find_latest_model():
            self.load_model_system(model_path)
    
    def _find_latest_model(self):
        """최신 모델 자동 탐지"""
        patterns = [
            os.path.join(CSIConfig.MODEL_SAVE_DIR, "*complete*.keras"),
            os.path.join(CSIConfig.MODEL_SAVE_DIR, "*model*.keras"),
            "*complete*.keras",
            "*model*.keras"
        ]
        
        model_files = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))
        
        if model_files:
            # 가장 최근 파일 반환
            return max(model_files, key=os.path.getctime)
        
        return None
    
    def load_model_system(self, model_path=None):
        """완전한 모델 시스템 로드"""
        if model_path is None:
            model_path = self._find_latest_model()
        
        if not model_path:
            raise ValueError("❌ 모델 파일을 찾을 수 없습니다!")
        
        self.logger.info(f"📥 모델 시스템 로딩: {os.path.basename(model_path)}")
        
        # 1. 모델 로드 (호환성 개선)
        try:
            # 첫 번째 시도: 기본 로드
            self.model = load_model(model_path)
            self.logger.info("   ✅ 모델 로드 완료")
        except Exception as e1:
            self.logger.warning(f"   ⚠️ 기본 로드 실패, 호환성 모드 시도: {e1}")
            try:
                # 두 번째 시도: compile=False로 로드
                self.model = load_model(model_path, compile=False)
                
                # 수동으로 컴파일 (옵티마이저, 손실함수 다시 설정)
                from tensorflow.keras.optimizers import Adam
                self.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                self.logger.info("   ✅ 호환성 모드로 모델 로드 완료")
                
            except Exception as e2:
                self.logger.error(f"   ❌ 모든 로드 방법 실패: {e2}")
                return False
            
            # 2. 스케일러 로드
            scaler_path = model_path.replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info("   ✅ 스케일러 로드 완료")
            else:
                self.logger.warning("   ⚠️ 스케일러 파일 없음")
            
            # 3. 메타데이터 로드
            metadata_path = model_path.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                # 설정값 업데이트
                model_config = self.metadata.get('model_config', {})
                self.window_size = model_config.get('window_size', self.window_size)
                self.stride = model_config.get('stride', self.stride)
                self.active_range = tuple(model_config.get('active_range', self.active_range))
                
                self.logger.info(f"   📋 설정 로드: 윈도우={self.window_size}, 스트라이드={self.stride}")
                
                # 훈련 통계 출력
                training_stats = self.metadata.get('training_stats', {})
                if training_stats:
                    self.logger.info(f"   📊 모델 성능: {training_stats.get('model_params', 0):,} 파라미터")
                    
                    test_results = training_stats.get('test_results')
                    if test_results and 'accuracy' in test_results:
                        self.logger.info(f"   🎯 테스트 정확도: {test_results['accuracy']:.1%}")
            
            self.logger.info("✅ 모델 시스템 로드 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            return False
    
    def preprocess_data(self, X):
        """데이터 전처리"""
        # 활성 특성만 선택
        start_feat, end_feat = self.active_range
        if X.shape[1] > end_feat:
            X_active = X[:, start_feat:end_feat+1]
        else:
            X_active = X
        
        # 정규화
        if self.scaler:
            X_normalized = self.scaler.transform(X_active)
        else:
            # 스케일러가 없는 경우 간단한 정규화
            X_normalized = (X_active - X_active.mean()) / (X_active.std() + 1e-8)
        
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
        
        return (np.array(sequences), sequence_timestamps, 
                sequence_start_times, sequence_end_times)
    
    def detect_fall_events(self, probabilities, start_times, end_times, 
                          fall_duration_threshold=None):
        """낙상 이벤트 감지 및 구간 분석"""
        fall_duration_threshold = fall_duration_threshold or CSIConfig.FALL_DURATION_THRESHOLD
        
        self.logger.info("🔍 낙상 이벤트 구간 분석...")
        
        # 임계값 이상인 시퀀스들 찾기
        fall_mask = probabilities >= self.confidence_threshold
        fall_indices = np.where(fall_mask)[0]
        
        if len(fall_indices) == 0:
            self.logger.info("   ✅ 낙상 이벤트가 감지되지 않았습니다.")
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
                    'sequence_count': 1,
                    'probabilities': [probabilities[idx]]
                }
            else:
                # 연속성 확인 (5개 간격 이하면 연속으로 간주)
                if idx - current_event['end_index'] <= 5:
                    # 기존 이벤트 확장
                    current_event['end_index'] = idx
                    current_event['end_time'] = end_times[idx]
                    current_event['max_probability'] = max(current_event['max_probability'], 
                                                          probabilities[idx])
                    current_event['probabilities'].append(probabilities[idx])
                    current_event['sequence_count'] += 1
                else:
                    # 이전 이벤트 완료 및 새 이벤트 시작
                    if current_event['sequence_count'] >= fall_duration_threshold:
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
                        'sequence_count': 1,
                        'probabilities': [probabilities[idx]]
                    }
        
        # 마지막 이벤트 처리
        if current_event and current_event['sequence_count'] >= fall_duration_threshold:
            current_event['avg_probability'] = np.mean(current_event['probabilities'])
            current_event['duration_seconds'] = self._calculate_duration(
                current_event['start_time'], current_event['end_time']
            )
            fall_events.append(current_event)
        
        # 결과 정리
        for i, event in enumerate(fall_events):
            event['event_id'] = i + 1
            
            # 신뢰도 레벨 추가
            if event['max_probability'] > 0.8:
                event['confidence_level'] = 'high'
            elif event['max_probability'] > 0.6:
                event['confidence_level'] = 'medium'
            else:
                event['confidence_level'] = 'low'
        
        self.logger.info(f"   📊 감지된 낙상 이벤트: {len(fall_events)}개")
        
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
    
    def _process_timestamps(self, df):
        """다양한 타임스탬프 형식 처리"""
        try:
            # 1. 표준 datetime 형식 시도
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return True
        except:
            pass
        
        try:
            # 2. 숫자 형태 (초 단위) 처리
            timestamp_numeric = pd.to_numeric(df['timestamp'], errors='coerce')
            if not timestamp_numeric.isna().all():
                base_time = pd.Timestamp.now().normalize()
                df['timestamp'] = base_time + pd.to_timedelta(timestamp_numeric, unit='s')
                return True
        except:
            pass
        
        try:
            # 3. MM:SS.f 형식 처리
            def parse_mmss(time_str):
                if isinstance(time_str, str) and ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 2:
                        minutes = float(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds
                return None
            
            timestamp_seconds = df['timestamp'].apply(parse_mmss)
            if not timestamp_seconds.isna().all():
                base_time = pd.Timestamp.now().normalize()
                df['timestamp'] = base_time + pd.to_timedelta(timestamp_seconds, unit='s')
                return True
        except:
            pass
        
        # 4. 인덱스 기반 생성 (최후 수단)
        base_time = pd.Timestamp.now().normalize()
        time_intervals = pd.to_timedelta(df.index * 0.1, unit='s')
        df['timestamp'] = base_time + time_intervals
        return False
    
    def analyze_csv_timeline(self, csv_path):
        """CSV 파일의 전체 타임라인 분석"""
        self.logger.info(f"📊 타임라인 분석: {os.path.basename(csv_path)}")
        
        if not self.model:
            raise ValueError("❌ 모델이 로드되지 않았습니다!")
        
        try:
            # 1. 데이터 로드
            df = pd.read_csv(csv_path)
            self.logger.info(f"   📄 데이터 크기: {df.shape}")
            
            # 2. 타임스탬프 처리
            self.logger.info("   🔍 타임스탬프 처리...")
            timestamp_parsed = self._process_timestamps(df)
            
            if timestamp_parsed:
                self.logger.info("   ✅ 타임스탬프 파싱 성공")
            else:
                self.logger.info("   ⚠️ 인덱스 기반 타임스탬프 생성")
            
            # 시간 범위 출력
            time_range = df['timestamp'].max() - df['timestamp'].min()
            self.logger.info(f"   ⏰ 측정 시간: {time_range.total_seconds():.1f}초")
            
            # 3. 특성 추출
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            if not feature_cols:
                raise ValueError("특성 컬럼이 없습니다!")
            
            X = df[feature_cols].values
            timestamps = df['timestamp'].values
            
            self.logger.info(f"   📈 특성 컬럼: {len(feature_cols)}개")
            
            # 4. 전처리
            X_processed = self.preprocess_data(X)
            self.logger.info(f"   🔧 전처리 완료: {X_processed.shape}")
            
            # 5. 시퀀스 생성
            X_seq, seq_timestamps, start_times, end_times = self.create_sequences_with_timestamps(
                X_processed, timestamps
            )
            
            self.logger.info(f"   🔄 생성된 시퀀스: {len(X_seq)}개")
            
            # 6. 예측 수행
            self.logger.info("   🔮 예측 수행 중...")
            probabilities = self.model.predict(X_seq, verbose=0).flatten()

            # 온도 스케일링으로 과신 보정
            probabilities = self.temperature_scaling(probabilities)
            
            self.logger.info(f"   📊 예측 완료 - 확률 범위: {probabilities.min():.3f} ~ {probabilities.max():.3f}")
            
            # 7. 낙상 이벤트 감지
            fall_events = self.detect_fall_events(probabilities, start_times, end_times)
            
            # 8. 타임라인 데이터 생성
            self.timeline_data = {
                'timestamps': start_times,
                'end_times': end_times,
                'probabilities': probabilities,
                'fall_mask': probabilities >= self.confidence_threshold,
                'original_timestamps': timestamps,
                'original_labels': df['label'].values if 'label' in df.columns else None,
                'file_info': {
                    'filename': os.path.basename(csv_path),
                    'total_samples': len(df),
                    'total_sequences': len(X_seq),
                    'measurement_duration': time_range.total_seconds(),
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            self.fall_events = fall_events
            
            # 9. 결과 요약
            self.print_fall_summary()
            
            return fall_events
            
        except Exception as e:
            self.logger.error(f"❌ 타임라인 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def print_fall_summary(self):
        """낙상 이벤트 요약 출력"""
        print(f"\n📋 낙상 감지 결과 요약")
        print("=" * 80)
        
        # 파일 정보
        if self.timeline_data and 'file_info' in self.timeline_data:
            file_info = self.timeline_data['file_info']
            print(f"📁 파일: {file_info['filename']}")
            print(f"📊 데이터: {file_info['total_samples']:,}개 샘플 → {file_info['total_sequences']:,}개 시퀀스")
            print(f"⏰ 측정 시간: {file_info['measurement_duration']:.1f}초")
            print(f"🎯 신뢰도 임계값: {file_info['confidence_threshold']:.1%}")
        
        if not self.fall_events:
            print("\n✅ 낙상 이벤트가 감지되지 않았습니다.")
            
            if self.timeline_data:
                max_prob = np.max(self.timeline_data['probabilities'])
                avg_prob = np.mean(self.timeline_data['probabilities'])
                high_prob_count = np.sum(self.timeline_data['probabilities'] > 0.3)
                
                print(f"\n📊 전체 통계:")
                print(f"   최대 확률: {max_prob:.1%}")
                print(f"   평균 확률: {avg_prob:.1%}")
                print(f"   30% 이상 확률: {high_prob_count}개 시퀀스")
            return
        
        print(f"\n🚨 총 {len(self.fall_events)}개의 낙상 이벤트가 감지되었습니다!")
        
        for event in self.fall_events:
            print(f"\n📅 낙상 이벤트 #{event['event_id']}")
            print(f"   ⏰ 시작: {event['start_time']}")
            print(f"   ⏰ 종료: {event['end_time']}")
            print(f"   ⏱️  지속: {event['duration_seconds']:.1f}초")
            print(f"   📊 최대 확률: {event['max_probability']:.1%}")
            print(f"   📊 평균 확률: {event['avg_probability']:.1%}")
            print(f"   🔢 시퀀스: {event['sequence_count']}개")
            
            # 신뢰도 표시
            confidence_icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            confidence_names = {'high': '높음', 'medium': '중간', 'low': '낮음'}
            
            confidence = event['confidence_level']
            icon = confidence_icons.get(confidence, '⚪')
            name = confidence_names.get(confidence, '알 수 없음')
            
            print(f"   🎯 신뢰도: {icon} {name}")
        
        # 전체 통계
        total_fall_time = sum(event['duration_seconds'] for event in self.fall_events)
        max_prob_overall = max(event['max_probability'] for event in self.fall_events)
        avg_prob_overall = np.mean([event['avg_probability'] for event in self.fall_events])
        
        high_confidence_count = sum(1 for event in self.fall_events if event['confidence_level'] == 'high')
        medium_confidence_count = sum(1 for event in self.fall_events if event['confidence_level'] == 'medium')
        low_confidence_count = sum(1 for event in self.fall_events if event['confidence_level'] == 'low')
        
        print(f"\n📊 전체 통계:")
        print(f"   총 낙상 시간: {total_fall_time:.1f}초")
        print(f"   최고 확률: {max_prob_overall:.1%}")
        print(f"   평균 확률: {avg_prob_overall:.1%}")
        print(f"   신뢰도 분포: 🔴{high_confidence_count}개, 🟡{medium_confidence_count}개, 🟢{low_confidence_count}개")
    
    def visualize_timeline(self, save_plot=True, figsize=(15, 12)):
        """타임라인 시각화"""
        if not self.timeline_data:
            print("❌ 타임라인 데이터가 없습니다!")
            return
        
        print("📊 타임라인 시각화 중...")
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        timestamps = pd.to_datetime(self.timeline_data['timestamps'])
        probabilities = self.timeline_data['probabilities']
        
        # 1. 확률 타임라인
        axes[0].plot(timestamps, probabilities, linewidth=1.5, color='blue', alpha=0.7, label='낙상 확률')
        axes[0].axhline(y=self.confidence_threshold, color='red', linestyle='--', 
                       alpha=0.8, label=f'임계값 ({self.confidence_threshold:.1%})')
        
        # 낙상 구간 강조
        if self.fall_events:
            for i, event in enumerate(self.fall_events):
                start_time = pd.to_datetime(event['start_time'])
                end_time = pd.to_datetime(event['end_time'])
                
                # 신뢰도에 따른 색상
                color_map = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}
                color = color_map.get(event['confidence_level'], 'red')
                
                axes[0].axvspan(start_time, end_time, alpha=0.3, color=color,
                               label=f'낙상 구간 ({event["confidence_level"]})' if i == 0 else "")
        
        axes[0].set_title('낙상 확률 타임라인', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('낙상 확률')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 2. 낙상 감지 상태
        fall_binary = (probabilities >= self.confidence_threshold).astype(int)
        axes[1].fill_between(timestamps, fall_binary, alpha=0.6, color='red', 
                            step='pre', label='낙상 감지')
        
        axes[1].set_title('낙상 감지 상태', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('감지 상태 (1=낙상, 0=정상)')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 3. 실제 라벨 비교 또는 이벤트 상세
        if self.timeline_data['original_labels'] is not None:
            original_timestamps = pd.to_datetime(self.timeline_data['original_timestamps'])
            original_labels = self.timeline_data['original_labels']
            
            axes[2].fill_between(original_timestamps, original_labels, alpha=0.6, 
                               color='green', step='pre', label='실제 라벨')
            axes[2].fill_between(timestamps, fall_binary*0.5, alpha=0.6, 
                               color='red', step='pre', label='예측 결과')
            
            axes[2].set_title('실제 라벨 vs 예측 결과 비교', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('라벨 값')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        else:
            # 낙상 이벤트 상세 정보
            if self.fall_events:
                event_probs = [event['max_probability'] for event in self.fall_events]
                event_names = [f"#{event['event_id']}\n({event['confidence_level']})" 
                              for event in self.fall_events]
                
                # 신뢰도별 색상
                colors = []
                for event in self.fall_events:
                    if event['confidence_level'] == 'high':
                        colors.append('red')
                    elif event['confidence_level'] == 'medium':
                        colors.append('orange')
                    else:
                        colors.append('gold')
                
                bars = axes[2].bar(range(len(self.fall_events)), event_probs, 
                                  alpha=0.7, color=colors)
                
                axes[2].set_title('낙상 이벤트별 최대 확률', fontsize=14, fontweight='bold')
                axes[2].set_ylabel('최대 확률')
                axes[2].set_xlabel('이벤트')
                axes[2].set_xticks(range(len(self.fall_events)))
                axes[2].set_xticklabels(event_names)
                
                # 값 표시
                for bar, prob in zip(bars, event_probs):
                    height = bar.get_height()
                    axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
            else:
                axes[2].text(0.5, 0.5, '감지된 낙상 이벤트가 없습니다', 
                           ha='center', va='center', transform=axes[2].transAxes,
                           fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[2].set_title('낙상 이벤트 없음', fontsize=14, fontweight='bold')
            
            axes[2].grid(True, alpha=0.3)
        
        # 전체 제목
        file_info = self.timeline_data.get('file_info', {})
        filename = file_info.get('filename', 'Unknown')
        fig.suptitle(f'CSI 낙상 감지 분석 결과: {filename}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = CSIConfig.RESULTS_DIR
            filename = f'fall_timeline_analysis_{timestamp}.png'
            filepath = os.path.join(results_dir, filename)
            
            plt.savefig(filepath, dpi=CSIConfig.VISUALIZATION_DPI, bbox_inches='tight')
            print(f"   💾 타임라인 저장: {filepath}")
        
        plt.show()
    
    def export_fall_events(self, output_file=None):
        """낙상 이벤트 결과 내보내기"""
        if not self.fall_events:
            print("📋 내보낼 낙상 이벤트가 없습니다.")
            return None
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = CSIConfig.RESULTS_DIR
            output_file = os.path.join(results_dir, f'fall_events_{timestamp}.json')
        
        # 분석 정보
        analysis_info = {
            'analysis_time': datetime.now().isoformat(),
            'model_info': {
                'confidence_threshold': self.confidence_threshold,
                'window_size': self.window_size,
                'stride': self.stride,
                'active_range': self.active_range
            },
            'file_info': self.timeline_data.get('file_info', {}) if self.timeline_data else {}
        }
        
        # 요약 정보
        summary = {
            'total_events': len(self.fall_events),
            'total_fall_time': sum(event['duration_seconds'] for event in self.fall_events),
            'max_probability': max(event['max_probability'] for event in self.fall_events),
            'avg_probability': np.mean([event['avg_probability'] for event in self.fall_events]),
            'confidence_distribution': {
                'high': sum(1 for e in self.fall_events if e['confidence_level'] == 'high'),
                'medium': sum(1 for e in self.fall_events if e['confidence_level'] == 'medium'),
                'low': sum(1 for e in self.fall_events if e['confidence_level'] == 'low')
            }
        }
        
        # 이벤트 상세 정보
        events_detail = []
        for event in self.fall_events:
            event_data = {
                'event_id': event['event_id'],
                'start_time': event['start_time'].isoformat() if hasattr(event['start_time'], 'isoformat') else str(event['start_time']),
                'end_time': event['end_time'].isoformat() if hasattr(event['end_time'], 'isoformat') else str(event['end_time']),
                'duration_seconds': event['duration_seconds'],
                'max_probability': float(event['max_probability']),
                'avg_probability': float(event['avg_probability']),
                'sequence_count': int(event['sequence_count']),
                'confidence_level': event['confidence_level'],
                'start_index': int(event['start_index']),
                'end_index': int(event['end_index'])
            }
            events_detail.append(event_data)
        
        # 전체 데이터 구성
        export_data = {
            'analysis_info': analysis_info,
            'summary': summary,
            'events': events_detail
        }
        
        # JSON 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 낙상 이벤트 분석 결과 저장: {output_file}")
        return output_file
    
    def analyze_and_visualize(self, csv_path, save_results=True):
        """분석 + 시각화 + 결과 저장 통합 함수"""
        print(f"🔍 종합 분석 시작: {os.path.basename(csv_path)}")
        print("=" * 60)
        
        try:
            # 1. 타임라인 분석
            fall_events = self.analyze_csv_timeline(csv_path)
            
            # 2. 시각화
            if self.timeline_data:
                self.visualize_timeline(save_plot=save_results)
            
            # 3. 결과 저장
            if save_results and fall_events:
                self.export_fall_events()
            
            print(f"\n🎉 종합 분석 완료!")
            print(f"   감지된 이벤트: {len(fall_events)}개")
            
            return fall_events
            
        except Exception as e:
            print(f"❌ 종합 분석 실패: {e}")
            return []

    def temperature_scaling(self, probabilities, temperature=7):
        """온도 스케일링으로 과신된 확률 보정"""
        
        # 과신 여부 체크
        high_conf_ratio = np.mean(probabilities > 0.95)
        
        if high_conf_ratio > 0.3:  # 30% 이상이 95% 넘으면 과신
            self.logger.warning(f"   ⚠️ 모델 과신 감지: {high_conf_ratio:.1%}가 95% 이상")
            self.logger.info(f"   🌡️ 온도 스케일링 적용 (T={temperature})")
            
            # 확률을 로짓으로 변환
            logits = np.log(probabilities / (1 - probabilities + 1e-8))
            
            # 온도로 나누기 (부드럽게 만들기)
            scaled_logits = logits / temperature
            
            # 다시 확률로 변환
            calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
            
            # 결과 출력
            new_high_conf_ratio = np.mean(calibrated_probs > 0.95)
            self.logger.info(f"   📊 보정 결과: 95% 이상 {high_conf_ratio:.1%} → {new_high_conf_ratio:.1%}")
            self.logger.info(f"   📊 평균 확률: {np.mean(probabilities):.3f} → {np.mean(calibrated_probs):.3f}")
            
            return calibrated_probs
        
        else:
            self.logger.info(f"   ✅ 정상적인 확률 분포 (99% 이상: {high_conf_ratio:.1%})")
            return probabilities

if __name__ == "__main__":
    import sys
    
    print("🔍 CSI 낙상 타임라인 분석기")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # 명령행 인자 처리
        csv_file = sys.argv[1]
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else CSIConfig.CONFIDENCE_THRESHOLD
        model_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"📁 분석 파일: {csv_file}")
        print(f"🎯 신뢰도 임계값: {confidence:.1%}")
        
        try:
            analyzer = FallTimelineAnalyzer(model_path, confidence)
            analyzer.analyze_and_visualize(csv_file)
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
    
    else:
        # 기본 테스트
        test_file = "35.csv"
        if os.path.exists(test_file):
            print(f"🧪 기본 테스트: {test_file}")
            
            try:
                analyzer = FallTimelineAnalyzer(confidence_threshold=0.3)
                analyzer.analyze_and_visualize(test_file)
            except Exception as e:
                print(f"❌ 테스트 실패: {e}")
        else:
            print(f"❌ 테스트 파일이 없습니다: {test_file}")
            print("💡 사용법:")
            print("  python analyzer.py <csv_file> [confidence_threshold] [model_path]")