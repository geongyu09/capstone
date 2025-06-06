# trainer.py
"""
CSI 낙상 감지 시스템 메인 훈련 클래스
대용량 데이터셋 처리 및 모델 학습 관리
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from config import CSIConfig
from data_generator import CSIDataGenerator
from model_builder import CSIModelBuilder

class CSITrainer:
    """CSI 낙상 감지 시스템 메인 훈련 클래스"""
    
    def __init__(self, data_directory=None, model_type='cnn_lstm_hybrid'):
        """
        Args:
            data_directory: 데이터 디렉토리 경로
            model_type: 사용할 모델 타입
        """
        self.data_directory = data_directory or CSIConfig.DEFAULT_DATA_DIR
        self.model_type = model_type
        
        # 전처리기
        self.scaler = RobustScaler()
        self.model_builder = None
        self.model = None
        
        # 설정값
        self.window_size = CSIConfig.WINDOW_SIZE
        self.stride = CSIConfig.STRIDE
        self.active_range = CSIConfig.ACTIVE_FEATURE_RANGE
        self.overlap_threshold = CSIConfig.OVERLAP_THRESHOLD
        
        # 학습 통계
        self.training_stats = {}
        self.data_stats = {}
        
        # 로깅 설정
        self.setup_logging()
        
        self.logger.info("🚀 CSI 트레이너 초기화 완료")
        self.logger.info(f"   데이터 디렉토리: {self.data_directory}")
        self.logger.info(f"   모델 타입: {self.model_type}")
    
    def setup_logging(self):
        """로깅 시스템 설정"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(CSIConfig.LOG_DIR, f'csi_training_{timestamp}.log')
        
        # 로거 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📋 로그 파일: {log_file}")
    
    def discover_csv_files(self, max_files=None):
        """CSV 파일 자동 탐지"""
        self.logger.info("🔍 CSV 파일 탐지 중...")
        
        patterns = [
            os.path.join(self.data_directory, "*.csv"),
            os.path.join(self.data_directory, "**", "*.csv"),
            "*.csv",  # 현재 디렉토리
        ]
        
        csv_files = []
        for pattern in patterns:
            found_files = glob.glob(pattern, recursive=True)
            csv_files.extend(found_files)
        
        # 중복 제거 및 정렬
        csv_files = sorted(list(set(csv_files)))
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        self.logger.info(f"📁 발견된 CSV 파일: {len(csv_files)}개")
        
        # 파일 정보 미리보기
        for i, file_path in enumerate(csv_files[:5]):
            try:
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.logger.info(f"   {i+1}. {os.path.basename(file_path)} ({file_size:.1f} KB)")
            except:
                self.logger.info(f"   {i+1}. {os.path.basename(file_path)}")
        
        if len(csv_files) > 5:
            self.logger.info(f"   ... 외 {len(csv_files)-5}개 파일")
        
        return csv_files
    
    def analyze_data_characteristics(self, sample_files=10):
        """데이터 특성 분석"""
        self.logger.info("🔬 데이터 특성 분석 시작...")
        
        csv_files = self.discover_csv_files()[:sample_files]
        
        if not csv_files:
            raise ValueError("❌ 분석할 CSV 파일이 없습니다!")
        
        total_samples = 0
        total_fall_samples = 0
        sampling_rates = []
        active_features_counts = []
        file_durations = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # 기본 통계
                total_samples += len(df)
                fall_samples = np.sum(df['label'] == 1) if 'label' in df.columns else 0
                total_fall_samples += fall_samples
                
                # 샘플링 주파수 계산
                if 'timestamp' in df.columns:
                    try:
                        timestamps = pd.to_datetime(df['timestamp'])
                        duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
                        if duration > 0:
                            sampling_rate = len(df) / duration
                            sampling_rates.append(sampling_rate)
                            file_durations.append(duration)
                    except:
                        pass
                
                # 활성 특성 수 계산
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                if feature_cols:
                    X = df[feature_cols].values
                    # 분산이 0이 아닌 특성 개수
                    active_count = np.sum(np.var(X, axis=0) > 1e-10)
                    active_features_counts.append(active_count)
                
            except Exception as e:
                self.logger.warning(f"파일 분석 실패: {file_path} - {e}")
                continue
        
        # 통계 계산
        avg_sampling_rate = np.mean(sampling_rates) if sampling_rates else CSIConfig.SAMPLING_RATE
        avg_active_features = np.mean(active_features_counts) if active_features_counts else CSIConfig.ACTIVE_FEATURE_COUNT
        fall_ratio = total_fall_samples / total_samples if total_samples > 0 else 0
        total_duration = np.sum(file_durations) if file_durations else 0
        
        # 결과 저장
        self.data_stats = {
            'total_files_analyzed': len(csv_files),
            'total_samples': total_samples,
            'total_fall_samples': total_fall_samples,
            'fall_ratio': fall_ratio,
            'avg_sampling_rate': avg_sampling_rate,
            'avg_active_features': avg_active_features,
            'total_duration_seconds': total_duration,
            'avg_file_duration': total_duration / len(file_durations) if file_durations else 0
        }
        
        # 결과 출력
        self.logger.info(f"📊 데이터 특성 분석 결과:")
        self.logger.info(f"   분석 파일: {len(csv_files)}개")
        self.logger.info(f"   총 샘플: {total_samples:,}개")
        self.logger.info(f"   낙상 비율: {fall_ratio:.1%}")
        self.logger.info(f"   평균 샘플링 주파수: {avg_sampling_rate:.0f}Hz")
        self.logger.info(f"   평균 활성 특성: {avg_active_features:.0f}개")
        self.logger.info(f"   총 측정 시간: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
        
        # 설정 자동 조정
        if avg_sampling_rate > 200:
            suggested_window = int(0.5 * avg_sampling_rate)
            suggested_stride = int(0.05 * avg_sampling_rate)
            
            if abs(suggested_window - self.window_size) > 10:
                self.logger.info(f"💡 권장 윈도우 크기: {suggested_window}개 (현재: {self.window_size})")
            if abs(suggested_stride - self.stride) > 5:
                self.logger.info(f"💡 권장 스트라이드: {suggested_stride}개 (현재: {self.stride})")
        
        return self.data_stats
    
    def prepare_scaler(self, sample_files=15):
        """스케일러 사전 학습"""
        self.logger.info("🔧 스케일러 준비 중...")
        
        csv_files = self.discover_csv_files()[:sample_files]
        sample_data = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                if not feature_cols:
                    continue
                
                X = df[feature_cols].values
                
                # 활성 특성만 선택
                start_feat, end_feat = self.active_range
                if X.shape[1] > end_feat:
                    X_active = X[:, start_feat:end_feat+1]
                else:
                    X_active = X
                
                # 메모리 절약을 위한 샘플링
                sample_size = min(len(X_active), 2000)
                if len(X_active) > sample_size:
                    indices = np.random.choice(len(X_active), sample_size, replace=False)
                    X_sampled = X_active[indices]
                else:
                    X_sampled = X_active
                
                sample_data.append(X_sampled)
                
            except Exception as e:
                self.logger.warning(f"스케일러 샘플 파일 스킵: {file_path} - {e}")
                continue
        
        if not sample_data:
            raise ValueError("❌ 스케일러 학습용 데이터가 없습니다!")
        
        # 스케일러 학습
        X_combined = np.vstack(sample_data)
        self.scaler.fit(X_combined)
        
        self.logger.info(f"✅ 스케일러 학습 완료: {X_combined.shape} 샘플")
        self.logger.info(f"   특성 범위: [{X_combined.min():.3f}, {X_combined.max():.3f}]")
        
        return self.scaler
    
    def create_data_generators(self, validation_split=None, test_split=None):
        """훈련/검증/테스트 데이터 제너레이터 생성"""
        validation_split = validation_split or CSIConfig.VALIDATION_SPLIT
        test_split = test_split or CSIConfig.TEST_SPLIT
        
        self.logger.info("📊 데이터 제너레이터 생성...")
        
        csv_files = self.discover_csv_files()
        
        if len(csv_files) < 3:
            raise ValueError("❌ 최소 3개 이상의 파일이 필요합니다!")
        
        # 파일 레벨에서 분할 (데이터 리크 방지)
        np.random.shuffle(csv_files)
        
        test_size = max(1, int(len(csv_files) * test_split))
        val_size = max(1, int(len(csv_files) * validation_split))
        train_size = len(csv_files) - test_size - val_size
        
        if train_size < 1:
            raise ValueError("❌ 훈련 파일이 부족합니다!")
        
        train_files = csv_files[:train_size]
        val_files = csv_files[train_size:train_size + val_size]
        test_files = csv_files[train_size + val_size:]
        
        self.logger.info(f"📈 데이터 분할:")
        self.logger.info(f"   훈련 파일: {len(train_files)}개")
        self.logger.info(f"   검증 파일: {len(val_files)}개")
        self.logger.info(f"   테스트 파일: {len(test_files)}개")
        
        # 제너레이터 생성
        train_generator = CSIDataGenerator(
            file_list=train_files,
            batch_size=CSIConfig.BATCH_SIZE,
            window_size=self.window_size,
            stride=self.stride,
            scaler=self.scaler,
            active_range=self.active_range,
            shuffle=True,
            logger=self.logger
        )
        
        val_generator = CSIDataGenerator(
            file_list=val_files,
            batch_size=CSIConfig.BATCH_SIZE,
            window_size=self.window_size,
            stride=self.stride,
            scaler=self.scaler,
            active_range=self.active_range,
            shuffle=False,
            logger=self.logger
        )
        
        test_generator = CSIDataGenerator(
            file_list=test_files,
            batch_size=CSIConfig.BATCH_SIZE,
            window_size=self.window_size,
            stride=self.stride,
            scaler=self.scaler,
            active_range=self.active_range,
            shuffle=False,
            logger=self.logger
        ) if test_files else None
        
        # 통계 출력
        self.logger.info(f"🔄 예상 시퀀스:")
        self.logger.info(f"   훈련: {train_generator.total_sequences:,}개")
        self.logger.info(f"   검증: {val_generator.total_sequences:,}개")
        if test_generator:
            self.logger.info(f"   테스트: {test_generator.total_sequences:,}개")
        
        return train_generator, val_generator, test_generator
    
    def build_model(self):
        """모델 구축"""
        self.logger.info(f"🏗️ {self.model_type} 모델 구축...")
        
        # 입력 형태 계산
        feature_count = self.active_range[1] - self.active_range[0] + 1
        input_shape = (self.window_size, feature_count)
        
        # 모델 빌더 생성
        self.model_builder = CSIModelBuilder(input_shape, self.logger)
        
        # 모델 구축
        self.model = self.model_builder.build_model(self.model_type)
        
        return self.model
    
    def train_model(self, epochs=None):
        """모델 학습 실행"""
        epochs = epochs or CSIConfig.EPOCHS
        
        self.logger.info("🚀 모델 학습 시작!")
        self.logger.info("=" * 60)
        
        try:
            # 1. 데이터 특성 분석
            self.analyze_data_characteristics()
            
            # 2. 스케일러 준비
            self.prepare_scaler()
            
            # 3. 데이터 제너레이터 생성
            train_gen, val_gen, test_gen = self.create_data_generators()
            
            # 4. 모델 구축
            self.build_model()
            
            # 5. 콜백 설정
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = os.path.join(
                CSIConfig.MODEL_SAVE_DIR, 
                f'best_csi_{self.model_type}_{timestamp}.keras'
            )
            
            callbacks = self.model_builder.create_callbacks(model_save_path)
            
            # 6. 학습 실행
            self.logger.info(f"📚 학습 시작: {epochs} 에포크")
            
            history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # 7. 테스트 평가 (있는 경우)
            test_results = None
            if test_gen:
                self.logger.info("📊 테스트 평가 중...")
                test_results = self.model.evaluate(test_gen, verbose=0)
                test_metrics = dict(zip(self.model.metrics_names, test_results))
                
                self.logger.info("🎯 최종 테스트 결과:")
                for metric, value in test_metrics.items():
                    if metric != 'loss':
                        self.logger.info(f"   {metric}: {value:.1%}")
                    else:
                        self.logger.info(f"   {metric}: {value:.4f}")
            
            # 8. 훈련 통계 저장
            self.training_stats = {
                'model_type': self.model_type,
                'model_params': self.model.count_params(),
                'data_stats': self.data_stats,
                'train_sequences': train_gen.total_sequences,
                'val_sequences': val_gen.total_sequences,
                'test_sequences': test_gen.total_sequences if test_gen else 0,
                'epochs_trained': len(history.history['loss']),
                'best_val_loss': min(history.history['val_loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'test_results': test_metrics if test_gen else None,
                'training_time': datetime.now().isoformat()
            }
            
            # 9. 완전한 시스템 저장
            complete_save_path = model_save_path.replace('.keras', '_complete.keras')
            self.save_complete_system(complete_save_path)
            
            self.logger.info("✅ 학습 완료!")
            
            return history
            
        except Exception as e:
            self.logger.error(f"❌ 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_complete_system(self, model_path):
        """완전한 시스템 저장 (모델 + 전처리기 + 메타데이터)"""
        self.logger.info(f"💾 완전한 시스템 저장: {model_path}")
        
        try:
            # 1. 모델 저장
            self.model.save(model_path)
            
            # 2. 스케일러 저장
            scaler_path = model_path.replace('.keras', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # 3. 메타데이터 저장
            metadata = {
                'system_info': {
                    'version': '2.0',
                    'created_at': datetime.now().isoformat(),
                    'model_type': self.model_type
                },
                'model_config': {
                    'window_size': self.window_size,
                    'stride': self.stride,
                    'active_range': self.active_range,
                    'overlap_threshold': self.overlap_threshold,
                    'input_shape': self.model.input_shape[1:] if self.model else None
                },
                'training_config': CSIConfig.get_data_config(),
                'training_stats': self.training_stats,
                'data_stats': self.data_stats
            }
            
            metadata_path = model_path.replace('.keras', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # 4. 저장된 파일들 정보
            saved_files = {
                'model': model_path,
                'scaler': scaler_path,
                'metadata': metadata_path
            }
            
            # 파일 크기 정보
            total_size = 0
            for file_type, file_path in saved_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    total_size += file_size
                    self.logger.info(f"   📄 {file_type}: {os.path.basename(file_path)} ({file_size:.1f} MB)")
            
            self.logger.info(f"   📦 총 크기: {total_size:.1f} MB")
            self.logger.info("✅ 완전한 시스템 저장 완료")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 저장 실패: {e}")
            return {}
    
    def quick_train(self, csv_file, epochs=10):
        """단일 파일로 빠른 학습 (테스트용)"""
        self.logger.info(f"🧪 빠른 학습 모드: {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"파일이 없습니다: {csv_file}")
        
        try:
            # 1. 데이터 로드
            df = pd.read_csv(csv_file)
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            
            if not feature_cols:
                raise ValueError("특성 컬럼이 없습니다!")
            
            X = df[feature_cols].values
            y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
            
            # 2. 활성 특성만 선택
            start_feat, end_feat = self.active_range
            if X.shape[1] > end_feat:
                X_active = X[:, start_feat:end_feat+1]
            else:
                X_active = X
            
            # 3. 스케일러 학습
            self.scaler.fit(X_active)
            X_normalized = self.scaler.transform(X_active)
            
            # 4. 시퀀스 생성
            sequences = []
            labels = []
            
            for i in range(0, len(X_normalized) - self.window_size + 1, self.stride):
                window_X = X_normalized[i:i + self.window_size]
                window_y = y[i:i + self.window_size]
                
                # 라벨링
                fall_ratio = np.mean(window_y == 1)
                sequence_label = 1 if fall_ratio >= self.overlap_threshold else 0
                
                sequences.append(window_X)
                labels.append(sequence_label)
            
            X_seq = np.array(sequences)
            y_seq = np.array(labels)
            
            self.logger.info(f"   생성된 시퀀스: {len(X_seq)}개")
            self.logger.info(f"   라벨 분포: 정상 {np.sum(y_seq==0)}, 낙상 {np.sum(y_seq==1)}")
            
            # 5. 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
            
            # 6. 모델 구축
            self.build_model()
            
            # 7. 학습
            self.logger.info(f"🎓 빠른 학습 시작 ({epochs} 에포크)...")
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=min(16, len(X_train)),
                verbose=1
            )
            
            # 8. 평가
            test_results = self.model.evaluate(X_test, y_test, verbose=0)
            test_metrics = dict(zip(self.model.metrics_names, test_results))
            
            self.logger.info("🎯 빠른 학습 결과:")
            for metric, value in test_metrics.items():
                if metric != 'loss':
                    self.logger.info(f"   {metric}: {value:.1%}")
                else:
                    self.logger.info(f"   {metric}: {value:.4f}")
            
            # 9. 저장
            timestamp = datetime.now().strftime('%H%M%S')
            quick_model_path = f"quick_model_{timestamp}.keras"
            self.save_complete_system(quick_model_path)
            
            return history, test_metrics
            
        except Exception as e:
            self.logger.error(f"❌ 빠른 학습 실패: {e}")
            raise
    
    def resume_training(self, model_path, additional_epochs=10):
        """기존 모델로부터 학습 재개"""
        self.logger.info(f"🔄 학습 재개: {model_path}")
        
        try:
            from tensorflow.keras.models import load_model
            
            # 1. 기존 모델 로드
            self.model = load_model(model_path)
            self.logger.info("   ✅ 기존 모델 로드 완료")
            
            # 2. 스케일러 로드
            scaler_path = model_path.replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info("   ✅ 스케일러 로드 완료")
            else:
                self.logger.warning("   ⚠️ 스케일러 파일 없음, 새로 학습")
                self.prepare_scaler()
            
            # 3. 메타데이터 로드
            metadata_path = model_path.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                model_config = metadata.get('model_config', {})
                self.window_size = model_config.get('window_size', self.window_size)
                self.stride = model_config.get('stride', self.stride)
                self.active_range = tuple(model_config.get('active_range', self.active_range))
                
                self.logger.info(f"   📋 설정 로드: 윈도우={self.window_size}, 스트라이드={self.stride}")
            
            # 4. 데이터 제너레이터 생성
            train_gen, val_gen, test_gen = self.create_data_generators()
            
            # 5. 추가 학습
            self.logger.info(f"🎓 추가 학습: {additional_epochs} 에포크")
            
            # 콜백 설정
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_model_path = model_path.replace('.keras', f'_resumed_{timestamp}.keras')
            
            from model_builder import CSIModelBuilder
            temp_builder = CSIModelBuilder(logger=self.logger)
            callbacks = temp_builder.create_callbacks(new_model_path)
            
            # 학습 실행
            history = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=additional_epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # 6. 새로운 모델 저장
            self.save_complete_system(new_model_path)
            
            self.logger.info("✅ 학습 재개 완료!")
            
            return history
            
        except Exception as e:
            self.logger.error(f"❌ 학습 재개 실패: {e}")
            raise
    
    def get_training_summary(self):
        """훈련 요약 정보"""
        if not self.training_stats:
            return "훈련 통계가 없습니다."
        
        summary = []
        summary.append("📊 훈련 요약")
        summary.append("=" * 40)
        
        # 모델 정보
        summary.append(f"🏗️ 모델 정보:")
        summary.append(f"   타입: {self.training_stats.get('model_type', 'Unknown')}")
        summary.append(f"   파라미터: {self.training_stats.get('model_params', 0):,}개")
        
        # 데이터 정보
        summary.append(f"\n📊 데이터 정보:")
        data_stats = self.training_stats.get('data_stats', {})
        summary.append(f"   총 샘플: {data_stats.get('total_samples', 0):,}개")
        summary.append(f"   낙상 비율: {data_stats.get('fall_ratio', 0):.1%}")
        summary.append(f"   샘플링 주파수: {data_stats.get('avg_sampling_rate', 0):.0f}Hz")
        
        # 학습 정보
        summary.append(f"\n🎓 학습 정보:")
        summary.append(f"   훈련 시퀀스: {self.training_stats.get('train_sequences', 0):,}개")
        summary.append(f"   검증 시퀀스: {self.training_stats.get('val_sequences', 0):,}개")
        summary.append(f"   학습 에포크: {self.training_stats.get('epochs_trained', 0)}개")
        
        # 성능 정보
        summary.append(f"\n🎯 성능 정보:")
        summary.append(f"   최종 훈련 손실: {self.training_stats.get('final_train_loss', 0):.4f}")
        summary.append(f"   최종 검증 손실: {self.training_stats.get('final_val_loss', 0):.4f}")
        summary.append(f"   최고 검증 손실: {self.training_stats.get('best_val_loss', 0):.4f}")
        
        # 테스트 결과
        test_results = self.training_stats.get('test_results')
        if test_results:
            summary.append(f"\n📈 테스트 결과:")
            for metric, value in test_results.items():
                if metric != 'loss':
                    summary.append(f"   {metric}: {value:.1%}")
                else:
                    summary.append(f"   {metric}: {value:.4f}")
        
        return '\n'.join(summary)
    
    def print_training_summary(self):
        """훈련 요약 출력"""
        print(self.get_training_summary())

if __name__ == "__main__":
    # 테스트 코드
    import sys
    
    print("🧪 CSI 트레이너 테스트")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # 명령행 인자가 있는 경우
        command = sys.argv[1]
        
        if command == "quick" and len(sys.argv) > 2:
            # 빠른 테스트
            csv_file = sys.argv[2]
            epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            
            trainer = CSITrainer(model_type='cnn_lstm_hybrid')
            history, metrics = trainer.quick_train(csv_file, epochs)
            trainer.print_training_summary()
            
        elif command == "full":
            # 전체 학습
            data_dir = sys.argv[2] if len(sys.argv) > 2 else CSIConfig.DEFAULT_DATA_DIR
            epochs = int(sys.argv[3]) if len(sys.argv) > 3 else CSIConfig.EPOCHS
            
            trainer = CSITrainer(data_directory=data_dir, model_type='cnn_lstm_hybrid')
            history = trainer.train_model(epochs)
            trainer.print_training_summary()
            
        else:
            print("❌ 잘못된 명령어")
            print("사용법:")
            print("  python trainer.py quick <csv_file> [epochs]")
            print("  python trainer.py full [data_dir] [epochs]")
    
    else:
        # 기본 테스트: 35.csv 파일로 빠른 학습
        test_file = "35.csv"
        if os.path.exists(test_file):
            print(f"🎯 기본 테스트: {test_file}")
            
            trainer = CSITrainer(model_type='cnn_lstm_hybrid')
            history, metrics = trainer.quick_train(test_file, epochs=3)
            trainer.print_training_summary()
            
            print("✅ 기본 테스트 완료!")
        else:
            print(f"❌ 테스트 파일이 없습니다: {test_file}")
            print("💡 사용법:")
            print("  python trainer.py quick <csv_file>")
            print("  python trainer.py full [data_directory]")