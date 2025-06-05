# multi_file_trainer.py
import pandas as pd
import numpy as np
import glob
import os
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 전역 설정값
DEFAULT_TEST_CSV = 'case32.csv'         # 기본 테스트 CSV 파일
DEFAULT_DATA_DIR = './capstone/csi_data'         # 기본 데이터 디렉토리
DEFAULT_WINDOW_SIZE = 30                # 기본 윈도우 크기 (테스트용)
DEFAULT_STRIDE = 5                      # 기본 스트라이드
DEFAULT_OVERLAP_THRESHOLD = 0.3         # 기본 임계값
DEFAULT_TARGET_FEATURES = 32            # 기본 특징 수 (테스트용)
DEFAULT_EPOCHS = 5                      # 기본 에포크 (테스트용)
DEFAULT_BATCH_SIZE = 16                 # 기본 배치 크기

# 추가 테스트 파일들 (자동 탐지용)
POSSIBLE_TEST_FILES = [
    'case32.csv',
    '32_labeled.csv', 
    'data.csv',
    'test.csv'
]

# 추가 데이터 디렉토리들 (자동 탐지용)
POSSIBLE_DATA_DIRS = [
    './csi_data',
    '../csi_data',
    './data',
    '../data',
    './csv_data',
    '../csv_data'
]

class MultiFileCSITrainer:
    """100개 이상의 CSI 파일로 대규모 학습을 수행하는 클래스"""
    
    def __init__(self, window_size=50, stride=5, overlap_threshold=0.3):
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.model = None
        
        # 로깅 설정
        self.setup_logging()
        
        # 상태 저장용
        self.training_stats = {}
    
    def setup_logging(self):
        """로깅 시스템 설정"""
        log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 CSI 학습 시스템 시작")
        self.logger.info(f"📋 설정: 윈도우={self.window_size}, 스트라이드={self.stride}, 임계값={self.overlap_threshold}")
    
    def analyze_single_file(self, file_path):
        """단일 CSV 파일 분석 (테스트용)"""
        print(f"\n📁 파일 분석: {os.path.basename(file_path)}")
        
        try:
            # CSV 읽기
            df = pd.read_csv(file_path)
            print(f"   📊 파일 형태: {df.shape}")
            print(f"   📋 컬럼 수: {len(df.columns)}")
            
            # 컬럼 확인
            print(f"   📝 주요 컬럼:")
            if 'timestamp' in df.columns:
                print(f"      ✅ timestamp: {df['timestamp'].dtype}")
            else:
                print(f"      ❌ timestamp 컬럼 없음")
            
            if 'label' in df.columns:
                label_counts = df['label'].value_counts().sort_index()
                print(f"      ✅ label: {dict(label_counts)}")
            else:
                print(f"      ❌ label 컬럼 없음")
            
            # 특징 컬럼 확인
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            print(f"      📈 특징 컬럼: {len(feature_cols)}개")
            
            if len(feature_cols) > 0:
                # 특징 통계
                X = df[feature_cols].values
                non_zero_features = np.sum(np.any(X != 0, axis=0))
                print(f"      📊 활성 특징: {non_zero_features}/{len(feature_cols)}개")
                
                # 일부 특징 값 확인
                sample_features = feature_cols[:5] + feature_cols[-5:]
                print(f"      🔍 샘플 특징값:")
                for feat in sample_features:
                    values = df[feat].values
                    print(f"         {feat}: [{values.min():.2f}, {values.max():.2f}], 평균 {values.mean():.2f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")
            return False
    
    def load_multiple_csv_files(self, data_directory, file_pattern="*.csv", max_files=None):
        """여러 CSV 파일을 메모리 효율적으로 로드"""
        self.logger.info(f"📁 다중 파일 로딩 시작: {data_directory}")
        
        # 파일 목록 수집
        if os.path.isfile(data_directory):
            # 단일 파일인 경우
            csv_files = [data_directory]
        else:
            # 디렉토리인 경우
            file_pattern_path = os.path.join(data_directory, "**", file_pattern)
            csv_files = glob.glob(file_pattern_path, recursive=True)
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        self.logger.info(f"   발견된 CSV 파일: {len(csv_files)}개")
        
        if len(csv_files) == 0:
            raise ValueError(f"❌ {data_directory}에서 CSV 파일을 찾을 수 없습니다!")
        
        # 파일 목록 출력
        for i, file_path in enumerate(csv_files[:5]):  # 처음 5개만
            print(f"   {i+1}. {os.path.basename(file_path)}")
        if len(csv_files) > 5:
            print(f"   ... 외 {len(csv_files)-5}개 파일")
        
        all_data = []
        all_labels = []
        file_info = []
        successful_files = 0
        failed_files = 0
        
        for i, file_path in enumerate(csv_files):
            try:
                self.logger.info(f"   로딩 중 ({i+1}/{len(csv_files)}): {os.path.basename(file_path)}")
                
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                
                # 기본 검증
                required_cols = ['timestamp', 'label']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"   ⚠️ 필수 컬럼 누락 {missing_cols}: {file_path}")
                    failed_files += 1
                    continue
                
                # 특징 컬럼 추출
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                if len(feature_cols) == 0:
                    self.logger.warning(f"   ⚠️ 특징 컬럼 없음: {file_path}")
                    failed_files += 1
                    continue
                
                # 데이터 추출
                X_file = df[feature_cols].values
                y_file = df['label'].values
                
                # 데이터 품질 검사
                if len(X_file) < self.window_size:
                    self.logger.warning(f"   ⚠️ 데이터 부족 (< {self.window_size}): {file_path}")
                    failed_files += 1
                    continue
                
                # 라벨 분포 확인
                unique_labels = np.unique(y_file)
                label_counts = np.bincount(y_file, minlength=2)
                fall_ratio = label_counts[1] / len(y_file) if len(y_file) > 0 else 0
                
                all_data.append(X_file)
                all_labels.append(y_file)
                
                file_info.append({
                    'filename': os.path.basename(file_path),
                    'path': file_path,
                    'samples': len(X_file),
                    'features': len(feature_cols),
                    'fall_ratio': fall_ratio,
                    'normal_count': int(label_counts[0]),
                    'fall_count': int(label_counts[1])
                })
                
                successful_files += 1
                
                # 진행 상황 출력
                if i % 10 == 0 and i > 0:
                    total_samples = sum(len(data) for data in all_data)
                    self.logger.info(f"   📊 중간 통계: {successful_files}개 파일, {total_samples:,}개 샘플")
                
            except Exception as e:
                self.logger.error(f"   ❌ 파일 로드 실패 {file_path}: {e}")
                failed_files += 1
                continue
        
        if len(all_data) == 0:
            raise ValueError("❌ 사용 가능한 데이터가 없습니다!")
        
        # 모든 데이터 결합
        self.logger.info("🔄 데이터 결합 중...")
        X_combined = np.vstack(all_data)
        y_combined = np.hstack(all_labels)
        
        # 최종 통계
        total_samples = len(X_combined)
        total_features = X_combined.shape[1]
        fall_samples = np.sum(y_combined == 1)
        normal_samples = np.sum(y_combined == 0)
        
        self.training_stats.update({
            'total_files': len(csv_files),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_samples': total_samples,
            'total_features': total_features,
            'fall_samples': fall_samples,
            'normal_samples': normal_samples,
            'imbalance_ratio': normal_samples / max(fall_samples, 1)
        })
        
        self.logger.info(f"✅ 로딩 완료!")
        self.logger.info(f"   성공: {successful_files}개, 실패: {failed_files}개")
        self.logger.info(f"   총 샘플: {total_samples:,}개")
        self.logger.info(f"   총 특징: {total_features}개")
        self.logger.info(f"   라벨 분포: 정상 {normal_samples:,}개, 낙상 {fall_samples:,}개")
        self.logger.info(f"   불균형 비율: {normal_samples/max(fall_samples,1):.1f}:1")
        
        return X_combined, y_combined, file_info
    
    def smart_feature_selection(self, X, y, target_features=64):
        """스마트 특징 선택"""
        self.logger.info("🎯 특징 선택 시작...")
        
        original_features = X.shape[1]
        self.logger.info(f"   원본 특징 수: {original_features}")
        
        # 1단계: 분산이 거의 0인 특징 제거
        self.logger.info("   1️⃣ 저분산 특징 제거...")
        variance_threshold = VarianceThreshold(threshold=0.01)
        X_var = variance_threshold.fit_transform(X)
        
        removed_by_variance = original_features - X_var.shape[1]
        self.logger.info(f"      제거된 특징: {removed_by_variance}개")
        self.logger.info(f"      남은 특징: {X_var.shape[1]}개")
        
        # 2단계: 통계적 특징 선택 (필요한 경우)
        if X_var.shape[1] > target_features:
            self.logger.info(f"   2️⃣ 상위 {target_features}개 특징 선택...")
            
            # 메모리 절약을 위해 샘플링 (데이터가 큰 경우)
            if len(X_var) > 30000:
                sample_size = 30000
                sample_indices = np.random.choice(len(X_var), sample_size, replace=False)
                X_sample = X_var[sample_indices]
                y_sample = y[sample_indices]
                self.logger.info(f"      샘플링: {len(X_sample):,}개로 특징 선택 수행")
            else:
                X_sample = X_var
                y_sample = y
            
            # F-통계량 기반 특징 선택
            k_selector = SelectKBest(f_classif, k=target_features)
            X_selected = k_selector.fit_transform(X_sample, y_sample)
            
            # 전체 데이터에 적용
            X_final = k_selector.transform(X_var)
            
            self.logger.info(f"      최종 선택: {X_final.shape[1]}개")
            
            # 선택기 저장
            self.feature_selector = {
                'variance_selector': variance_threshold,
                'k_selector': k_selector,
                'selected_indices': k_selector.get_support(indices=True)
            }
        else:
            X_final = X_var
            self.feature_selector = {
                'variance_selector': variance_threshold,
                'k_selector': None,
                'selected_indices': np.arange(X_var.shape[1])
            }
        
        self.logger.info(f"✅ 특징 선택 완료: {original_features} → {X_final.shape[1]}개")
        return X_final
    
    def create_sequences(self, X, y):
        """시계열 시퀀스 생성"""
        self.logger.info(f"⏰ 시퀀스 생성 시작...")
        self.logger.info(f"   설정: 윈도우={self.window_size}, 스트라이드={self.stride}")
        
        sequences = []
        labels = []
        
        total_possible = (len(X) - self.window_size) // self.stride + 1
        self.logger.info(f"   생성 가능한 시퀀스: 최대 {total_possible:,}개")
        
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            # 윈도우 추출
            window_X = X[i:i + self.window_size]
            window_y = y[i:i + self.window_size]
            
            # 스마트 라벨링
            fall_ratio = np.sum(window_y == 1) / len(window_y)
            
            if fall_ratio >= self.overlap_threshold:
                # 충분한 낙상 비율
                sequence_label = 1
            elif fall_ratio > 0:
                # 일부 낙상이 있는 경우 - 위치 고려
                fall_indices = np.where(window_y == 1)[0]
                
                # 낙상이 윈도우 후반부(70% 이후) 또는 전반부(30% 이전)에 있으면 중요
                if (len(fall_indices) > 0 and 
                    (fall_indices[-1] >= len(window_y) * 0.7 or fall_indices[0] <= len(window_y) * 0.3)):
                    sequence_label = 1
                else:
                    sequence_label = 0
            else:
                sequence_label = 0
            
            sequences.append(window_X)
            labels.append(sequence_label)
            
            # 진행 상황 출력
            if len(sequences) % 5000 == 0:
                self.logger.info(f"   진행: {len(sequences):,}개 시퀀스 생성됨")
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels)
        
        # 라벨 분포
        unique_labels, counts = np.unique(y_seq, return_counts=True)
        self.logger.info(f"✅ 시퀀스 생성 완료: {X_seq.shape[0]:,}개")
        
        for label, count in zip(unique_labels, counts):
            label_name = "낙상" if label == 1 else "정상"
            percentage = (count / len(y_seq)) * 100
            self.logger.info(f"   {label_name}: {count:,}개 ({percentage:.1f}%)")
        
        return X_seq, y_seq
    
    def preprocess_data(self, X):
        """데이터 전처리 (정규화)"""
        self.logger.info("🔧 데이터 전처리...")
        
        # 원본 형태 저장
        original_shape = X.shape
        
        # 2D로 변환하여 정규화
        X_2d = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_2d)
        
        # 원본 형태로 복원
        X_normalized = X_normalized.reshape(original_shape)
        
        self.logger.info(f"   ✅ 정규화 완료")
        self.logger.info(f"   데이터 범위: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        
        return X_normalized
    
    def build_model(self, input_shape):
        """LSTM 모델 구축"""
        self.logger.info(f"🏗️ 모델 구축...")
        self.logger.info(f"   입력 형태: {input_shape}")
        
        n_features = input_shape[1]
        
        # 특징 수에 따른 모델 선택
        if n_features > 64:
            # 고차원용 모델
            self.logger.info("   📦 고차원 특징용 모델")
            model = Sequential([
                Dense(64, activation='relu', input_shape=input_shape, name='feature_reduction'),
                Dropout(0.3),
                BatchNormalization(),
                
                LSTM(64, return_sequences=True, name='lstm_1'),
                Dropout(0.4),
                
                LSTM(32, return_sequences=False, name='lstm_2'),
                Dropout(0.4),
                
                Dense(16, activation='relu', name='dense_1'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(1, activation='sigmoid', name='output')
            ])
        else:
            # 표준 모델
            self.logger.info("   🏛️ 표준 LSTM 모델")
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_1'),
                Dropout(0.4),
                
                LSTM(32, return_sequences=False, name='lstm_2'),
                Dropout(0.4),
                
                Dense(16, activation='relu', name='dense_1'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(1, activation='sigmoid', name='output')
            ])
        
        # 컴파일 (metrics 문제 해결)
        try:
            # TensorFlow 2.x 호환성을 위한 metrics 설정
            from tensorflow.keras import metrics
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    metrics.BinaryAccuracy(name='accuracy'),
                    metrics.Precision(name='precision'),
                    metrics.Recall(name='recall')
                ]
            )
        except Exception as e:
            # 백업 방법: 기본 metrics만 사용
            self.logger.warning(f"   ⚠️ 고급 metrics 실패, 기본 설정 사용: {e}")
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        self.model = model
        self.logger.info(f"✅ 모델 구축 완료")
        self.logger.info(f"   파라미터 수: {model.count_params():,}개")
        
        # 모델 구조 출력
        model.summary()
        
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """모델 학습"""
        self.logger.info("🚀 모델 학습 시작...")
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 클래스 가중치 계산
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"   클래스 가중치: {class_weight_dict}")
        self.logger.info(f"   훈련 데이터: {X_train.shape}")
        self.logger.info(f"   에포크: {epochs}, 배치: {batch_size}")
        
        # 학습 실행
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            validation_split=0.2 if X_val is None else None,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        self.logger.info("✅ 학습 완료!")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """모델 평가"""
        self.logger.info("📊 모델 평가...")
        
        try:
            # 평가 실행
            results = self.model.evaluate(X_test, y_test, verbose=0)
            
            # 결과 정리 (metrics 이름 확인)
            metrics_names = self.model.metrics_names
            performance = {}
            
            for i, metric_name in enumerate(metrics_names):
                if i < len(results):
                    performance[metric_name] = float(results[i])
            
            # 로그 출력
            for metric, value in performance.items():
                if metric == 'loss':
                    self.logger.info(f"   {metric}: {value:.4f}")
                else:
                    self.logger.info(f"   {metric}: {value:.4f} ({value*100:.1f}%)")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ 모델 평가 실패: {e}")
            
            # 수동 평가 (백업)
            try:
                y_pred = self.model.predict(X_test, verbose=0)
                y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                accuracy = accuracy_score(y_test, y_pred_binary)
                
                # precision과 recall 계산 (zero_division 처리)
                try:
                    precision = precision_score(y_test, y_pred_binary, zero_division=0)
                    recall = recall_score(y_test, y_pred_binary, zero_division=0)
                except:
                    precision = 0.0
                    recall = 0.0
                
                performance = {
                    'loss': 0.0,  # 계산하지 않음
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                }
                
                self.logger.info("   📊 수동 평가 결과:")
                for metric, value in performance.items():
                    if metric == 'loss':
                        self.logger.info(f"   {metric}: N/A")
                    else:
                        self.logger.info(f"   {metric}: {value:.4f} ({value*100:.1f}%)")
                
                return performance
                
            except Exception as e2:
                self.logger.error(f"❌ 수동 평가도 실패: {e2}")
                return {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    def save_model(self, model_path):
        """모델과 전처리기 저장 - JSON 직렬화 문제 해결"""
        self.logger.info(f"💾 모델 저장: {model_path}")
        
        try:
            # 1. 모델 저장 (Keras 네이티브 형식 사용)
            if model_path.endswith('.h5'):
                # .keras 확장자로 변경 (권장 형식)
                keras_model_path = model_path.replace('.h5', '.keras')
                self.model.save(keras_model_path)
                self.logger.info(f"   📦 모델 저장: {keras_model_path} (Keras 네이티브 형식)")
                model_path = keras_model_path
            else:
                self.model.save(model_path)
            
            # 2. 전처리기 저장
            scaler_path = model_path.replace('.keras', '_scaler.pkl').replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # 3. 특징 선택기 저장
            selector_path = None
            if self.feature_selector:
                selector_path = model_path.replace('.keras', '_selector.pkl').replace('.h5', '_selector.pkl')
                with open(selector_path, 'wb') as f:
                    pickle.dump(self.feature_selector, f)
            
            # 4. 메타데이터 저장 - JSON 직렬화 안전 처리
            def make_json_safe(obj):
                """모든 객체를 JSON 직렬화 가능하게 변환"""
                import numpy as np
                
                if obj is None:
                    return None
                elif isinstance(obj, (bool, str)):
                    return obj
                elif isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_safe(item) for item in obj]
                else:
                    # 알 수 없는 타입은 문자열로 변환
                    return str(obj)
            
            # 안전한 메타데이터 생성
            metadata = {
                'window_size': int(self.window_size),
                'stride': int(self.stride), 
                'overlap_threshold': float(self.overlap_threshold),
                'model_architecture': 'lstm_v2',
                'training_date': datetime.now().isoformat(),
                'model_format': 'keras_native' if model_path.endswith('.keras') else 'h5'
            }
            
            # training_stats가 있으면 안전하게 추가
            if hasattr(self, 'training_stats') and self.training_stats:
                metadata['training_stats'] = make_json_safe(self.training_stats)
            
            # JSON 저장
            metadata_path = model_path.replace('.keras', '_metadata.json').replace('.h5', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info("✅ 저장 완료")
            
            # 저장된 파일들 정보
            saved_files = {
                'model': model_path,
                'scaler': scaler_path,
                'metadata': metadata_path
            }
            
            if selector_path:
                saved_files['selector'] = selector_path
            
            # 파일 크기 정보
            for file_type, file_path in saved_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    self.logger.info(f"   📄 {file_type}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ 모델 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 최소한의 저장 시도
            try:
                # 기본 H5 형식으로 모델만 저장
                basic_model_path = model_path.replace('.keras', '.h5')
                self.model.save(basic_model_path)
                self.logger.info(f"   📦 기본 모델 저장 완료: {basic_model_path}")
                
                return {'model': basic_model_path}
                
            except Exception as e2:
                self.logger.error(f"❌ 기본 저장도 실패: {e2}")
                return {}
    
        
    def auto_detect_files(self):
        """사용 가능한 파일과 디렉토리 자동 탐지"""
        print("🔍 사용 가능한 파일/디렉토리 자동 탐지...")
        
        # 현재 디렉토리 파일들 확인
        current_files = os.listdir('.')
        print(f"📁 현재 디렉토리 파일들: {len(current_files)}개")
        
        # CSV 파일들 찾기
        csv_files = [f for f in current_files if f.endswith('.csv')]
        if csv_files:
            print(f"   📄 발견된 CSV 파일들:")
            for i, csv_file in enumerate(csv_files[:5]):  # 최대 5개만 표시
                file_size = os.path.getsize(csv_file) / 1024  # KB
                print(f"      {i+1}. {csv_file} ({file_size:.1f} KB)")
            if len(csv_files) > 5:
                print(f"      ... 외 {len(csv_files)-5}개")
        
        # 디렉토리들 확인
        dirs = [d for d in current_files if os.path.isdir(d)]
        if dirs:
            print(f"   📂 발견된 디렉토리들:")
            for i, dir_name in enumerate(dirs[:5]):
                csv_count = len(glob.glob(os.path.join(dir_name, "**", "*.csv"), recursive=True))
                print(f"      {i+1}. {dir_name}/ (CSV: {csv_count}개)")
        
        # 상위 디렉토리 확인
        parent_dir = '..'
        if os.path.exists(parent_dir):
            parent_files = os.listdir(parent_dir)
            parent_dirs = [d for d in parent_files if os.path.isdir(os.path.join(parent_dir, d))]
            if parent_dirs:
                print(f"   📂 상위 디렉토리들:")
                for dir_name in parent_dirs[:5]:
                    full_path = os.path.join(parent_dir, dir_name)
                    csv_count = len(glob.glob(os.path.join(full_path, "**", "*.csv"), recursive=True))
                    if csv_count > 0:
                        print(f"      {dir_name}/ (CSV: {csv_count}개)")
        
        return csv_files, dirs
    
    def find_best_test_file(self):
        """가장 적합한 테스트 파일 찾기"""
        # 1. 미리 정의된 파일들 확인
        for test_file in POSSIBLE_TEST_FILES:
            if os.path.exists(test_file):
                return test_file
        
        # 2. 현재 디렉토리의 CSV 파일들 확인
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            # 가장 큰 파일 선택 (데이터가 많을 가능성)
            csv_files_with_size = [(f, os.path.getsize(f)) for f in csv_files]
            largest_file = max(csv_files_with_size, key=lambda x: x[1])[0]
            return largest_file
        
        return None
    
    def find_best_data_dir(self):
        """가장 적합한 데이터 디렉토리 찾기"""
        # 1. 미리 정의된 디렉토리들 확인
        for data_dir in POSSIBLE_DATA_DIRS:
            if os.path.exists(data_dir):
                csv_count = len(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
                if csv_count > 0:
                    return data_dir
        
        # 2. 현재 디렉토리의 폴더들 확인
        dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        for dir_name in dirs:
            csv_count = len(glob.glob(os.path.join(dir_name, "**", "*.csv"), recursive=True))
            if csv_count > 0:
                return dir_name
        
        # 3. 상위 디렉토리 확인
        parent_dirs = [d for d in os.listdir('..') if os.path.isdir(os.path.join('..', d))]
        for dir_name in parent_dirs:
            full_path = os.path.join('..', dir_name)
            csv_count = len(glob.glob(os.path.join(full_path, "**", "*.csv"), recursive=True))
            if csv_count > 0:
                return full_path
        
        return None

        return {
            'model': model_path,
            'scaler': scaler_path,
            'selector': selector_path if self.feature_selector else None,
            'metadata': metadata_path
        }

# 테스트 실행 함수
def test_trainer(test_csv=None, data_dir=None):
    """학습 클래스 테스트"""
    print("🧪 MultiFileCSITrainer 테스트")
    print("=" * 40)
    
    # 1. 트레이너 초기화
    trainer = MultiFileCSITrainer(
        window_size=DEFAULT_WINDOW_SIZE,
        stride=DEFAULT_STRIDE,
        overlap_threshold=DEFAULT_OVERLAP_THRESHOLD
    )
    
    # 2. 파일/디렉토리 자동 탐지
    if test_csv is None and data_dir is None:
        print("🔍 자동 탐지 모드...")
        trainer.auto_detect_files()
        
        # 최적 파일/디렉토리 찾기
        test_csv = trainer.find_best_test_file()
        data_dir = trainer.find_best_data_dir()
        
        print(f"\n🎯 자동 탐지 결과:")
        print(f"   📄 추천 테스트 파일: {test_csv}")
        print(f"   📂 추천 데이터 디렉토리: {data_dir}")
    
    # 기본값 설정
    if test_csv is None:
        test_csv = DEFAULT_TEST_CSV
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    print(f"\n📋 최종 설정:")
    print(f"   📄 테스트 파일: {test_csv}")
    print(f"   📂 데이터 디렉토리: {data_dir}")
    
    # 파일 존재 여부 상세 확인
    print(f"\n🔍 파일 존재 여부 확인:")
    test_csv_exists = os.path.exists(test_csv)
    data_dir_exists = os.path.exists(data_dir)
    
    print(f"   📄 {test_csv}: {'✅ 존재' if test_csv_exists else '❌ 없음'}")
    if test_csv_exists:
        file_size = os.path.getsize(test_csv) / 1024
        print(f"      크기: {file_size:.1f} KB")
    
    print(f"   📂 {data_dir}: {'✅ 존재' if data_dir_exists else '❌ 없음'}")
    if data_dir_exists:
        csv_count = len(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
        print(f"      CSV 파일: {csv_count}개")
        
        # 몇 개 파일 예시 보여주기
        if csv_count > 0:
            csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
            print(f"      예시 파일들:")
            for i, csv_file in enumerate(csv_files[:3]):
                rel_path = os.path.relpath(csv_file, data_dir)
                file_size = os.path.getsize(csv_file) / 1024
                print(f"         {i+1}. {rel_path} ({file_size:.1f} KB)")
            if csv_count > 3:
                print(f"         ... 외 {csv_count-3}개")
    
    # 3. 단일 파일 테스트
    if test_csv_exists:
        print(f"\n📄 단일 파일 테스트: {test_csv}")
        trainer.analyze_single_file(test_csv)
        
        try:
            X, y, file_info = trainer.load_multiple_csv_files(test_csv)
            
            print(f"✅ 데이터 로드 성공: {X.shape}")
            
            # 특징 선택
            X_selected = trainer.smart_feature_selection(X, y, target_features=DEFAULT_TARGET_FEATURES)
            
            # 시퀀스 생성
            X_seq, y_seq = trainer.create_sequences(X_selected, y)
            
            # 전처리
            X_processed = trainer.preprocess_data(X_seq)
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_seq, test_size=0.2, random_state=42
            )
            
            print(f"✅ 전처리 완료")
            print(f"   훈련: {X_train.shape}, 테스트: {X_test.shape}")
            
            # 모델 구축
            input_shape = (X_train.shape[1], X_train.shape[2])
            trainer.build_model(input_shape)
            
            # 간단한 학습 (테스트용)
            print(f"\n🚀 테스트 학습 ({DEFAULT_EPOCHS} 에포크)...")
            history = trainer.train_model(X_train, y_train, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
            
            # 평가
            performance = trainer.evaluate_model(X_test, y_test)
            
            # 저장
            timestamp = datetime.now().strftime('%H%M%S')
            model_path = f"test_model_{timestamp}.h5"
            saved_files = trainer.save_model(model_path)
            
            print(f"\n✅ 단일 파일 테스트 완료!")
            print(f"   성능: {performance}")
            print(f"   저장된 파일: {list(saved_files.keys())}")
            
            return trainer, saved_files
            
        except Exception as e:
            print(f"❌ 단일 파일 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 다중 파일 테스트
    elif data_dir_exists:
        print(f"\n📂 다중 파일 테스트: {data_dir}")
        try:
            csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
            if len(csv_files) == 0:
                print(f"❌ {data_dir}에 CSV 파일이 없습니다.")
                return None, None
            
            print(f"   발견된 파일: {len(csv_files)}개")
            
            # 첫 번째 파일로 단일 분석
            first_file = csv_files[0]
            print(f"   첫 번째 파일 분석: {os.path.relpath(first_file, data_dir)}")
            trainer.analyze_single_file(first_file)
            
            # 다중 파일 학습 (최대 5개 파일로 제한)
            print(f"\n🎓 다중 파일 학습 테스트 (최대 5개 파일)...")
            X, y, file_info = trainer.load_multiple_csv_files(data_dir, max_files=5)
            
            print(f"✅ 다중 파일 로드 성공: {X.shape}")
            print(f"   사용된 파일: {len(file_info)}개")
            
            # 나머지 과정은 단일 파일과 동일
            X_selected = trainer.smart_feature_selection(X, y, target_features=DEFAULT_TARGET_FEATURES)
            X_seq, y_seq = trainer.create_sequences(X_selected, y)
            X_processed = trainer.preprocess_data(X_seq)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_seq, test_size=0.2, random_state=42
            )
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            trainer.build_model(input_shape)
            
            history = trainer.train_model(X_train, y_train, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
            performance = trainer.evaluate_model(X_test, y_test)
            
            timestamp = datetime.now().strftime('%H%M%S')
            model_path = f"multi_test_model_{timestamp}.h5"
            saved_files = trainer.save_model(model_path)
            
            print(f"\n✅ 다중 파일 테스트 완료!")
            print(f"   성능: {performance}")
            print(f"   저장된 파일: {list(saved_files.keys())}")
            
            return trainer, saved_files
            
        except Exception as e:
            print(f"❌ 다중 파일 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    else:
        print(f"\n❌ 사용 가능한 파일이나 디렉토리가 없습니다!")
        print(f"\n📋 해결 방법:")
        print(f"   1. CSV 파일을 현재 디렉토리에 복사")
        print(f"   2. 데이터 폴더를 생성하고 CSV 파일들 넣기")
        print(f"   3. 직접 경로 지정:")
        print(f"      test_trainer('실제파일.csv')")
        print(f"      test_trainer(data_dir='실제폴더경로')")
        
        # 도움말 정보
        print(f"\n💡 현재 디렉토리 정보:")
        print(f"   경로: {os.getcwd()}")
        current_files = os.listdir('.')
        csv_files = [f for f in current_files if f.endswith('.csv')]
        dirs = [d for d in current_files if os.path.isdir(d)]
        
        if csv_files:
            print(f"   CSV 파일들: {csv_files}")
        if dirs:
            print(f"   디렉토리들: {dirs}")
        
        return None, None

def quick_test():
    """빠른 테스트 (기본 설정)"""
    return test_trainer()

def custom_test(csv_file, window_size=DEFAULT_WINDOW_SIZE, target_features=DEFAULT_TARGET_FEATURES):
    """커스텀 설정으로 테스트"""
    print(f"🔧 커스텀 테스트: 윈도우={window_size}, 특징={target_features}")
    
    # 커스텀 트레이너
    trainer = MultiFileCSITrainer(
        window_size=window_size,
        stride=max(1, window_size // 10),  # 윈도우 크기에 비례한 스트라이드
        overlap_threshold=DEFAULT_OVERLAP_THRESHOLD
    )
    
    if os.path.exists(csv_file):
        trainer.analyze_single_file(csv_file)
        
        try:
            X, y, file_info = trainer.load_multiple_csv_files(csv_file)
            X_selected = trainer.smart_feature_selection(X, y, target_features=target_features)
            X_seq, y_seq = trainer.create_sequences(X_selected, y)
            X_processed = trainer.preprocess_data(X_seq)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_seq, test_size=0.2, random_state=42
            )
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            trainer.build_model(input_shape)
            
            history = trainer.train_model(X_train, y_train, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE)
            performance = trainer.evaluate_model(X_test, y_test)
            
            timestamp = datetime.now().strftime('%H%M%S')
            model_path = f"custom_model_w{window_size}_f{target_features}_{timestamp}.h5"
            saved_files = trainer.save_model(model_path)
            
            print(f"\n✅ 커스텀 테스트 완료!")
            print(f"   성능: {performance}")
            
            return trainer, saved_files
            
        except Exception as e:
            print(f"❌ 커스텀 테스트 실패: {e}")
            return None, None
    else:
        print(f"❌ 파일이 없습니다: {csv_file}")
        return None, None

if __name__ == "__main__":
    import sys
    
    print("🚀 CSI 학습 클래스 테스트")
    print("=" * 50)
    print("📋 사용 가능한 옵션:")
    print("   1. 기본 테스트: python multi_file_trainer.py")
    print("   2. 커스텀 파일: python multi_file_trainer.py your_file.csv")
    print("   3. 디렉토리: python multi_file_trainer.py --dir ./your_data")
    print("")
    
    # 명령행 인자 처리
    if len(sys.argv) == 1:
        # 기본 테스트
        print("🔄 기본 테스트 실행...")
        test_trainer()
        
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg.endswith('.csv'):
            # CSV 파일 지정
            print(f"📄 지정된 CSV 파일로 테스트: {arg}")
            test_trainer(test_csv=arg)
        else:
            # 디렉토리 지정
            print(f"📂 지정된 디렉토리로 테스트: {arg}")
            test_trainer(data_dir=arg)
            
    elif len(sys.argv) == 3 and sys.argv[1] == '--dir':
        # --dir 옵션
        data_dir = sys.argv[2]
        print(f"📂 디렉토리 테스트: {data_dir}")
        test_trainer(data_dir=data_dir)
        
    else:
        print("❌ 잘못된 인자입니다.")
        print("💡 사용법:")
        print("   python multi_file_trainer.py")
        print("   python multi_file_trainer.py case32.csv")
        print("   python multi_file_trainer.py --dir ./csi_data")