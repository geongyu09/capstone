import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class CSIFallDetection:
    def __init__(self, window_size=50, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self.model = None
        
    def load_csv_files(self, data_directory):
        """
        여러 CSV 파일을 로드하여 하나의 데이터셋으로 합치기
        
        Args:
            data_directory: CSV 파일들이 있는 디렉토리 경로
        
        Returns:
            X: 특징 데이터 (n_samples, n_features)
            y: 라벨 데이터 (n_samples,)
            timestamps: 타임스탬프 (n_samples,)
        """
        all_data = []
        csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
        
        print(f"발견된 CSV 파일 수: {len(csv_files)}")
        
        for file_path in csv_files:
            print(f"로딩 중: {os.path.basename(file_path)}")
            
            try:
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                
                # 컬럼 확인
                if 'timestamp' not in df.columns or 'label' not in df.columns:
                    print(f"경고: {file_path}에 필수 컬럼이 없습니다.")
                    continue
                
                # 특징 컬럼 추출 (feat_0 ~ feat_63 또는 feat_255)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                if len(feature_cols) == 0:
                    print(f"경고: {file_path}에 특징 컬럼이 없습니다.")
                    continue
                
                # 데이터 추가
                file_data = {
                    'features': df[feature_cols].values,
                    'labels': df['label'].values,
                    'timestamps': pd.to_datetime(df['timestamp']).values,
                    'filename': os.path.basename(file_path)
                }
                all_data.append(file_data)
                
                print(f"  - 형태: {file_data['features'].shape}")
                print(f"  - 라벨 분포: {np.bincount(file_data['labels'])}")
                
            except Exception as e:
                print(f"오류: {file_path} 로딩 실패 - {e}")
        
        if not all_data:
            raise ValueError("로드된 데이터가 없습니다!")
        
        # 모든 데이터 합치기
        X = np.vstack([data['features'] for data in all_data])
        y = np.hstack([data['labels'] for data in all_data])
        timestamps = np.hstack([data['timestamps'] for data in all_data])
        
        print(f"\n전체 데이터 통계:")
        print(f"- 총 샘플 수: {X.shape[0]}")
        print(f"- 특징 수: {X.shape[1]}")
        print(f"- 라벨 분포: {np.bincount(y)}")
        
        return X, y, timestamps
    
    def load_csv_files_by_label(self, fall_dir, normal_dir):
        """
        라벨별로 분리된 폴더에서 CSV 파일 로드
        
        Args:
            fall_dir: 낙상 데이터 폴더 경로
            normal_dir: 정상 데이터 폴더 경로
        
        Returns:
            X: 특징 데이터 (n_samples, n_features)
            y: 라벨 데이터 (n_samples,)
            timestamps: 타임스탬프 (n_samples,)
        """
        all_data = []
        
        # 낙상 데이터 로드 (label=1)
        fall_files = glob.glob(os.path.join(fall_dir, "*.csv"))
        print(f"낙상 데이터 파일 수: {len(fall_files)}")
        
        for file_path in fall_files:
            print(f"낙상 데이터 로딩: {os.path.basename(file_path)}")
            try:
                df = pd.read_csv(file_path)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                file_data = {
                    'features': df[feature_cols].values,
                    'labels': np.ones(len(df)),  # 모두 1로 설정
                    'timestamps': pd.to_datetime(df['timestamp']).values,
                    'filename': os.path.basename(file_path)
                }
                all_data.append(file_data)
                print(f"  - 형태: {file_data['features'].shape}, 라벨: 낙상")
                
            except Exception as e:
                print(f"오류: {file_path} 로딩 실패 - {e}")
        
        # 정상 데이터 로드 (label=0)
        normal_files = glob.glob(os.path.join(normal_dir, "*.csv"))
        print(f"정상 데이터 파일 수: {len(normal_files)}")
        
        for file_path in normal_files:
            print(f"정상 데이터 로딩: {os.path.basename(file_path)}")
            try:
                df = pd.read_csv(file_path)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                file_data = {
                    'features': df[feature_cols].values,
                    'labels': np.zeros(len(df)),  # 모두 0으로 설정
                    'timestamps': pd.to_datetime(df['timestamp']).values,
                    'filename': os.path.basename(file_path)
                }
                all_data.append(file_data)
                print(f"  - 형태: {file_data['features'].shape}, 라벨: 정상")
                
            except Exception as e:
                print(f"오류: {file_path} 로딩 실패 - {e}")
        
        if not all_data:
            raise ValueError("로드된 데이터가 없습니다!")
        
        # 모든 데이터 합치기
        X = np.vstack([data['features'] for data in all_data])
        y = np.hstack([data['labels'] for data in all_data])
        timestamps = np.hstack([data['timestamps'] for data in all_data])
        
        print(f"\n전체 데이터 통계:")
        print(f"- 총 샘플 수: {X.shape[0]}")
        print(f"- 특징 수: {X.shape[1]}")
        print(f"- 라벨 분포: 정상={np.sum(y==0)}, 낙상={np.sum(y==1)}")
        
        return X, y, timestamps
    
    def create_sequences(self, X, y):
        """
        시계열 데이터를 LSTM 입력 형태로 변환 (슬라이딩 윈도우)
        
        Args:
            X: 특징 데이터 (n_samples, n_features)
            y: 라벨 데이터 (n_samples,)
        
        Returns:
            X_seq: 시퀀스 데이터 (n_sequences, window_size, n_features)
            y_seq: 시퀀스 라벨 (n_sequences,)
        """
        X_sequences = []
        y_sequences = []
        
        print(f"슬라이딩 윈도우 적용: window_size={self.window_size}, stride={self.stride}")
        
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            # 윈도우 추출
            window_X = X[i:i + self.window_size]
            window_y = y[i:i + self.window_size]
            
            # 윈도우 내 라벨 결정 (다수결 또는 최대값)
            # 낙상 감지에서는 보통 하나라도 낙상이면 낙상으로 분류
            sequence_label = 1 if np.any(window_y == 1) else 0
            
            X_sequences.append(window_X)
            y_sequences.append(sequence_label)
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        print(f"생성된 시퀀스:")
        print(f"- 시퀀스 수: {X_seq.shape[0]}")
        print(f"- 시퀀스 형태: {X_seq.shape}")
        print(f"- 라벨 분포: {np.bincount(y_seq)}")
        
        return X_seq, y_seq
    
    def preprocess_data(self, X, feature_selection=True):
        """데이터 전처리 (정규화 + 특징 선택)"""
        print("데이터 전처리 중...")
        
        # 원본 형태 저장
        original_shape = X.shape
        print(f"원본 형태: {original_shape}")
        
        # 2D로 변환하여 처리
        X_2d = X.reshape(-1, X.shape[-1])
        
        # 특징 선택 (256개가 너무 많을 경우)
        if feature_selection and X.shape[-1] > 128:
            print(f"특징 선택 수행: {X.shape[-1]}개 → 128개")
            
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            
            # 1. 분산이 낮은 특징 제거
            var_selector = VarianceThreshold(threshold=0.01)
            X_var = var_selector.fit_transform(X_2d)
            
            # 2. 상위 128개 특징 선택
            if X_var.shape[1] > 128:
                # 임시 라벨 생성 (실제로는 y 라벨 사용해야 함)
                temp_y = np.random.randint(0, 2, size=X_var.shape[0])
                k_selector = SelectKBest(f_classif, k=128)
                X_selected = k_selector.fit_transform(X_var, temp_y)
                
                self.var_selector = var_selector
                self.k_selector = k_selector
            else:
                X_selected = X_var
                self.var_selector = var_selector
                self.k_selector = None
        else:
            X_selected = X_2d
            self.var_selector = None
            self.k_selector = None
        
        # 정규화
        X_normalized = self.scaler.fit_transform(X_selected)
        
        # 원본 형태로 복원 (특징 수는 변경될 수 있음)
        new_shape = (original_shape[0], original_shape[1], X_normalized.shape[1])
        X_normalized = X_normalized.reshape(new_shape)
        
        print(f"전처리 후 형태: {X_normalized.shape}")
        return X_normalized
    
    def build_model(self, input_shape, model_type='standard'):
        """LSTM 모델 구축 - 256개 특징에 최적화"""
        print(f"모델 구축 중... 입력 형태: {input_shape}")
        print(f"특징 수: {input_shape[1]}개")
        
        if model_type == 'lightweight':
            # 경량 모델 (256개 특징이 많을 때)
            model = Sequential([
                # 특징 차원 축소를 위한 Dense 층
                Dense(64, activation='relu', input_shape=input_shape),
                Dropout(0.3),
                
                # LSTM 층들
                LSTM(32, return_sequences=True),
                Dropout(0.3),
                
                LSTM(16, return_sequences=False),
                Dropout(0.3),
                
                # 출력층
                Dense(8, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
        elif model_type == 'cnn_lstm':
            # CNN + LSTM 하이브리드 모델
            model = Sequential([
                # CNN 층으로 지역적 패턴 추출
                Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
                Dropout(0.3),
                
                Conv1D(32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                
                # LSTM 층으로 시간적 패턴 학습
                LSTM(32, return_sequences=False),
                Dropout(0.3),
                
                # 출력층
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
        else:  # standard
            # 표준 모델 (더 큰 LSTM 유닛 사용)
            model = Sequential([
                # 첫 번째 LSTM 층 (유닛 수 증가)
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.4),
                
                # 두 번째 LSTM 층
                LSTM(64, return_sequences=False),
                Dropout(0.4),
                
                # 완전연결층
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(16, activation='relu'),
                Dropout(0.2),
                
                # 출력층 (이진 분류)
                Dense(1, activation='sigmoid')
            ])
        
        # 특징 수에 따른 학습률 조정
        learning_rate = 0.001 if input_shape[1] <= 64 else 0.0005
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("모델 구조:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """모델 학습"""
        print("모델 학습 시작...")
        
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
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # 검증 데이터 설정
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = 0.2
        
        # 학습
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """모델 평가"""
        print("모델 평가 중...")
        
        # 예측
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # 평가 지표 계산
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n분류 보고서:")
        print(classification_report(y_test, y_pred))
        
        print("\n혼동 행렬:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # 성능 지표
        accuracy = np.mean(y_pred == y_test)
        
        if np.sum(y_test) > 0:  # 실제 양성 샘플이 있는 경우
            recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
            precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-7)
            
            print(f"\n성능 지표:")
            print(f"정확도: {accuracy:.4f}")
            print(f"정밀도: {precision:.4f}")
            print(f"재현율: {recall:.4f}")
            print(f"F1-점수: {f1:.4f}")
        
        return y_pred_prob, y_pred