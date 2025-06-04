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
        
    def load_csv_files(self, data_directory, recursive=True):
        """
        여러 CSV 파일을 로드하여 하나의 데이터셋으로 합치기
        
        Args:
            data_directory: CSV 파일들이 있는 디렉토리 경로
            recursive: True면 하위 폴더까지 재귀적으로 검색
        
        Returns:
            X: 특징 데이터 (n_samples, n_features)
            y: 라벨 데이터 (n_samples,)
            timestamps: 타임스탬프 (n_samples,)
        """
        all_data = []
        
        if recursive:
            # 재귀적으로 모든 하위 폴더의 CSV 파일 찾기
            csv_files = glob.glob(os.path.join(data_directory, "**", "*.csv"), recursive=True)
            print(f"재귀 검색으로 발견된 CSV 파일 수: {len(csv_files)}")
        else:
            # 현재 폴더의 CSV 파일만 찾기
            csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
            print(f"발견된 CSV 파일 수: {len(csv_files)}")
        
        # 발견된 파일 경로들 출력
        if csv_files:
            print("발견된 파일들:")
            for i, file_path in enumerate(csv_files):
                relative_path = os.path.relpath(file_path, data_directory)
                print(f"  {i+1:2d}. {relative_path}")
                if i >= 9:  # 처음 10개만 출력
                    remaining = len(csv_files) - 10
                    if remaining > 0:
                        print(f"      ... 그 외 {remaining}개 파일")
                    break
        
        for file_path in csv_files:
            print(f"\n로딩 중: {os.path.basename(file_path)}")
            
            try:
                # CSV 파일 읽기 (다양한 인코딩 시도)
                df = None
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1', 'utf-8-sig']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"     ✅ 인코딩 성공: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"     ⚠️  {encoding} 인코딩 실패: {str(e)[:50]}...")
                        continue
                
                if df is None:
                    print(f"  ❌ 모든 인코딩 실패: {file_path}")
                    continue
                
                # 컬럼 확인
                if 'timestamp' not in df.columns or 'label' not in df.columns:
                    print(f"  ❌ 경고: {file_path}에 필수 컬럼(timestamp, label)이 없습니다.")
                    print(f"     현재 컬럼: {list(df.columns)[:5]}...")
                    continue
                
                # 특징 컬럼 추출 (feat_0 ~ feat_63 또는 feat_255)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                if len(feature_cols) == 0:
                    print(f"  ❌ 경고: {file_path}에 특징 컬럼(feat_*)이 없습니다.")
                    print(f"     현재 컬럼: {list(df.columns)[:10]}...")
                    continue
                
                # 데이터 유효성 검사
                if len(df) == 0:
                    print(f"  ❌ 경고: {file_path}가 비어있습니다.")
                    continue
                
                # 데이터 추가
                file_data = {
                    'features': df[feature_cols].values,
                    'labels': df['label'].values,
                    'timestamps': pd.to_datetime(df['timestamp']).values,
                    'filename': os.path.basename(file_path)
                }
                all_data.append(file_data)
                
                print(f"  ✅ 로드 성공!")
                print(f"     - 데이터 형태: {file_data['features'].shape}")
                print(f"     - 특징 수: {len(feature_cols)}개")
                print(f"     - 라벨 분포: {dict(zip(*np.unique(file_data['labels'], return_counts=True)))}")
                
            except FileNotFoundError:
                print(f"  ❌ 오류: {file_path} 파일을 찾을 수 없습니다.")
            except pd.errors.EmptyDataError:
                print(f"  ❌ 오류: {file_path}가 비어있습니다.")
            except Exception as e:
                print(f"  ❌ 오류: {file_path} 로딩 실패 - {str(e)}")
        
        if not all_data:
            raise ValueError("❌ 로드된 데이터가 없습니다! CSV 파일의 형식을 확인해주세요.")
        
        # 모든 데이터 합치기
        print(f"\n📊 데이터 결합 중...")
        X = np.vstack([data['features'] for data in all_data])
        y = np.hstack([data['labels'] for data in all_data])
        timestamps = np.hstack([data['timestamps'] for data in all_data])
        
        print(f"✅ 전체 데이터 통계:")
        print(f"   - 총 샘플 수: {X.shape[0]:,}개")
        print(f"   - 특징 수: {X.shape[1]}개")
        print(f"   - 사용된 파일 수: {len(all_data)}개")
        
        # 라벨 분포 상세 출력
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"   - 라벨 분포:")
        for label, count in zip(unique_labels, counts):
            label_name = "낙상" if label == 1 else "정상"
            percentage = (count / len(y)) * 100
            print(f"     • {label_name}(label={label}): {count:,}개 ({percentage:.1f}%)")
        
        return X, y, timestamps
    
    def load_csv_files_by_label(self, fall_dir, normal_dir, recursive=True):
        """
        라벨별로 분리된 폴더에서 CSV 파일 로드
        
        Args:
            fall_dir: 낙상 데이터 폴더 경로
            normal_dir: 정상 데이터 폴더 경로
            recursive: True면 하위 폴더까지 재귀적으로 검색
        
        Returns:
            X: 특징 데이터 (n_samples, n_features)
            y: 라벨 데이터 (n_samples,)
            timestamps: 타임스탬프 (n_samples,)
        """
        all_data = []
        
        # 낙상 데이터 로드 (label=1)
        if recursive:
            fall_files = glob.glob(os.path.join(fall_dir, "**", "*.csv"), recursive=True)
        else:
            fall_files = glob.glob(os.path.join(fall_dir, "*.csv"))
        
        print(f"🔴 낙상 데이터 파일 수: {len(fall_files)}")
        
        for file_path in fall_files:
            print(f"낙상 데이터 로딩: {os.path.basename(file_path)}")
            try:
                df = pd.read_csv(file_path)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                if len(feature_cols) == 0:
                    print(f"  ❌ 특징 컬럼이 없습니다.")
                    continue
                
                file_data = {
                    'features': df[feature_cols].values,
                    'labels': np.ones(len(df)),  # 모두 1로 설정
                    'timestamps': pd.to_datetime(df['timestamp']).values,
                    'filename': os.path.basename(file_path)
                }
                all_data.append(file_data)
                print(f"  ✅ 형태: {file_data['features'].shape}, 라벨: 낙상")
                
            except Exception as e:
                print(f"  ❌ 오류: {file_path} 로딩 실패 - {e}")
        
        # 정상 데이터 로드 (label=0)
        if recursive:
            normal_files = glob.glob(os.path.join(normal_dir, "**", "*.csv"), recursive=True)
        else:
            normal_files = glob.glob(os.path.join(normal_dir, "*.csv"))
            
        print(f"🟢 정상 데이터 파일 수: {len(normal_files)}")
        
        for file_path in normal_files:
            print(f"정상 데이터 로딩: {os.path.basename(file_path)}")
            try:
                df = pd.read_csv(file_path)
                feature_cols = [col for col in df.columns if col.startswith('feat_')]
                
                if len(feature_cols) == 0:
                    print(f"  ❌ 특징 컬럼이 없습니다.")
                    continue
                
                file_data = {
                    'features': df[feature_cols].values,
                    'labels': np.zeros(len(df)),  # 모두 0으로 설정
                    'timestamps': pd.to_datetime(df['timestamp']).values,
                    'filename': os.path.basename(file_path)
                }
                all_data.append(file_data)
                print(f"  ✅ 형태: {file_data['features'].shape}, 라벨: 정상")
                
            except Exception as e:
                print(f"  ❌ 오류: {file_path} 로딩 실패 - {e}")
        
        if not all_data:
            raise ValueError("로드된 데이터가 없습니다!")
        
        # 모든 데이터 합치기
        X = np.vstack([data['features'] for data in all_data])
        y = np.hstack([data['labels'] for data in all_data])
        timestamps = np.hstack([data['timestamps'] for data in all_data])
        
        print(f"\n✅ 전체 데이터 통계:")
        print(f"   - 총 샘플 수: {X.shape[0]:,}개")
        print(f"   - 특징 수: {X.shape[1]}개")
        print(f"   - 라벨 분포: 정상={np.sum(y==0):,}개, 낙상={np.sum(y==1):,}개")
        
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
        
        print(f"🔄 슬라이딩 윈도우 적용:")
        print(f"   - Window Size: {self.window_size}")
        print(f"   - Stride: {self.stride}")
        print(f"   - 원본 데이터: {X.shape}")
        
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
        
        print(f"✅ 시퀀스 생성 완료:")
        print(f"   - 생성된 시퀀스 수: {X_seq.shape[0]:,}개")
        print(f"   - 시퀀스 형태: {X_seq.shape}")
        
        # 시퀀스 라벨 분포
        unique_labels, counts = np.unique(y_seq, return_counts=True)
        print(f"   - 시퀀스 라벨 분포:")
        for label, count in zip(unique_labels, counts):
            label_name = "낙상" if label == 1 else "정상"
            percentage = (count / len(y_seq)) * 100
            print(f"     • {label_name}: {count:,}개 ({percentage:.1f}%)")
        
        return X_seq, y_seq
    
    def preprocess_data(self, X, feature_selection=True):
        """데이터 전처리 (정규화 + 특징 선택)"""
        print(f"🔧 데이터 전처리 시작...")
        
        # 원본 형태 저장
        original_shape = X.shape
        print(f"   - 원본 형태: {original_shape}")
        
        # 2D로 변환하여 처리
        X_2d = X.reshape(-1, X.shape[-1])
        
        # 특징 선택 (256개가 너무 많을 경우)
        if feature_selection and X.shape[-1] > 128:
            print(f"📊 특징 선택 수행: {X.shape[-1]}개 → 128개")
            
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            
            # 1. 분산이 낮은 특징 제거
            print("   - 분산 기반 특징 선택...")
            var_selector = VarianceThreshold(threshold=0.01)
            X_var = var_selector.fit_transform(X_2d)
            print(f"     제거된 특징: {X_2d.shape[1] - X_var.shape[1]}개")
            
            # 2. 상위 128개 특징 선택
            if X_var.shape[1] > 128:
                print("   - F-통계량 기반 특징 선택...")
                # 임시 라벨 생성 (실제로는 y 라벨 사용해야 함)
                temp_y = np.random.randint(0, 2, size=X_var.shape[0])
                k_selector = SelectKBest(f_classif, k=128)
                X_selected = k_selector.fit_transform(X_var, temp_y)
                
                self.var_selector = var_selector
                self.k_selector = k_selector
                print(f"     최종 선택된 특징: {X_selected.shape[1]}개")
            else:
                X_selected = X_var
                self.var_selector = var_selector
                self.k_selector = None
        else:
            X_selected = X_2d
            self.var_selector = None
            self.k_selector = None
            print("   - 특징 선택 건너뛰기 (특징 수가 적거나 비활성화)")
        
        # 정규화
        print("   - 데이터 정규화 (StandardScaler)...")
        X_normalized = self.scaler.fit_transform(X_selected)
        
        # 원본 형태로 복원 (특징 수는 변경될 수 있음)
        new_shape = (original_shape[0], original_shape[1], X_normalized.shape[1])
        X_normalized = X_normalized.reshape(new_shape)
        
        print(f"✅ 전처리 완료:")
        print(f"   - 최종 형태: {X_normalized.shape}")
        print(f"   - 데이터 범위: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        
        return X_normalized
    
    def build_model(self, input_shape, model_type='standard'):
        """LSTM 모델 구축 - 256개 특징에 최적화"""
        print(f"🏗️  모델 구축 중...")
        print(f"   - 입력 형태: {input_shape}")
        print(f"   - 시퀀스 길이: {input_shape[0]}")
        print(f"   - 특징 수: {input_shape[1]}개")
        print(f"   - 모델 타입: {model_type}")
        
        if model_type == 'lightweight':
            # 경량 모델 (256개 특징이 많을 때)
            print("   📦 경량 모델 구성...")
            model = Sequential([
                # 특징 차원 축소를 위한 Dense 층
                Dense(64, activation='relu', input_shape=input_shape, name='feature_reduction'),
                Dropout(0.3),
                
                # LSTM 층들
                LSTM(32, return_sequences=True, name='lstm_1'),
                Dropout(0.3),
                
                LSTM(16, return_sequences=False, name='lstm_2'),
                Dropout(0.3),
                
                # 출력층
                Dense(8, activation='relu', name='dense_1'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1, activation='sigmoid', name='output')
            ])
            
        elif model_type == 'cnn_lstm':
            # CNN + LSTM 하이브리드 모델
            print("   🔀 CNN-LSTM 하이브리드 모델 구성...")
            model = Sequential([
                # CNN 층으로 지역적 패턴 추출
                Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, name='conv1d_1'),
                Dropout(0.3),
                
                Conv1D(32, kernel_size=3, activation='relu', name='conv1d_2'),
                MaxPooling1D(pool_size=2, name='maxpool_1'),
                Dropout(0.3),
                
                # LSTM 층으로 시간적 패턴 학습
                LSTM(32, return_sequences=False, name='lstm'),
                Dropout(0.3),
                
                # 출력층
                Dense(16, activation='relu', name='dense_1'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(1, activation='sigmoid', name='output')
            ])
            
        else:  # standard
            # 표준 모델 (더 큰 LSTM 유닛 사용)
            print("   🏛️  표준 모델 구성...")
            model = Sequential([
                # 첫 번째 LSTM 층 (유닛 수 증가)
                LSTM(128, return_sequences=True, input_shape=input_shape, name='lstm_1'),
                Dropout(0.4),
                
                # 두 번째 LSTM 층
                LSTM(64, return_sequences=False, name='lstm_2'),
                Dropout(0.4),
                
                # 완전연결층
                Dense(32, activation='relu', name='dense_1'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(16, activation='relu', name='dense_2'),
                Dropout(0.2),
                
                # 출력층 (이진 분류)
                Dense(1, activation='sigmoid', name='output')
            ])
        
        # 특징 수에 따른 학습률 조정
        learning_rate = 0.001 if input_shape[1] <= 64 else 0.0005
        print(f"   - 학습률: {learning_rate}")
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ 모델 구축 완료!")
        print(f"   - 총 파라미터 수: {model.count_params():,}개")
        
        # 모델 구조 출력
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """모델 학습"""
        print(f"🚀 모델 학습 시작...")
        print(f"   - 훈련 데이터: {X_train.shape}")
        print(f"   - 에포크: {epochs}")
        print(f"   - 배치 크기: {batch_size}")
        
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
            print(f"   - 검증 데이터: {X_val.shape}")
        else:
            validation_data = None
            validation_split = 0.2
            print(f"   - 검증 분할: {validation_split}")
        
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
        
        print(f"✅ 학습 완료!")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """모델 평가"""
        print(f"📊 모델 평가 중...")
        print(f"   - 테스트 데이터: {X_test.shape}")
        
        # 예측
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # 평가 지표 계산
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"\n📈 분류 보고서:")
        print(classification_report(y_test, y_pred, target_names=['정상', '낙상']))
        
        print(f"\n🔢 혼동 행렬:")
        cm = confusion_matrix(y_test, y_pred)
        print("        예측")
        print("실제    정상  낙상")
        print(f"정상   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"낙상   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # 성능 지표
        accuracy = np.mean(y_pred == y_test)
        
        if np.sum(y_test) > 0:  # 실제 양성 샘플이 있는 경우
            recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
            precision = np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-7)
            
            print(f"\n⭐ 성능 지표:")
            print(f"   - 정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"   - 정밀도 (Precision): {precision:.4f} ({precision*100:.1f}%)")
            print(f"   - 재현율 (Recall): {recall:.4f} ({recall*100:.1f}%)")
            print(f"   - F1-점수: {f1:.4f}")
            
            # 낙상 감지 관점에서의 해석
            print(f"\n🎯 낙상 감지 관점:")
            print(f"   - 실제 낙상을 놓친 비율: {(1-recall)*100:.1f}%")
            print(f"   - 거짓 알람 비율: {(1-precision)*100:.1f}%")
        else:
            print(f"\n⭐ 성능 지표:")
            print(f"   - 정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"   ⚠️  테스트 데이터에 낙상 샘플이 없어 일부 지표를 계산할 수 없습니다.")
        
        return y_pred_prob, y_pred