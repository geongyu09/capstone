def load_and_preprocess_csv(self, csv_path, window_size=50, stride=1):
        """CSV 파일 로드 및 전처리"""
        print(f"📁 데이터 로드: {csv_path}")
        
        # CSV 로드
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # 특징 컬럼 추출
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
        X = df[feature_cols].values
        y = df['label'].values if 'label' in df.columns else None
        
        print(f"   원본 데이터: {X.shape}")
        
        # 슬라이딩 윈도우 적용
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(X) - window_size + 1, stride):
            window_X = X[i:i + window_size]
            X_sequences.append(window_X)
            
            if y is not None:
                window_y = y[i:i + window_size]
                sequence_label = 1 if np.any(window_y == 1) else 0
                y_sequences.append(sequence_label)
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences) if y is not None else None
        
        # 특징 수에 따른 정규화
        original_shape = X_seq.shape
        X_2d = X_seq.reshape(-1, X_seq.shape[-1])
        
        # 256개 특징이면 128개로 축소 (모델과 맞추기)
        if X_seq.shape[-1] > 128:
            print(f"   📊 특징 축소: {X_seq.shape[-1]}개 → 128개")
            
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
            
            # 1. 분산 기반 특징 제거
            var_selector = VarianceThreshold(threshold=0.01)
            X_# 학습된 모델로 새로운 데이터 테스트
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTester:
    def __init__(self, model_path):
        """모델 테스터 초기화"""
        try:
            # 호환성 모드로 모델 로드
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # 모델 재컴파일
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ 모델 로드 완료: {model_path}")
            print(f"📊 모델 입력 형태: {self.model.input_shape}")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("💡 해결 방법:")
            print("   1. python run_training.py 로 모델을 다시 학습하세요")
            print("   2. 또는 다른 모델 파일을 사용하세요")
            raise
        
        self.scaler = StandardScaler()
    
    def load_and_preprocess_csv(self, csv_path, window_size=50, stride=1):
        """CSV 파일 로드 및 전처리"""
        print(f"📁 데이터 로드: {csv_path}")
        
        # CSV 로드 (인코딩 문제 대응)
        df = None
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"   ✅ 인코딩 성공: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"모든 인코딩 실패: {csv_path}")
        
        # 특징 컬럼 추출
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
        X = df[feature_cols].values
        y = df['label'].values if 'label' in df.columns else None
        
        print(f"   원본 데이터: {X.shape}")
        print(f"   원본 특징 수: {len(feature_cols)}개")
        
        # 슬라이딩 윈도우 적용
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(X) - window_size + 1, stride):
            window_X = X[i:i + window_size]
            X_sequences.append(window_X)
            
            if y is not None:
                window_y = y[i:i + window_size]
                sequence_label = 1 if np.any(window_y == 1) else 0
                y_sequences.append(sequence_label)
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences) if y is not None else None
        
        print(f"   윈도우 적용 후: {X_seq.shape}")
        
        # 모델 입력 차원과 맞추기
        model_input_features = self.model.input_shape[-1]  # 128 또는 256
        current_features = X_seq.shape[-1]
        
        print(f"   모델 기대 특징 수: {model_input_features}")
        print(f"   현재 데이터 특징 수: {current_features}")
        
        # 차원 조정
        if current_features != model_input_features:
            print(f"   🔧 특징 수 조정: {current_features} → {model_input_features}")
            
            if current_features > model_input_features:
                # 특징 수 줄이기 (256 → 128)
                print(f"   📊 특징 선택 수행...")
                
                original_shape = X_seq.shape
                X_2d = X_seq.reshape(-1, X_seq.shape[-1])
                
                # 1. 분산 기반 특징 제거
                from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
                
                var_selector = VarianceThreshold(threshold=0.01)
                X_var = var_selector.fit_transform(X_2d)
                print(f"      분산 제거 후: {X_var.shape[1]}개")
                
                # 2. 상위 특징 선택
                if X_var.shape[1] > model_input_features:
                    temp_y = np.random.randint(0, 2, size=X_var.shape[0])
                    k_selector = SelectKBest(f_classif, k=model_input_features)
                    X_selected = k_selector.fit_transform(X_var, temp_y)
                    print(f"      최종 선택: {X_selected.shape[1]}개")
                else:
                    X_selected = X_var
                
                # 정규화
                X_normalized = self.scaler.fit_transform(X_selected)
                
                # 원본 형태로 복원
                new_shape = (original_shape[0], original_shape[1], X_normalized.shape[1])
                X_seq = X_normalized.reshape(new_shape)
                
            elif current_features < model_input_features:
                # 특징 수 늘리기 (패딩)
                print(f"   📊 특징 패딩 수행...")
                padding_size = model_input_features - current_features
                
                # 제로 패딩 추가
                padding = np.zeros((X_seq.shape[0], X_seq.shape[1], padding_size))
                X_seq = np.concatenate([X_seq, padding], axis=-1)
                
                # 정규화
                original_shape = X_seq.shape
                X_2d = X_seq.reshape(-1, X_seq.shape[-1])
                X_normalized = self.scaler.fit_transform(X_2d)
                X_seq = X_normalized.reshape(original_shape)
        else:
            # 차원이 같으면 정규화만
            original_shape = X_seq.shape
            X_2d = X_seq.reshape(-1, X_seq.shape[-1])
            X_normalized = self.scaler.fit_transform(X_2d)
            X_seq = X_normalized.reshape(original_shape)
        
        print(f"   ✅ 최종 데이터: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def predict_single_file(self, csv_path, threshold=0.5):
        """단일 파일에 대한 예측"""
        print(f"\n🔍 파일 분석: {csv_path}")
        
        X_seq, y_true = self.load_and_preprocess_csv(csv_path)
        
        # 예측 수행
        predictions = self.model.predict(X_seq, verbose=0)
        y_pred_prob = predictions.flatten()
        y_pred = (y_pred_prob > threshold).astype(int)
        
        # 결과 분석
        fall_probability = np.mean(y_pred_prob)
        fall_count = np.sum(y_pred)
        total_windows = len(y_pred)
        
        print(f"📊 예측 결과:")
        print(f"   - 총 윈도우 수: {total_windows}")
        print(f"   - 낙상 감지 윈도우: {fall_count}")
        print(f"   - 평균 낙상 확률: {fall_probability:.3f}")
        print(f"   - 낙상 감지 비율: {fall_count/total_windows*100:.1f}%")
        
        # 최종 판단
        if fall_count > total_windows * 0.3:  # 30% 이상 낙상으로 감지되면
            final_prediction = "🚨 낙상 감지!"
        elif fall_count > 0:
            final_prediction = "⚠️  의심스러운 활동"
        else:
            final_prediction = "✅ 정상 활동"
        
        print(f"🎯 최종 판단: {final_prediction}")
        
        # 실제 라벨이 있으면 정확도 계산
        if y_true is not None:
            accuracy = np.mean(y_pred == y_true)
            print(f"📈 정확도: {accuracy:.3f}")
        
        return {
            'predictions': y_pred_prob,
            'binary_predictions': y_pred,
            'fall_probability': fall_probability,
            'fall_count': fall_count,
            'final_prediction': final_prediction
        }
    
    def batch_test(self, file_paths, threshold=0.5):
        """여러 파일 일괄 테스트"""
        print(f"\n📂 일괄 테스트: {len(file_paths)}개 파일")
        
        results = {}
        
        for file_path in file_paths:
            try:
                result = self.predict_single_file(file_path, threshold)
                results[file_path] = result
            except Exception as e:
                print(f"❌ {file_path} 테스트 실패: {e}")
        
        # 전체 결과 요약
        print(f"\n📋 전체 결과 요약:")
        print("-" * 50)
        
        for file_path, result in results.items():
            filename = file_path.split('/')[-1]
            status = result['final_prediction']
            prob = result['fall_probability']
            print(f"{filename:20s} | {status:15s} | 확률: {prob:.3f}")
        
        return results
    
    def create_prediction_plot(self, csv_path, threshold=0.5):
        """예측 결과 시각화"""
        X_seq, y_true = self.load_and_preprocess_csv(csv_path)
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        plt.figure(figsize=(12, 6))
        
        # 예측 확률 플롯
        plt.subplot(2, 1, 1)
        plt.plot(predictions, label='낙상 확률', color='red', alpha=0.7)
        plt.axhline(y=threshold, color='orange', linestyle='--', label=f'임계값 ({threshold})')
        plt.fill_between(range(len(predictions)), predictions, alpha=0.3, color='red')
        plt.ylabel('낙상 확률')
        plt.title('시간에 따른 낙상 감지 확률')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 이진 예측 플롯
        plt.subplot(2, 1, 2)
        binary_pred = (predictions > threshold).astype(int)
        plt.plot(binary_pred, 'bo-', markersize=3, label='낙상 감지')
        if y_true is not None:
            plt.plot(y_true, 'ro-', markersize=3, alpha=0.7, label='실제 라벨')
        plt.ylabel('낙상 감지')
        plt.xlabel('시간 윈도우')
        plt.title('낙상 감지 결과')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 사용 예시
def main():
    """메인 테스트 함수"""
    print("🧪 낙상 감지 모델 테스트")
    print("=" * 50)
    
    # 모델 로드
    model_path = "./models/csi_fall_detection_128features.h5"
    
    try:
        tester = ModelTester(model_path)
        
        # 1. 단일 파일 테스트
        test_file = "./csi_data/case1/5_labeled.csv"
        result = tester.predict_single_file(test_file)
        
        # 2. 시각화
        tester.create_prediction_plot(test_file)
        
        # 3. 여러 파일 일괄 테스트
        test_files = [
            "./csi_data/case1/40.csv",
            "./csi_data/case2/4.csv",
            "./csi_data/case3/17_labeled.csv"
        ]
        batch_results = tester.batch_test(test_files)
        
    except FileNotFoundError:
        print("❌ 모델 파일을 찾을 수 없습니다!")
        print("   먼저 run_training.py를 실행하여 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()