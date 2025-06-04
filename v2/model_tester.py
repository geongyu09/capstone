# model_tester.py
import pandas as pd
import numpy as np
import os
import glob
import pickle
import json
from datetime import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class CSIModelTester:
    """학습된 CSI 모델을 테스트하는 클래스"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.metadata = None
        
        # 모델 정보
        self.window_size = None
        self.stride = None
        self.overlap_threshold = None
    
    def find_latest_model(self, pattern="*model*.keras"):
        """가장 최근 학습된 모델 파일 찾기"""
        print("🔍 학습된 모델 파일 찾는 중...")
        
        # .keras 파일들 찾기
        keras_files = glob.glob(pattern)
        
        # .h5 파일들도 찾기
        h5_files = glob.glob(pattern.replace('.keras', '.h5'))
        
        all_models = keras_files + h5_files
        
        if not all_models:
            print("❌ 학습된 모델 파일을 찾을 수 없습니다!")
            print("💡 다음 명령으로 먼저 학습을 진행하세요:")
            print("   python v2/multi_file_trainer.py")
            return None
        
        # 가장 최근 파일 선택
        latest_model = max(all_models, key=os.path.getctime)
        
        print(f"✅ 최신 모델 발견: {latest_model}")
        
        # 파일 정보 출력
        file_size = os.path.getsize(latest_model) / 1024  # KB
        create_time = datetime.fromtimestamp(os.path.getctime(latest_model))
        print(f"   📄 크기: {file_size:.1f} KB")
        print(f"   🕐 생성 시간: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return latest_model
    
    def load_model_and_preprocessors(self, model_path):
        """모델과 전처리기들 로드"""
        print(f"📥 모델 로딩: {os.path.basename(model_path)}")
        
        try:
            # 1. 모델 로드
            self.model = load_model(model_path)
            print(f"   ✅ 모델 로드 완료")
            print(f"   📊 모델 구조: {self.model.input_shape} → {self.model.output_shape}")
            
            # 2. 전처리기 로드
            base_path = model_path.replace('.keras', '').replace('.h5', '')
            
            # 스케일러 로드
            scaler_path = base_path + '_scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"   ✅ 스케일러 로드 완료")
            else:
                print(f"   ⚠️ 스케일러 파일 없음: {scaler_path}")
            
            # 특징 선택기 로드
            selector_path = base_path + '_selector.pkl'
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.feature_selector = pickle.load(f)
                print(f"   ✅ 특징 선택기 로드 완료")
            else:
                print(f"   ⚠️ 특징 선택기 파일 없음: {selector_path}")
            
            # 메타데이터 로드
            metadata_path = base_path + '_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"   ✅ 메타데이터 로드 완료")
                
                # 메타데이터에서 설정값 추출
                self.window_size = self.metadata.get('window_size', 50)
                self.stride = self.metadata.get('stride', 5)
                self.overlap_threshold = self.metadata.get('overlap_threshold', 0.3)
                
                print(f"   📋 모델 설정: 윈도우={self.window_size}, 스트라이드={self.stride}")
            else:
                print(f"   ⚠️ 메타데이터 파일 없음: {metadata_path}")
                # 기본값 사용
                self.window_size = 50
                self.stride = 5
                self.overlap_threshold = 0.3
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def preprocess_data(self, X):
        """데이터 전처리 (학습 시와 동일하게)"""
        print("🔧 데이터 전처리...")
        
        # 1. 특징 선택 적용
        if self.feature_selector:
            print("   📊 특징 선택 적용...")
            
            if 'variance_selector' in self.feature_selector:
                X_var = self.feature_selector['variance_selector'].transform(X)
                
                if self.feature_selector.get('k_selector'):
                    X_selected = self.feature_selector['k_selector'].transform(X_var)
                else:
                    X_selected = X_var
            else:
                # 간단한 선택 (기본값)
                X_selected = X[:, 10:246] if X.shape[1] > 246 else X
        else:
            print("   ⚠️ 특징 선택기 없음, 원본 데이터 사용")
            X_selected = X
        
        print(f"      특징 선택 후: {X.shape[1]} → {X_selected.shape[1]}개")
        
        # 2. 정규화 적용
        if self.scaler:
            print("   📏 정규화 적용...")
            X_normalized = self.scaler.transform(X_selected)
        else:
            print("   ⚠️ 스케일러 없음, 정규화 건너뛰기")
            X_normalized = X_selected
        
        return X_normalized
    
    def create_sequences(self, X, y=None):
        """시계열 시퀀스 생성"""
        print("⏰ 시퀀스 생성...")
        
        sequences = []
        labels = []
        
        for i in range(0, len(X) - self.window_size + 1, self.stride):
            window_X = X[i:i + self.window_size]
            sequences.append(window_X)
            
            if y is not None:
                window_y = y[i:i + self.window_size]
                # 라벨링 (학습 시와 동일)
                fall_ratio = np.sum(window_y == 1) / len(window_y)
                sequence_label = 1 if fall_ratio >= self.overlap_threshold else 0
                labels.append(sequence_label)
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels) if y is not None else None
        
        print(f"   ✅ 생성된 시퀀스: {X_seq.shape[0]}개")
        print(f"   📊 시퀀스 형태: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def test_on_csv(self, csv_path):
        """CSV 파일에서 모델 테스트"""
        print(f"\n📄 CSV 파일 테스트: {os.path.basename(csv_path)}")
        
        try:
            # 1. CSV 로드
            df = pd.read_csv(csv_path)
            print(f"   📊 파일 크기: {df.shape}")
            
            # 2. 특징 추출
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            X = df[feature_cols].values
            y = df['label'].values if 'label' in df.columns else None
            
            print(f"   📈 특징 수: {len(feature_cols)}개")
            if y is not None:
                unique, counts = np.unique(y, return_counts=True)
                print(f"   🏷️ 라벨 분포: {dict(zip(unique, counts))}")
            
            # 3. 전처리
            X_processed = self.preprocess_data(X)
            
            # 4. 시퀀스 생성
            X_seq, y_seq = self.create_sequences(X_processed, y)
            
            # 5. 예측 수행
            print("🔮 예측 수행...")
            predictions = self.model.predict(X_seq, verbose=0)
            pred_probabilities = predictions.flatten()
            pred_labels = (pred_probabilities > 0.5).astype(int)
            
            print(f"   ✅ 예측 완료: {len(predictions)}개 시퀀스")
            
            # 6. 결과 분석
            self.analyze_predictions(pred_probabilities, pred_labels, y_seq)
            
            # 7. 시각화
            self.visualize_results(pred_probabilities, pred_labels, y_seq)
            
            return pred_probabilities, pred_labels, y_seq
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def analyze_predictions(self, pred_probs, pred_labels, true_labels=None):
        """예측 결과 분석"""
        print("\n📊 예측 결과 분석:")
        
        # 기본 통계
        print(f"   🔮 예측 통계:")
        print(f"      평균 확률: {np.mean(pred_probs):.3f}")
        print(f"      최대 확률: {np.max(pred_probs):.3f}")
        print(f"      최소 확률: {np.min(pred_probs):.3f}")
        
        # 예측 분포
        unique_preds, pred_counts = np.unique(pred_labels, return_counts=True)
        print(f"   📈 예측 분포:")
        for label, count in zip(unique_preds, pred_counts):
            label_name = "낙상" if label == 1 else "정상"
            percentage = (count / len(pred_labels)) * 100
            print(f"      {label_name}: {count}개 ({percentage:.1f}%)")
        
        # 실제 라벨이 있으면 성능 평가
        if true_labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
            
            print(f"   🎯 성능 평가:")
            
            accuracy = accuracy_score(true_labels, pred_labels)
            print(f"      정확도: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            if len(np.unique(true_labels)) > 1:  # 두 클래스 모두 있는 경우
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"      정밀도: {precision:.3f} ({precision*100:.1f}%)")
                print(f"      재현율: {recall:.3f} ({recall*100:.1f}%)")
                print(f"      F1-점수: {f1:.3f}")
                
                # 혼동 행렬
                cm = confusion_matrix(true_labels, pred_labels)
                print(f"   📋 혼동 행렬:")
                print(f"      실제\\예측  정상  낙상")
                print(f"      정상     {cm[0,0]:4d}  {cm[0,1]:4d}")
                print(f"      낙상     {cm[1,0]:4d}  {cm[1,1]:4d}")
                
                # 낙상 감지 관점
                if cm[1,1] + cm[1,0] > 0:  # 실제 낙상이 있는 경우
                    miss_rate = cm[1,0] / (cm[1,1] + cm[1,0])
                    false_alarm_rate = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                    
                    print(f"   🎯 낙상 감지 관점:")
                    print(f"      놓친 낙상: {miss_rate*100:.1f}%")
                    print(f"      거짓 알람: {false_alarm_rate*100:.1f}%")
    
    def visualize_results(self, pred_probs, pred_labels, true_labels=None):
        """결과 시각화"""
        print("\n📊 결과 시각화 중...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 예측 확률 분포
            axes[0,0].hist(pred_probs, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0,0].axvline(x=0.5, color='red', linestyle='--', label='임계값 (0.5)')
            axes[0,0].set_title('예측 확률 분포')
            axes[0,0].set_xlabel('낙상 확률')
            axes[0,0].set_ylabel('빈도')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. 시간에 따른 예측 확률
            axes[0,1].plot(pred_probs, linewidth=1.5, color='blue', label='예측 확률')
            axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='임계값')
            
            if true_labels is not None:
                # 실제 낙상 구간 표시
                fall_indices = np.where(true_labels == 1)[0]
                if len(fall_indices) > 0:
                    axes[0,1].scatter(fall_indices, pred_probs[fall_indices], 
                                    color='red', s=20, alpha=0.7, label='실제 낙상')
            
            axes[0,1].set_title('시간에 따른 예측 확률')
            axes[0,1].set_xlabel('시퀀스 번호')
            axes[0,1].set_ylabel('낙상 확률')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. 예측 라벨 분포 (파이 차트)
            unique_preds, pred_counts = np.unique(pred_labels, return_counts=True)
            labels = ['정상' if x == 0 else '낙상' for x in unique_preds]
            colors = ['lightblue' if x == 0 else 'lightcoral' for x in unique_preds]
            
            axes[1,0].pie(pred_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('예측 라벨 분포')
            
            # 4. 실제 vs 예측 비교 (라벨이 있는 경우)
            if true_labels is not None:
                # 실제 라벨별 예측 확률 분포
                normal_probs = pred_probs[true_labels == 0]
                fall_probs = pred_probs[true_labels == 1]
                
                axes[1,1].hist(normal_probs, bins=20, alpha=0.7, label='실제 정상', color='blue', density=True)
                axes[1,1].hist(fall_probs, bins=20, alpha=0.7, label='실제 낙상', color='red', density=True)
                axes[1,1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='임계값')
                axes[1,1].set_title('실제 라벨별 예측 확률')
                axes[1,1].set_xlabel('예측 확률')
                axes[1,1].set_ylabel('밀도')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            else:
                axes[1,1].text(0.5, 0.5, '실제 라벨 없음', ha='center', va='center', 
                              transform=axes[1,1].transAxes, fontsize=14)
                axes[1,1].set_title('실제 라벨 없음')
            
            plt.tight_layout()
            
            # 이미지 저장
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f'model_test_results_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 결과 이미지 저장: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
    
    def quick_test(self, csv_path=None):
        """빠른 테스트 실행"""
        print("🚀 CSI 모델 빠른 테스트")
        print("=" * 50)
        
        # 1. 모델 찾기 및 로드
        model_path = self.find_latest_model()
        if not model_path:
            return False
        
        if not self.load_model_and_preprocessors(model_path):
            return False
        
        # 2. 테스트 파일 찾기 (다양한 경로 지원)
        if csv_path is None:
            test_files = [
                # 현재 디렉토리
                '32_labeled.csv', 
                'case32.csv', 
                'test.csv',
                # 상위 디렉토리
                '../32_labeled.csv',
                '../case32.csv',
                # csi_data 하위 디렉토리들
                '../csi_data/case3/32_labeled.csv',
                '../csi_data/case2/5.csv', 
                '../csi_data/case1/40.csv',
                '../csi_data/case1/32_labeled.csv',
                '../csi_data/case2/32_labeled.csv',
                '../csi_data/case3/5.csv',
                # 현재 디렉토리의 하위 폴더들
                './case1/32_labeled.csv',
                './case2/5.csv',
                './case3/40.csv'
            ]
            
            csv_path = None
            print("🔍 테스트 파일 찾는 중...")
            
            for test_file in test_files:
                if os.path.exists(test_file):
                    csv_path = test_file
                    print(f"   ✅ 발견: {test_file}")
                    break
                else:
                    print(f"   ❌ 없음: {test_file}")
            
            if csv_path is None:
                print("\n🔍 추가 검색 중...")
                # 추가 검색: 현재 디렉토리의 모든 CSV
                current_csv = glob.glob("*.csv")
                print(f"   현재 디렉토리 CSV: {current_csv}")
                
                # 추가 검색: 상위 디렉토리 CSV
                parent_csv = glob.glob("../*.csv")
                print(f"   상위 디렉토리 CSV: {parent_csv}")
                
                # 추가 검색: csi_data 하위의 모든 CSV
                csi_data_csv = glob.glob("../csi_data/**/*.csv", recursive=True)
                print(f"   csi_data 하위 CSV: {len(csi_data_csv)}개")
                
                if csi_data_csv:
                    print(f"   csi_data CSV 예시:")
                    for i, csv_file in enumerate(csi_data_csv[:5]):
                        file_size = os.path.getsize(csv_file) / 1024
                        print(f"      {i+1}. {csv_file} ({file_size:.1f} KB)")
                    if len(csi_data_csv) > 5:
                        print(f"      ... 외 {len(csi_data_csv)-5}개")
                
                # 가장 큰 CSV 파일 선택
                all_csv = current_csv + parent_csv + csi_data_csv
                if all_csv:
                    csv_sizes = [(f, os.path.getsize(f)) for f in all_csv]
                    csv_path = max(csv_sizes, key=lambda x: x[1])[0]
                    print(f"   🎯 가장 큰 파일 선택: {csv_path}")
                else:
                    print("❌ 테스트할 CSV 파일이 없습니다!")
                    print("\n💡 해결 방법:")
                    print("   1. CSV 파일을 현재 디렉토리로 복사")
                    print("   2. 직접 경로 지정:")
                    print("      tester.quick_test('../csi_data/case3/32_labeled.csv')")
                    print("   3. 파일 경로 확인:")
                    print("      ls ../csi_data/case*/*.csv")
                    return False
        
        # 파일 존재 재확인
        if not os.path.exists(csv_path):
            print(f"❌ 선택된 파일이 존재하지 않습니다: {csv_path}")
            return False
        
        # 3. 테스트 실행
        print(f"\n📄 테스트 파일: {csv_path}")
        file_size = os.path.getsize(csv_path) / 1024
        print(f"   파일 크기: {file_size:.1f} KB")
        
        pred_probs, pred_labels, true_labels = self.test_on_csv(csv_path)
        
        if pred_probs is not None:
            print(f"\n✅ 모델 테스트 완료!")
            print(f"📄 테스트 파일: {os.path.basename(csv_path)}")
            print(f"🔮 예측 수행: {len(pred_probs)}개 시퀀스")
            
            # 간단한 요약
            fall_predictions = np.sum(pred_labels == 1)
            max_prob = np.max(pred_probs)
            
            print(f"\n📋 테스트 요약:")
            print(f"   낙상 예측: {fall_predictions}개 시퀀스")
            print(f"   최대 확률: {max_prob:.3f}")
            
            if max_prob > 0.8:
                print(f"   🚨 높은 확률의 낙상 감지됨!")
            elif max_prob > 0.5:
                print(f"   ⚠️ 낙상 가능성 있음")
            else:
                print(f"   ✅ 정상 상태로 판단됨")
            
            return True
        
        return False

def main():
    """메인 실행 함수"""
    tester = CSIModelTester()
    
    print("🧪 CSI 모델 테스터")
    print("=" * 30)
    print("1. 빠른 테스트 (q)")
    print("2. 특정 파일 테스트 (f)")
    print("3. 모델 정보만 확인 (i)")
    print("4. 추천 파일로 테스트 (r)")
    
    choice = input("\n선택하세요 (q/f/i/r): ").strip().lower()
    
    if choice == 'q':
        # 빠른 테스트
        tester.quick_test()
        
    elif choice == 'f':
        # 특정 파일 테스트
        csv_file = input("CSV 파일 경로: ").strip()
        if os.path.exists(csv_file):
            tester.quick_test(csv_file)
        else:
            print(f"❌ 파일이 없습니다: {csv_file}")
    
    elif choice == 'i':
        # 모델 정보만 확인
        model_path = tester.find_latest_model()
        if model_path:
            tester.load_model_and_preprocessors(model_path)
    
    elif choice == 'r':
        # 추천 파일로 테스트
        recommended_files = [
            '../csi_data/case3/32_labeled.csv',
            '../csi_data/case2/5.csv', 
            '../csi_data/case1/40.csv',
        ]
        
        print("🎯 추천 테스트 파일들:")
        for i, file_path in enumerate(recommended_files):
            exists = "✅" if os.path.exists(file_path) else "❌"
            print(f"   {i+1}. {exists} {file_path}")
        
        try:
            file_num = int(input("\n파일 번호 선택 (1-5): ")) - 1
            if 0 <= file_num < len(recommended_files):
                selected_file = recommended_files[file_num]
                if os.path.exists(selected_file):
                    tester.quick_test(selected_file)
                else:
                    print(f"❌ 선택한 파일이 없습니다: {selected_file}")
            else:
                print("❌ 잘못된 번호입니다.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
    
    else:
        print("기본값으로 빠른 테스트를 실행합니다...")
        tester.quick_test()

# 추가: 직접 파일 지정하여 테스트하는 간단한 함수
def test_specific_file(csv_path):
    """특정 파일로 바로 테스트"""
    tester = CSIModelTester()
    return tester.quick_test(csv_path)

if __name__ == "__main__":
    main()