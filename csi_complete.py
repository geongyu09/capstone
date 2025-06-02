# 간단 실행 스크립트 - run_training.py
from csi_fall_detection import CSIFallDetection
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

def main():
    print("🚀 CSI 낙상 감지 모델 학습 시작!")
    print("=" * 50)
    
    # 폴더 확인
    data_dir = "./csi_data"
    if not os.path.exists(data_dir):
        print(f"❌ 오류: {data_dir} 폴더가 없습니다!")
        print("다음과 같이 폴더를 만들고 CSV 파일들을 넣어주세요:")
        print("mkdir csi_data")
        print("# 그 다음 CSV 파일들을 csi_data 폴더에 복사")
        return
    
    # CSV 파일 개수 확인
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"📁 발견된 CSV 파일 수: {len(csv_files)}")
    
    if len(csv_files) == 0:
        print("❌ CSV 파일이 없습니다! csi_data 폴더에 CSV 파일들을 넣어주세요.")
        return
    
    # 1. 모델 초기화
    print("\n1️⃣ 모델 초기화...")
    detector = CSIFallDetection(window_size=50, stride=1)
    
    try:
        # 2. 데이터 로드
        print("\n2️⃣ 데이터 로드 중...")
        X, y, timestamps = detector.load_csv_files(data_dir)
        
        # 라벨 분포 확인
        fall_count = np.sum(y == 1)
        normal_count = np.sum(y == 0)
        
        print(f"📊 라벨 분포:")
        print(f"   - 낙상 데이터: {fall_count}개")
        print(f"   - 정상 데이터: {normal_count}개")
        
        if fall_count == 0:
            print("⚠️  경고: 낙상 데이터가 없습니다! 일부 CSV 파일의 label을 1로 설정해주세요.")
        
        if normal_count == 0:
            print("⚠️  경고: 정상 데이터가 없습니다! 일부 CSV 파일의 label을 0으로 설정해주세요.")
        
        # 3. 시퀀스 생성
        print("\n3️⃣ 시계열 시퀀스 생성 중...")
        X_seq, y_seq = detector.create_sequences(X, y)
        
        # 4. 데이터 전처리
        print("\n4️⃣ 데이터 전처리 중...")
        X_seq = detector.preprocess_data(X_seq, feature_selection=True)
        
        # 5. 데이터 분할 (라벨이 한 종류만 있으면 stratify 제거)
        print("\n5️⃣ 훈련/테스트 데이터 분할...")
        
        if len(np.unique(y_seq)) > 1:  # 라벨이 2종류 이상
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, 
                test_size=0.2, 
                stratify=y_seq,
                random_state=42
            )
        else:  # 라벨이 1종류뿐
            print("⚠️  라벨이 한 종류뿐입니다. stratify 없이 분할합니다.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, 
                test_size=0.2, 
                random_state=42
            )
        
        print(f"   - 훈련 세트: {X_train.shape}")
        print(f"   - 테스트 세트: {X_test.shape}")
        print(f"   - 훈련 라벨 분포: {np.bincount(y_train)}")
        print(f"   - 테스트 라벨 분포: {np.bincount(y_test)}")
        
        # 6. 모델 구축
        print("\n6️⃣ 모델 구축 중...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 특징 수에 따라 모델 타입 자동 선택
        if X_train.shape[2] > 100:
            model_type = 'lightweight'
            print(f"   특징 수가 많아 경량 모델을 사용합니다. ({X_train.shape[2]}개 특징)")
        else:
            model_type = 'standard'
            print(f"   표준 모델을 사용합니다. ({X_train.shape[2]}개 특징)")
        
        detector.build_model(input_shape, model_type=model_type)
        
        # 7. 모델 학습
        print("\n7️⃣ 모델 학습 시작...")
        print("   (학습 중에는 progress bar가 표시됩니다)")
        
        epochs = 150 if X_train.shape[2] > 100 else 100
        batch_size = 16 if X_train.shape[2] > 100 else 32
        
        print(f"   - 에포크: {epochs}")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 모델 타입: {model_type}")
        
        history = detector.train_model(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # 8. 모델 평가
        print("\n8️⃣ 모델 평가 중...")
        y_pred_prob, y_pred = detector.evaluate_model(X_test, y_test)
        
        # 9. 모델 저장
        print("\n9️⃣ 모델 저장 중...")
        os.makedirs('./models', exist_ok=True)
        model_name = f'./models/csi_fall_detection_{X_train.shape[2]}features.h5'
        detector.model.save(model_name)
        
        print(f"✅ 모델 저장 완료: {model_name}")
        print(f"✅ 학습 완료!")
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("📋 학습 결과 요약:")
        print(f"   - 사용된 CSV 파일: {len(csv_files)}개")
        print(f"   - 총 데이터 포인트: {X.shape[0]}개")
        print(f"   - 생성된 시퀀스: {X_seq.shape[0]}개")
        print(f"   - 특징 수: {X_train.shape[2]}개")
        print(f"   - 모델 타입: {model_type}")
        print(f"   - 저장된 모델: {model_name}")
        print("=" * 50)
        
        # 특징 중요도 분석 (옵션)
        if hasattr(detector, 'k_selector') and detector.k_selector:
            feature_scores = detector.k_selector.scores_
            selected_features = detector.k_selector.get_support(indices=True)
            print(f"\n🔍 선택된 특징 인덱스 (상위 10개):")
            top_features = selected_features[np.argsort(feature_scores[selected_features])[-10:]]
            print(f"   {top_features}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("상세 오류 정보:")
        import traceback
        traceback.print_exc()
        
        # 일반적인 해결 방법 제시
        print("\n💡 문제 해결 방법:")
        print("1. CSV 파일에 timestamp, label, feat_* 컬럼이 있는지 확인")
        print("2. 메모리 부족 시 window_size를 줄이거나 stride를 늘려보세요")
        print("3. 필요한 라이브러리가 설치되어 있는지 확인:")
        print("   pip install tensorflow pandas scikit-learn numpy matplotlib")

if __name__ == "__main__":
    main()