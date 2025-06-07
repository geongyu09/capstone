"""
학습된 모델 수동 저장 스크립트
저장 오류 발생 시 사용
"""

import os
import glob
import json
import pickle
from datetime import datetime
import tensorflow as tf
from config import Config

def find_latest_model():
    """가장 최근 학습된 모델 찾기"""
    model_files = glob.glob(os.path.join(Config.MODEL_DIR, "*_best.keras"))
    
    if not model_files:
        print("❌ 저장된 모델을 찾을 수 없습니다.")
        return None
    
    # 파일 수정 시간 기준으로 가장 최근 모델 선택
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"📂 가장 최근 모델: {latest_model}")
    
    return latest_model

def save_model_manually(model_path):
    """모델을 수동으로 저장"""
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return False
    
    try:
        # 모델 로드
        print("📂 모델 로딩 중...")
        model = tf.keras.models.load_model(model_path)
        
        # 새로운 파일명 생성
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        if base_name.endswith('_best'):
            base_name = base_name[:-5]  # '_best' 제거
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_name = f"{base_name}_manual_{timestamp}"
        
        # 새로운 경로들
        new_model_path = os.path.join(Config.MODEL_DIR, f"{new_model_name}.keras")
        metadata_path = os.path.join(Config.MODEL_DIR, f"{new_model_name}_metadata.json")
        
        # 모델 저장
        print("💾 모델 저장 중...")
        model.save(new_model_path)
        
        # 메타데이터 생성
        print("📋 메타데이터 생성 중...")
        metadata = {
            'experiment_name': new_model_name,
            'model_type': 'hybrid',  # 기본값
            'timestamp': timestamp,
            'original_model_path': model_path,
            'config': {
                'window_size': Config.WINDOW_SIZE,
                'stride': Config.STRIDE,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'total_features': Config.TOTAL_FEATURES
            },
            'input_shape': [Config.WINDOW_SIZE, Config.TOTAL_FEATURES],
            'total_params': int(model.count_params()),
            'manual_save': True,
            'note': '학습 완료 후 수동으로 저장된 모델'
        }
        
        # 메타데이터 저장 (안전한 방식)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print("✅ 모델 수동 저장 완료!")
        print(f"   모델: {new_model_path}")
        print(f"   메타데이터: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 수동 저장 실패: {e}")
        return False

def list_all_models():
    """모든 모델 나열"""
    print("\n📂 저장된 모델 목록:")
    print("-" * 40)
    
    # .keras 파일들
    keras_files = glob.glob(os.path.join(Config.MODEL_DIR, "*.keras"))
    
    if not keras_files:
        print("저장된 모델이 없습니다.")
        return
    
    for i, model_file in enumerate(sorted(keras_files), 1):
        file_name = os.path.basename(model_file)
        file_size = os.path.getsize(model_file) / 1024 / 1024  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
        
        print(f"{i:2d}. {file_name}")
        print(f"    크기: {file_size:.1f} MB")
        print(f"    수정: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def main():
    """메인 함수"""
    print("🔧 모델 수동 저장 도구")
    print("=" * 40)
    
    # 현재 모델 상태 확인
    list_all_models()
    
    # 가장 최근 모델 찾기
    latest_model = find_latest_model()
    
    if latest_model:
        print(f"\n💡 가장 최근 학습된 모델을 수동으로 저장하시겠습니까?")
        print(f"   모델: {os.path.basename(latest_model)}")
        
        choice = input("저장하시겠습니까? (y/n): ").lower().strip()
        
        if choice == 'y':
            success = save_model_manually(latest_model)
            
            if success:
                print("\n🎉 저장 완료! 이제 평가를 실행할 수 있습니다:")
                print("   python main.py --mode evaluate")
                print("   또는")
                print("   python quick_start.py → 4. 모델 평가")
            else:
                print("❌ 저장에 실패했습니다.")
        else:
            print("🔄 저장을 취소했습니다.")
    
    # 최종 모델 목록 다시 표시
    print("\n📂 최종 모델 목록:")
    list_all_models()

if __name__ == "__main__":
    main()
