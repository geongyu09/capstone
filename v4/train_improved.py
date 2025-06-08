"""
향상된 모델 학습 스크립트
기존 trainer.py를 수정하여 향상된 모델을 사용합니다.
"""

import os
import sys
import glob
from datetime import datetime

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import setup_logging, create_timestamp
from data_generator import create_data_generators
from improved_model_fixed import build_improved_model, compile_improved_model, FocalLoss

def train_with_improved_model():
    """향상된 모델로 학습 실행"""
    
    logger = setup_logging()
    logger.info("🚀 향상된 모델 학습 시작")
    
    try:
        # 1. 데이터 제너레이터 준비
        logger.info("📊 데이터 제너레이터 준비 중...")
        train_gen, val_gen, test_gen = create_data_generators()
        
        logger.info(f"   훈련 시퀀스: {train_gen.total_sequences:,}개")
        logger.info(f"   검증 시퀀스: {val_gen.total_sequences:,}개")
        logger.info(f"   테스트 시퀀스: {test_gen.total_sequences:,}개")
        
        # 2. 향상된 모델 생성
        logger.info("🧠 향상된 모델 구축 중...")
        input_shape = (Config.WINDOW_SIZE, Config.TOTAL_FEATURES)
        model = build_improved_model(input_shape)
        model = compile_improved_model(model, use_focal_loss=True)
        
        # 모델 요약
        print("\n📋 향상된 모델 아키텍처:")
        model.summary()
        
        # 3. 학습 설정
        experiment_name = f"improved_hybrid_{create_timestamp()}"
        
        # 콜백 설정
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # 향상된 모델이므로 더 오래 기다림
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f"./models/{experiment_name}_best.keras",
                monitor='val_loss',
                save_best_only=True,
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
        
        # 4. 학습 실행
        logger.info(f"🏋️ 모델 학습 시작 (실험명: {experiment_name})")
        
        epochs = 50  # 향상된 모델이므로 더 적은 에포크로 시작
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # 5. 최종 모델 저장
        final_model_path = f"./models/{experiment_name}_final.keras"
        model.save(final_model_path)
        
        # 6. 메타데이터 저장
        import json
        metadata = {
            'experiment_name': experiment_name,
            'model_type': 'improved_hybrid',
            'timestamp': create_timestamp(),
            'architecture': 'Multi-scale CNN + Bidirectional LSTM with Attention',
            'training_info': {
                'epochs_trained': len(history.history['loss']),
                'best_val_loss': min(history.history['val_loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            },
            'config': {
                'window_size': Config.WINDOW_SIZE,
                'total_features': Config.TOTAL_FEATURES,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': 0.001,
                'loss_function': 'focal_loss'
            },
            'improvements': [
                'Multi-scale CNN with residual connections',
                'Bidirectional LSTM with custom attention',
                'Focal Loss for class imbalance',
                'Improved data augmentation strategy'
            ]
        }
        
        metadata_path = f"./models/{experiment_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 7. 결과 출력
        logger.info("✅ 향상된 모델 학습 완료!")
        logger.info(f"   실험명: {experiment_name}")
        logger.info(f"   최종 모델: {final_model_path}")
        logger.info(f"   최고 모델: ./models/{experiment_name}_best.keras")
        logger.info(f"   메타데이터: {metadata_path}")
        
        print(f"\n📊 학습 결과:")
        print(f"   에포크 수: {len(history.history['loss'])}")
        print(f"   최고 검증 손실: {min(history.history['val_loss']):.4f}")
        print(f"   최종 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")
        
        return experiment_name, history
        
    except Exception as e:
        logger.error(f"❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_improved_model(experiment_name):
    """향상된 모델 평가"""
    
    logger = setup_logging()
    logger.info(f"📊 향상된 모델 평가: {experiment_name}")
    
    try:
        from evaluator import evaluate_saved_model
        
        # 최고 성능 모델 평가
        best_model_name = f"{experiment_name}_best"
        results = evaluate_saved_model(best_model_name, detailed=True)
        
        if results:
            logger.info("✅ 향상된 모델 평가 완료!")
            
            # 성능 비교
            print(f"\n📈 향상된 모델 성능:")
            print(f"   정확도: {results['basic_metrics'].get('accuracy', 0):.3f}")
            if 'best_f1_score' in results:
                print(f"   최고 F1: {results['best_f1_score']:.3f}")
                print(f"   최적 임계값: {results['best_threshold']:.3f}")
            
            return results
        else:
            logger.error("평가 실패")
            return None
            
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}")
        return None

def main():
    """메인 실행 함수"""
    
    print("🚀 향상된 CSI 낙상 감지 모델 학습")
    print("=" * 50)
    
    print("🤔 어떤 작업을 수행하시겠습니까?")
    print("1. 향상된 모델 학습 (추천)")
    print("2. 기존 모델과 성능 비교")
    print("3. 학습된 향상된 모델 평가")
    
    choice = input("\n선택 (1-3): ").strip()
    
    if choice == "1":
        print("\n🏋️ 향상된 모델 학습을 시작합니다...")
        print("⏰ 예상 소요시간: 30분 ~ 1시간")
        
        confirm = input("계속하시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            experiment_name, history = train_with_improved_model()
            
            if experiment_name:
                print(f"\n✅ 학습 완료: {experiment_name}")
                
                # 자동으로 평가 실행
                eval_choice = input("\n평가를 바로 실행하시겠습니까? (y/n): ").lower()
                if eval_choice == 'y':
                    evaluate_improved_model(experiment_name)
        else:
            print("학습을 취소했습니다.")
    
    elif choice == "2":
        print("\n📊 기존 모델과 성능 비교")
        print("1. 먼저 기존 모델 평가: python main.py --mode evaluate")
        print("2. 향상된 모델 학습 후 비교")
        
    elif choice == "3":
        # 사용 가능한 향상된 모델 찾기
        improved_models = glob.glob("./models/improved_hybrid_*_best.keras")
        
        if not improved_models:
            print("❌ 학습된 향상된 모델이 없습니다.")
            print("먼저 선택 1번으로 모델을 학습하세요.")
        else:
            print(f"\n📋 사용 가능한 향상된 모델: {len(improved_models)}개")
            latest_model = sorted(improved_models)[-1]
            model_name = os.path.splitext(os.path.basename(latest_model))[0]
            
            print(f"평가 대상: {model_name}")
            evaluate_improved_model(model_name)
    
    else:
        print("올바른 선택이 아닙니다.")

if __name__ == "__main__":
    main()
