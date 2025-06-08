"""
데이터 분석 결과를 바탕으로 한 Recall 최적화 학습
- 데이터는 충분 (낙상 64.38%)
- 문제는 모델/학습 설정에 있음
"""

import os
import sys
import numpy as np
from datetime import datetime

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 최적화된 config 사용
import config_optimized as Config
from config_optimized import Config as Cfg, ModelConfig
from utils import setup_logging, create_timestamp
from data_generator import create_data_generators
from improved_model_fixed import build_improved_model, FocalLoss

def create_optimized_model():
    """최적화된 Recall 모델 생성"""
    
    logger = setup_logging()
    logger.info("🎯 Recall 최적화 모델 생성")
    
    # 입력 형태
    input_shape = (Cfg.WINDOW_SIZE, Cfg.TOTAL_FEATURES)
    
    # 향상된 모델 구축 (드롭아웃 완화)
    model = build_improved_model(input_shape)
    
    # 커스텀 컴파일
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    
    # 옵티마이저 (안정적 학습)
    optimizer = Adam(
        learning_rate=Cfg.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # 적당한 Focal Loss
    focal_loss = FocalLoss(
        alpha=Cfg.FOCAL_LOSS_ALPHA,
        gamma=Cfg.FOCAL_LOSS_GAMMA
    )
    
    # 상세한 메트릭
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss,
        metrics=metrics
    )
    
    logger.info("✅ 최적화된 모델 컴파일 완료")
    logger.info(f"   학습률: {Cfg.LEARNING_RATE}")
    logger.info(f"   Focal Loss: α={Cfg.FOCAL_LOSS_ALPHA}, γ={Cfg.FOCAL_LOSS_GAMMA}")
    
    return model

def train_optimized_model():
    """최적화된 설정으로 학습"""
    
    logger = setup_logging()
    logger.info("🚀 Recall 최적화 학습 시작")
    
    try:
        # 1. 데이터 제너레이터
        logger.info("📊 데이터 제너레이터 준비...")
        
        # Config 설정
        import config
        config.Config.OVERLAP_THRESHOLD = Cfg.OVERLAP_THRESHOLD
        config.Config.BATCH_SIZE = Cfg.BATCH_SIZE
        
        train_gen, val_gen, test_gen = create_data_generators()
        
        logger.info(f"   훈련 시퀀스: {train_gen.total_sequences:,}개")
        logger.info(f"   검증 시퀀스: {val_gen.total_sequences:,}개")
        logger.info(f"   테스트 시퀀스: {test_gen.total_sequences:,}개")
        
        # 클래스 분포 재확인
        logger.info("📈 클래스 분포 재확인...")
        sample_labels = []
        for i in range(min(3, len(train_gen))):
            _, y_batch = train_gen[i]
            sample_labels.extend(y_batch)
        
        if sample_labels:
            fall_ratio = np.mean(sample_labels)
            logger.info(f"   낙상 비율: {fall_ratio*100:.1f}% (예상: ~64%)")
        
        # 2. 최적화된 모델 생성
        logger.info("🧠 최적화된 모델 구축...")
        model = create_optimized_model()
        
        print("\n📋 최적화된 모델 요약:")
        model.summary()
        
        # 3. 학습 콜백 (Recall 중심)
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
        
        class DetailedMonitor(Callback):
            """상세한 성능 모니터링"""
            def __init__(self):
                super().__init__()
                self.best_recall = 0
                self.epochs_since_improvement = 0
                
            def on_epoch_end(self, epoch, logs=None):
                # 주요 메트릭 추출
                val_recall = logs.get('val_recall', 0)
                val_precision = logs.get('val_precision', 0)
                val_accuracy = logs.get('val_accuracy', 0)
                val_loss = logs.get('val_loss', 0)
                
                # F1 Score 계산
                if val_precision + val_recall > 0:
                    f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
                else:
                    f1 = 0
                
                # 진행상황 출력
                print(f\"\\n📊 Epoch {epoch+1} 결과:\")\n                print(f\"   Recall: {val_recall:.4f} | Precision: {val_precision:.4f} | F1: {f1:.4f}\")\n                print(f\"   Accuracy: {val_accuracy:.4f} | Loss: {val_loss:.4f}\")\n                \n                # Recall 개선 확인\n                if val_recall > self.best_recall:\n                    improvement = val_recall - self.best_recall\n                    self.best_recall = val_recall\n                    self.epochs_since_improvement = 0\n                    print(f\"   🎯 새로운 최고 Recall! (+{improvement:.4f})\")\n                else:\n                    self.epochs_since_improvement += 1\n                    if self.epochs_since_improvement >= 5:\n                        print(f\"   ⚠️ {self.epochs_since_improvement}번째 에포크 동안 Recall 개선 없음\")\n                \n                # 목표 달성 확인\n                if val_recall >= 0.8:\n                    print(\"   🎉 목표 Recall 80% 달성!\")\n                elif val_recall >= 0.7:\n                    print(\"   ✅ 양호한 Recall 70% 이상\")\n                elif val_recall >= 0.5:\n                    print(\"   👍 괜찮은 Recall 50% 이상\")\n                elif val_recall < 0.3 and epoch > 10:\n                    print(\"   🚨 Recall이 여전히 낮습니다. 설정 재검토 필요\")\n        \n        experiment_name = f\"recall_optimized_{create_timestamp()}\"\n        \n        callbacks = [\n            DetailedMonitor(),\n            \n            # Recall 기준 Early Stopping (여유있게)\n            EarlyStopping(\n                monitor='val_recall',\n                patience=20,  # 충분한 시간\n                mode='max',\n                restore_best_weights=True,\n                verbose=1\n            ),\n            \n            # 모델 저장 (Recall 기준)\n            ModelCheckpoint(\n                filepath=f\"./models/{experiment_name}_best.keras\",\n                monitor='val_recall',\n                save_best_only=True,\n                mode='max',\n                verbose=1\n            ),\n            \n            # 학습률 감소 (점진적)\n            ReduceLROnPlateau(\n                monitor='val_recall',\n                factor=0.7,  # 더 부드럽게\n                patience=10,\n                mode='max',\n                min_lr=1e-6,\n                verbose=1\n            )\n        ]\n        \n        # 4. 학습 실행\n        logger.info(f\"🏋️ 최적화된 학습 시작 (실험명: {experiment_name})\")\n        logger.info(f\"   에포크: {Cfg.EPOCHS}\")\n        logger.info(f\"   예상 시간: 1-2시간\")\n        \n        # 학습 진행\n        history = model.fit(\n            train_gen,\n            validation_data=val_gen,\n            epochs=Cfg.EPOCHS,\n            callbacks=callbacks,\n            verbose=1,\n            class_weight=Cfg.CLASS_WEIGHTS\n        )\n        \n        # 5. 결과 분석\n        logger.info(\"📊 학습 결과 분석...\")\n        \n        best_recall = max(history.history.get('val_recall', [0]))\n        best_precision = max(history.history.get('val_precision', [0]))\n        best_accuracy = max(history.history.get('val_accuracy', [0]))\n        \n        # F1 Score 계산\n        val_recalls = history.history.get('val_recall', [0])\n        val_precisions = history.history.get('val_precision', [0])\n        f1_scores = []\n        for r, p in zip(val_recalls, val_precisions):\n            if r + p > 0:\n                f1_scores.append(2 * (r * p) / (r + p))\n            else:\n                f1_scores.append(0)\n        best_f1 = max(f1_scores) if f1_scores else 0\n        \n        logger.info(f\"✅ 최적화 학습 완료!\")\n        logger.info(f\"   최고 Recall: {best_recall:.4f}\")\n        logger.info(f\"   최고 Precision: {best_precision:.4f}\")\n        logger.info(f\"   최고 F1 Score: {best_f1:.4f}\")\n        logger.info(f\"   최고 Accuracy: {best_accuracy:.4f}\")\n        \n        # 성과 평가\n        if best_recall >= 0.8:\n            logger.info(\"🎉 탁월한 성과! Recall 80% 이상 달성\")\n            success_level = \"excellent\"\n        elif best_recall >= 0.7:\n            logger.info(\"🌟 훌륭한 성과! Recall 70% 이상 달성\")\n            success_level = \"great\"\n        elif best_recall >= 0.5:\n            logger.info(\"👍 좋은 성과! Recall 50% 이상 달성\")\n            success_level = \"good\"\n        elif best_recall >= 0.3:\n            logger.info(\"📈 개선됨! Recall 30% 이상 달성\")\n            success_level = \"improved\"\n        else:\n            logger.warning(\"🤔 추가 최적화 필요. Recall이 아직 낮음\")\n            success_level = \"needs_work\"\n        \n        # 6. 메타데이터 저장\n        import json\n        \n        metadata = {\n            'experiment_name': experiment_name,\n            'model_type': 'recall_optimized_hybrid',\n            'timestamp': create_timestamp(),\n            'data_analysis': {\n                'fall_sequences_ratio': '64.38%',\n                'data_balance': 'good',\n                'problem_identified': 'model_training_settings'\n            },\n            'optimizations': [\n                f'학습률 최적화: {Cfg.LEARNING_RATE}',\n                f'적당한 클래스 가중치: {Cfg.CLASS_WEIGHTS}',\n                f'드롭아웃 완화',\n                f'안정적 Focal Loss: α={Cfg.FOCAL_LOSS_ALPHA}, γ={Cfg.FOCAL_LOSS_GAMMA}',\n                f'충분한 에포크: {Cfg.EPOCHS}'\n            ],\n            'results': {\n                'best_recall': float(best_recall),\n                'best_precision': float(best_precision),\n                'best_f1_score': float(best_f1),\n                'best_accuracy': float(best_accuracy),\n                'success_level': success_level,\n                'epochs_trained': len(history.history['loss'])\n            }\n        }\n        \n        # 메타데이터 저장\n        metadata_path = f\"./models/{experiment_name}_metadata.json\"\n        with open(metadata_path, 'w', encoding='utf-8') as f:\n            json.dump(metadata, f, indent=2, ensure_ascii=False)\n        \n        logger.info(f\"💾 결과 저장 완료: {metadata_path}\")\n        \n        return experiment_name, history, metadata\n        \n    except Exception as e:\n        logger.error(f\"❌ 최적화 학습 실패: {e}\")\n        import traceback\n        traceback.print_exc()\n        return None, None, None\n\ndef quick_test():\n    \"\"\"빠른 테스트 (10 에포크)\"\"\"\n    print(\"🧪 빠른 최적화 테스트 (10 에포크)\")\n    \n    original_epochs = Cfg.EPOCHS\n    Cfg.EPOCHS = 10\n    \n    try:\n        experiment_name, history, metadata = train_optimized_model()\n        \n        if history and 'val_recall' in history.history:\n            final_recall = history.history['val_recall'][-1]\n            max_recall = max(history.history['val_recall'])\n            \n            print(f\"\\n🎯 10 에포크 결과:\")\n            print(f\"   최종 Recall: {final_recall:.4f}\")\n            print(f\"   최고 Recall: {max_recall:.4f}\")\n            \n            if max_recall > 0.4:\n                print(\"✅ 좋은 시작! 전체 학습을 진행하세요.\")\n                return True\n            elif max_recall > 0.2:\n                print(\"📈 개선 중! 더 긴 학습이 필요할 수 있습니다.\")\n                return True\n            else:\n                print(\"⚠️ 개선이 필요합니다. 설정을 다시 검토하세요.\")\n                return False\n        else:\n            print(\"❌ 테스트 실패\")\n            return False\n    finally:\n        Cfg.EPOCHS = original_epochs\n\ndef main():\n    \"\"\"메인 실행 함수\"\"\"\n    \n    print(\"🎯 CSI 낙상 감지 Recall 최적화 학습\")\n    print(\"=\" * 50)\n    \n    # 분석 결과 요약\n    print(\"📊 데이터 분석 결과 요약:\")\n    print(\"   ✅ 낙상 시퀀스: 64.38% (충분함!)\")\n    print(\"   ✅ 클래스 불균형: 심각하지 않음\")\n    print(\"   🎯 문제: 모델/학습 설정\")\n    print(\"   💡 해결: 안정적 학습 + 적당한 가중치\")\n    \n    # 최적화된 설정 출력\n    Cfg.print_config()\n    \n    print(\"\\n🤔 어떤 작업을 수행하시겠습니까?\")\n    print(\"1. 빠른 최적화 테스트 (10 에포크, 추천)\")\n    print(\"2. 전체 최적화 학습 (80 에포크)\")\n    print(\"3. 설정만 확인\")\n    \n    choice = input(\"\\n선택 (1-3): \").strip()\n    \n    if choice == \"1\":\n        success = quick_test()\n        \n        if success:\n            full_train = input(\"\\n전체 학습을 진행하시겠습니까? (y/n): \").lower()\n            if full_train == 'y':\n                train_optimized_model()\n        \n    elif choice == \"2\":\n        print(\"\\n🏋️ 전체 최적화 학습을 시작합니다...\")\n        print(\"⏰ 예상 소요시간: 1-2시간\")\n        \n        confirm = input(\"계속하시겠습니까? (y/n): \").lower()\n        if confirm == 'y':\n            experiment_name, history, metadata = train_optimized_model()\n            \n            if metadata:\n                success_level = metadata['results']['success_level']\n                best_recall = metadata['results']['best_recall']\n                \n                print(f\"\\n🎯 최종 결과: {success_level}\")\n                print(f\"   최고 Recall: {best_recall:.4f}\")\n                \n                if success_level in ['excellent', 'great', 'good']:\n                    print(\"\\n🎉 Recall 문제가 해결되었습니다!\")\n                else:\n                    print(\"\\n📈 개선되었지만 추가 튜닝이 필요할 수 있습니다.\")\n        \n    elif choice == \"3\":\n        print(\"\\n✅ 설정 확인 완료\")\n        \n    else:\n        print(\"올바른 선택이 아닙니다.\")\n\nif __name__ == \"__main__\":\n    main()\n