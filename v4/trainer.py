"""
CSI 낙상 감지 v4 - 모델 학습기
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import tensorflow as tf

from config import Config
from utils import (
    setup_logging, save_model_artifacts, create_timestamp,
    calculate_class_weights, format_time, memory_usage_mb,
    plot_training_history
)
from data_generator import create_data_generators
from model_builder import create_model


class CSIModelTrainer:
    """CSI 낙상 감지 모델 학습기"""
    
    def __init__(self, 
                 model_type: str = 'hybrid',
                 experiment_name: Optional[str] = None,
                 logger=None):
        """
        Args:
            model_type: 모델 타입 ('hybrid', 'simple', 'cnn')
            experiment_name: 실험 이름 (None이면 자동 생성)
            logger: 로거 객체
        """
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_{create_timestamp()}"
        self.logger = logger or setup_logging()
        
        # 초기화
        self.model = None
        self.model_builder = None
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.history = None
        self.class_weights = None
        
        self.logger.info(f"🚀 모델 학습기 초기화")
        self.logger.info(f"   모델 타입: {self.model_type}")
        self.logger.info(f"   실험 이름: {self.experiment_name}")
    
    def prepare_data(self) -> None:
        """데이터 제너레이터 준비"""
        self.logger.info("📊 데이터 제너레이터 준비 중...")
        
        start_time = time.time()
        
        # 데이터 제너레이터 생성
        self.train_gen, self.val_gen, self.test_gen = create_data_generators()
        
        # 통계 출력
        self.logger.info("✅ 데이터 제너레이터 준비 완료")
        self.logger.info(f"   훈련 시퀀스: {self.train_gen.total_sequences:,}개")
        self.logger.info(f"   검증 시퀀스: {self.val_gen.total_sequences:,}개")
        self.logger.info(f"   테스트 시퀀스: {self.test_gen.total_sequences:,}개")
        
        # 클래스 분포 확인 (샘플링)
        self.logger.info("📈 클래스 분포 분석 중 (샘플링)...")
        sample_labels = []
        sample_size = min(1000, len(self.train_gen.sequences))
        
        for i in range(0, sample_size, 10):  # 10개씩 건너뛰며 샘플링
            try:
                file_path, start_idx, end_idx = self.train_gen.sequences[i]
                _, y = self.train_gen._load_sequence(file_path, start_idx, end_idx)
                sample_labels.append(y)
            except:
                continue
        
        if sample_labels:
            sample_labels = np.array(sample_labels)
            fall_ratio = np.mean(sample_labels)
            self.logger.info(f"   샘플 클래스 분포: 정상 {1-fall_ratio:.3f}, 낙상 {fall_ratio:.3f}")
            
            # 클래스 가중치 계산
            self.class_weights = calculate_class_weights(sample_labels)
            self.logger.info(f"   클래스 가중치: {self.class_weights}")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"   준비 시간: {format_time(elapsed_time)}")
    
    def build_model(self) -> None:
        """모델 구축 및 컴파일"""
        self.logger.info(f"🧠 {self.model_type} 모델 구축 중...")
        
        # 입력 형태 설정
        input_shape = (Config.WINDOW_SIZE, Config.TOTAL_FEATURES)
        
        # 모델 생성
        self.model, self.model_builder = create_model(
            model_type=self.model_type,
            input_shape=input_shape,
            learning_rate=Config.LEARNING_RATE,
            class_weights=self.class_weights
        )
        
        # 모델 요약 출력
        self.model_builder.print_model_summary()
        
        self.logger.info("✅ 모델 구축 완료")
    
    def train(self, 
              epochs: int = Config.EPOCHS,
              patience: int = 10,
              save_best: bool = True) -> Dict[str, Any]:
        """모델 학습"""
        
        if self.model is None:
            raise ValueError("모델이 구축되지 않았습니다. build_model()을 먼저 호출하세요.")
        
        if self.train_gen is None:
            raise ValueError("데이터가 준비되지 않았습니다. prepare_data()를 먼저 호출하세요.")
        
        self.logger.info(f"🏋️ 모델 학습 시작")
        self.logger.info(f"   에포크: {epochs}")
        self.logger.info(f"   조기 종료 patience: {patience}")
        self.logger.info(f"   메모리 사용량: {memory_usage_mb():.1f} MB")
        
        # 콜백 설정
        callbacks = self.model_builder.get_callbacks(
            model_name=self.experiment_name,
            patience=patience
        )
        
        # 학습 시작
        start_time = time.time()
        
        try:
            self.history = self.model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # 학습 완료
            training_time = time.time() - start_time
            self.logger.info(f"✅ 학습 완료!")
            self.logger.info(f"   총 학습 시간: {format_time(training_time)}")
            self.logger.info(f"   최종 메모리 사용량: {memory_usage_mb():.1f} MB")
            
            # 최고 성능 에포크 정보
            best_epoch = np.argmin(self.history.history['val_loss']) + 1
            best_val_loss = np.min(self.history.history['val_loss'])
            best_val_acc = self.history.history['val_accuracy'][best_epoch - 1]
            
            self.logger.info(f"   최고 성능 에포크: {best_epoch}")
            self.logger.info(f"   최고 검증 손실: {best_val_loss:.4f}")
            self.logger.info(f"   최고 검증 정확도: {best_val_acc:.4f}")
            
            # 모델 저장
            if save_best:
                self._save_model()
            
            # 학습 히스토리 시각화
            self._plot_training_results()
            
            return {
                'history': self.history.history,
                'training_time': training_time,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_acc,
                'experiment_name': self.experiment_name
            }
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ 학습이 사용자에 의해 중단되었습니다.")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 학습 중 오류 발생: {e}")
            raise
    
    def evaluate(self) -> Dict[str, Any]:
        """테스트 데이터로 모델 평가"""
        
        if self.model is None:
            raise ValueError("모델이 없습니다.")
        
        if self.test_gen is None:
            raise ValueError("테스트 데이터가 준비되지 않았습니다.")
        
        self.logger.info("📊 모델 평가 시작...")
        
        # 평가 실행
        start_time = time.time()
        test_results = self.model.evaluate(self.test_gen, verbose=1)
        evaluation_time = time.time() - start_time
        
        # 결과 정리
        metric_names = self.model.metrics_names
        results = dict(zip(metric_names, test_results))
        
        self.logger.info(f"✅ 평가 완료 (시간: {format_time(evaluation_time)})")
        self.logger.info(f"📊 테스트 결과:")
        for metric, value in results.items():
            self.logger.info(f"   {metric}: {value:.4f}")
        
        return {
            'test_results': results,
            'evaluation_time': evaluation_time,
            'experiment_name': self.experiment_name
        }
    
    def _save_model(self) -> None:
        """모델 및 관련 파일들 저장"""
        self.logger.info("💾 모델 저장 중...")
        
        # 메타데이터 생성
        metadata = {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'timestamp': create_timestamp(),
            'config': {
                'window_size': Config.WINDOW_SIZE,
                'stride': Config.STRIDE,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE,
                'total_features': Config.TOTAL_FEATURES
            },
            'model_config': self.model_builder.model_config,
            'class_weights': self.class_weights,
            'input_shape': self.model.input_shape[1:],  # 배치 차원 제외
            'training_samples': self.train_gen.total_sequences if self.train_gen else 0,
            'validation_samples': self.val_gen.total_sequences if self.val_gen else 0,
            'test_samples': self.test_gen.total_sequences if self.test_gen else 0
        }
        
        # 학습 히스토리 추가
        if self.history:
            metadata['training_history'] = {
                'epochs': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'best_val_loss': float(np.min(self.history.history['val_loss'])),
                'best_epoch': int(np.argmin(self.history.history['val_loss']) + 1)
            }
        
        # 파일 저장 (스케일러는 전처리 단계에서 이미 저장됨)
        try:
            saved_paths = save_model_artifacts(
                model=self.model,
                scaler=None,  # 전처리에서 별도 관리
                metadata=metadata,
                model_dir=Config.MODEL_DIR,
                model_name=self.experiment_name
            )
            
            self.logger.info("✅ 모델 저장 완료")
            for artifact_type, path in saved_paths.items():
                self.logger.info(f"   {artifact_type}: {path}")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 저장 실패: {e}")
    
    def _plot_training_results(self) -> None:
        """학습 결과 시각화"""
        if self.history is None:
            return
        
        try:
            # 저장 경로
            plot_path = os.path.join(Config.LOG_DIR, f"{self.experiment_name}_training_history.png")
            
            # 플롯 생성
            plot_training_history(self.history.history, save_path=plot_path)
            
            self.logger.info(f"📊 학습 히스토리 저장: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"학습 히스토리 시각화 실패: {e}")
    
    def run_complete_training(self, 
                            epochs: int = Config.EPOCHS,
                            patience: int = 10,
                            evaluate_after_training: bool = True) -> Dict[str, Any]:
        """전체 학습 파이프라인 실행"""
        
        self.logger.info(f"🚀 전체 학습 파이프라인 시작: {self.experiment_name}")
        
        start_time = time.time()
        results = {}
        
        try:
            # 1. 데이터 준비
            self.prepare_data()
            
            # 2. 모델 구축
            self.build_model()
            
            # 3. 학습
            training_results = self.train(epochs=epochs, patience=patience)
            if training_results:
                results.update(training_results)
            
            # 4. 평가 (선택적)
            if evaluate_after_training and training_results:
                eval_results = self.evaluate()
                results.update(eval_results)
            
            # 총 소요 시간
            total_time = time.time() - start_time
            results['total_time'] = total_time
            
            self.logger.info(f"🎉 전체 파이프라인 완료!")
            self.logger.info(f"   총 소요 시간: {format_time(total_time)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 실행 실패: {e}")
            raise


def train_model(model_type: str = 'hybrid',
               experiment_name: Optional[str] = None,
               epochs: int = Config.EPOCHS,
               patience: int = 10) -> Dict[str, Any]:
    """모델 학습 헬퍼 함수"""
    
    trainer = CSIModelTrainer(
        model_type=model_type,
        experiment_name=experiment_name
    )
    
    return trainer.run_complete_training(
        epochs=epochs,
        patience=patience,
        evaluate_after_training=True
    )


if __name__ == "__main__":
    # 학습 테스트
    print("🧪 CSI 모델 학습기 테스트")
    print("=" * 50)
    
    # 설정 출력
    print(f"학습 설정:")
    print(f"  윈도우 크기: {Config.WINDOW_SIZE}")
    print(f"  배치 크기: {Config.BATCH_SIZE}")
    print(f"  에포크: {Config.EPOCHS}")
    print(f"  학습률: {Config.LEARNING_RATE}")
    
    # 사용자 확인
    choice = input("\n테스트 학습을 시작하시겠습니까? (y/n): ").lower().strip()
    
    if choice == 'y':
        try:
            # 간단한 모델로 테스트
            results = train_model(
                model_type='simple',
                experiment_name=f'test_{create_timestamp()}',
                epochs=3,  # 테스트용으로 짧게
                patience=2
            )
            
            print(f"\n✅ 테스트 학습 완료!")
            print(f"실험 이름: {results.get('experiment_name', 'Unknown')}")
            print(f"총 소요 시간: {format_time(results.get('total_time', 0))}")
            
            if 'test_results' in results:
                print(f"테스트 정확도: {results['test_results'].get('accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("테스트를 취소했습니다.")
