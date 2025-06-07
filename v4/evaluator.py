"""
CSI 낙상 감지 v4 - 모델 평가기
"""

import os
import glob
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf

from config import Config
from utils import (
    setup_logging, load_model_artifacts, evaluate_model,
    print_evaluation_results, plot_confusion_matrix, plot_roc_curve
)
from data_generator import CSIDataGenerator, create_data_generators


class CSIModelEvaluator:
    """CSI 낙상 감지 모델 평가기"""
    
    def __init__(self, model_path: str, logger=None):
        """
        Args:
            model_path: 모델 파일 경로 또는 실험 이름
            logger: 로거 객체
        """
        self.model_path = model_path
        self.logger = logger or setup_logging()
        
        self.model = None
        self.scaler = None
        self.metadata = None
        self.test_gen = None
        
        self.logger.info(f"📊 모델 평가기 초기화")
        self.logger.info(f"   모델 경로: {model_path}")
    
    def load_model(self) -> None:
        """모델 및 관련 파일들 로드"""
        self.logger.info("📂 모델 로딩 중...")
        
        try:
            # 모델 파일 경로 결정
            if self.model_path.endswith('.keras'):
                # 직접 파일 경로인 경우
                model_file = self.model_path
                model_name = os.path.splitext(os.path.basename(model_file))[0]
            else:
                # 실험 이름인 경우
                model_name = self.model_path
                model_file = os.path.join(Config.MODEL_DIR, f"{model_name}.keras")
            
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")
            
            # 모델 아티팩트 로드
            self.model, self.scaler, self.metadata = load_model_artifacts(
                Config.MODEL_DIR, model_name
            )
            
            self.logger.info("✅ 모델 로딩 완료")
            self.logger.info(f"   실험 이름: {self.metadata.get('experiment_name', 'Unknown')}")
            self.logger.info(f"   모델 타입: {self.metadata.get('model_type', 'Unknown')}")
            self.logger.info(f"   입력 형태: {self.metadata.get('input_shape', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def prepare_test_data(self) -> None:
        """테스트 데이터 준비"""
        self.logger.info("🔍 테스트 데이터 준비 중...")
        
        try:
            # 데이터 제너레이터 생성
            _, _, self.test_gen = create_data_generators()
            
            self.logger.info("✅ 테스트 데이터 준비 완료")
            self.logger.info(f"   테스트 시퀀스: {self.test_gen.total_sequences:,}개")
            
        except Exception as e:
            self.logger.error(f"❌ 테스트 데이터 준비 실패: {e}")
            raise
    
    def evaluate(self, detailed: bool = True) -> Dict[str, Any]:
        """모델 평가 실행"""
        
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        if self.test_gen is None:
            raise ValueError("테스트 데이터가 준비되지 않았습니다. prepare_test_data()를 먼저 호출하세요.")
        
        self.logger.info("📈 모델 평가 시작...")
        
        # 기본 평가
        test_results = self.model.evaluate(self.test_gen, verbose=1)
        metric_names = self.model.metrics_names
        basic_results = dict(zip(metric_names, test_results))
        
        results = {
            'basic_metrics': basic_results,
            'model_info': {
                'experiment_name': self.metadata.get('experiment_name', 'Unknown'),
                'model_type': self.metadata.get('model_type', 'Unknown'),
                'total_params': self.model.count_params()
            }
        }
        
        # 상세 평가
        if detailed:
            self.logger.info("🔬 상세 평가 실행 중...")
            detailed_results = self._detailed_evaluation()
            results.update(detailed_results)
        
        # 결과 출력
        self._print_results(results)
        
        return results
    
    def _detailed_evaluation(self) -> Dict[str, Any]:
        """상세 평가 실행"""
        
        # 예측 수집
        y_true = []
        y_scores = []
        
        self.logger.info("🔄 예측 수집 중...")
        
        for i in range(len(self.test_gen)):
            X_batch, y_batch = self.test_gen[i]
            scores_batch = self.model.predict(X_batch, verbose=0)
            
            y_true.extend(y_batch)
            y_scores.extend(scores_batch.flatten())
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"   진행률: {i+1}/{len(self.test_gen)} 배치")
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # 다양한 임계값으로 평가
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_results = {}
        
        for threshold in thresholds:
            eval_result = evaluate_model(
                model=self.model,
                X_test=None,  # 이미 예측이 완료됨
                y_test=y_true,
                threshold=threshold
            )
            
            # 예측값 직접 설정
            eval_result['predictions']['y_scores'] = y_scores.tolist()
            eval_result['predictions']['y_pred'] = (y_scores > threshold).astype(int).tolist()
            
            threshold_results[f'threshold_{threshold}'] = eval_result
        
        # 최적 임계값 찾기
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            f1 = threshold_results[f'threshold_{threshold}']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # 시각화
        self._create_visualizations(y_true, y_scores, best_threshold)
        
        return {
            'detailed_metrics': threshold_results,
            'best_threshold': best_threshold,
            'best_f1_score': best_f1,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_scores': y_scores.tolist()
            }
        }
    
    def _create_visualizations(self, y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> None:
        """시각화 생성"""
        
        try:
            experiment_name = self.metadata.get('experiment_name', 'unknown')
            
            # 혼동 행렬
            y_pred = (y_scores > threshold).astype(int)
            cm_path = os.path.join(Config.LOG_DIR, f"{experiment_name}_confusion_matrix.png")
            plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
            
            # ROC 커브
            roc_path = os.path.join(Config.LOG_DIR, f"{experiment_name}_roc_curve.png")
            roc_auc = plot_roc_curve(y_true, y_scores, save_path=roc_path)
            
            self.logger.info(f"📊 시각화 저장 완료")
            self.logger.info(f"   혼동 행렬: {cm_path}")
            self.logger.info(f"   ROC 커브: {roc_path} (AUC: {roc_auc:.3f})")
            
        except Exception as e:
            self.logger.warning(f"시각화 생성 실패: {e}")
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """결과 출력"""
        
        print(f"\n📊 모델 평가 결과")
        print("=" * 50)
        
        # 모델 정보
        model_info = results['model_info']
        print(f"🤖 모델 정보:")
        print(f"   실험 이름: {model_info['experiment_name']}")
        print(f"   모델 타입: {model_info['model_type']}")
        print(f"   총 파라미터: {model_info['total_params']:,}")
        
        # 기본 메트릭
        basic_metrics = results['basic_metrics']
        print(f"\n📈 기본 성능:")
        for metric, value in basic_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # 상세 결과 (있는 경우)
        if 'best_threshold' in results:
            print(f"\n🎯 최적 임계값: {results['best_threshold']}")
            print(f"🏆 최고 F1 점수: {results['best_f1_score']:.4f}")
            
            # 최적 임계값에서의 성능
            best_key = f"threshold_{results['best_threshold']}"
            if best_key in results['detailed_metrics']:
                best_result = results['detailed_metrics'][best_key]
                print_evaluation_results(best_result)


def evaluate_saved_model(model_name: str, detailed: bool = True) -> Dict[str, Any]:
    """저장된 모델 평가 헬퍼 함수"""
    
    evaluator = CSIModelEvaluator(model_name)
    evaluator.load_model()
    evaluator.prepare_test_data()
    
    return evaluator.evaluate(detailed=detailed)


def list_available_models() -> List[str]:
    """사용 가능한 모델 목록 반환"""
    
    model_files = glob.glob(os.path.join(Config.MODEL_DIR, "*.keras"))
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in model_files]
    
    return sorted(model_names)


if __name__ == "__main__":
    # 평가 테스트
    print("🧪 CSI 모델 평가기 테스트")
    print("=" * 50)
    
    # 사용 가능한 모델 확인
    available_models = list_available_models()
    
    if not available_models:
        print("❌ 평가할 모델이 없습니다.")
        print("   먼저 모델을 학습하세요: python main.py --mode train")
    else:
        print(f"📂 사용 가능한 모델: {len(available_models)}개")
        for i, model_name in enumerate(available_models, 1):
            print(f"   {i}. {model_name}")
        
        # 사용자 모델 선택
        try:
            choice = input(f"\n평가할 모델 번호를 선택하세요 (1-{len(available_models)}): ").strip()
            model_idx = int(choice) - 1
            
            if 0 <= model_idx < len(available_models):
                selected_model = available_models[model_idx]
                print(f"\n🎯 선택된 모델: {selected_model}")
                
                # 평가 실행
                results = evaluate_saved_model(selected_model, detailed=True)
                
                print(f"\n✅ 평가 완료!")
                
            else:
                print("❌ 잘못된 선택입니다.")
                
        except (ValueError, KeyboardInterrupt):
            print("❌ 평가를 취소했습니다.")
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            import traceback
            traceback.print_exc()
