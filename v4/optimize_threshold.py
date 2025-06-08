"""
임계값 최적화 스크립트
현재 모델의 최적 임계값을 찾아 성능을 개선합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score

from evaluator import evaluate_saved_model
from utils import setup_logging

def find_optimal_threshold(model_name="hybrid_20250607_143159_best"):
    """최적 임계값 찾기"""
    
    logger = setup_logging()
    logger.info("🎯 임계값 최적화 시작")
    
    # 모델 평가 실행하여 예측 결과 얻기
    results = evaluate_saved_model(model_name, detailed=True)
    
    if 'predictions' not in results:
        logger.error("예측 결과를 찾을 수 없습니다.")
        return None
    
    y_true = np.array(results['predictions']['y_true'])
    y_scores = np.array(results['predictions']['y_scores'])
    
    logger.info(f"데이터 크기: {len(y_true)}개")
    logger.info(f"낙상 비율: {np.mean(y_true):.3f}")
    
    # 다양한 임계값 테스트
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    metrics = {
        'threshold': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'tn': [],
        'fp': [],
        'fn': [],
        'tp': []
    }
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # 메트릭 계산
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 혼동 행렬
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 결과 저장
        metrics['threshold'].append(threshold)
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['tn'].append(tn)
        metrics['fp'].append(fp)
        metrics['fn'].append(fn)
        metrics['tp'].append(tp)
        
        logger.info(f"임계값 {threshold:.2f}: F1={f1:.3f}, Acc={accuracy:.3f}, "
                   f"Prec={precision:.3f}, Rec={recall:.3f}")
    
    # 최적 임계값 찾기
    best_f1_idx = np.argmax(metrics['f1'])
    best_threshold = metrics['threshold'][best_f1_idx]
    best_f1 = metrics['f1'][best_f1_idx]
    
    logger.info(f"🏆 최적 임계값: {best_threshold:.2f} (F1: {best_f1:.3f})")
    
    # 시각화
    plot_threshold_analysis(metrics)
    
    # 최적 임계값에서의 상세 결과
    print_detailed_results(metrics, best_f1_idx)
    
    return best_threshold, metrics

def plot_threshold_analysis(metrics):
    """임계값 분석 결과 시각화"""
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(15, 10))
    
    # 1. 메트릭 변화
    plt.subplot(2, 3, 1)
    plt.plot(metrics['threshold'], metrics['accuracy'], 'b-o', label='Accuracy')
    plt.plot(metrics['threshold'], metrics['precision'], 'r-s', label='Precision')
    plt.plot(metrics['threshold'], metrics['recall'], 'g-^', label='Recall')
    plt.plot(metrics['threshold'], metrics['f1'], 'k-d', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance vs Threshold')
    plt.legend()
    plt.grid(True)
    
    # 2. F1 점수 상세
    plt.subplot(2, 3, 2)
    plt.plot(metrics['threshold'], metrics['f1'], 'k-o', linewidth=2)
    best_idx = np.argmax(metrics['f1'])
    plt.plot(metrics['threshold'][best_idx], metrics['f1'][best_idx], 
             'ro', markersize=10, label=f'Best: {metrics["threshold"][best_idx]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Optimization')
    plt.legend()
    plt.grid(True)
    
    # 3. 정밀도-재현율 트레이드오프
    plt.subplot(2, 3, 3)
    plt.plot(metrics['recall'], metrics['precision'], 'b-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    # 4. 혼동 행렬 요소
    plt.subplot(2, 3, 4)
    plt.plot(metrics['threshold'], metrics['tp'], 'g-o', label='True Positive')
    plt.plot(metrics['threshold'], metrics['fp'], 'r-s', label='False Positive')
    plt.plot(metrics['threshold'], metrics['fn'], 'orange', marker='^', label='False Negative')
    plt.plot(metrics['threshold'], metrics['tn'], 'b-d', label='True Negative')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Elements')
    plt.legend()
    plt.grid(True)
    
    # 5. 거짓 양성률 vs 재현율
    plt.subplot(2, 3, 5)
    fpr = np.array(metrics['fp']) / (np.array(metrics['fp']) + np.array(metrics['tn']))
    plt.plot(fpr, metrics['recall'], 'purple', marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Space')
    plt.grid(True)
    
    # 6. 임계값별 성능 히트맵
    plt.subplot(2, 3, 6)
    performance_matrix = np.array([
        metrics['accuracy'],
        metrics['precision'], 
        metrics['recall'],
        metrics['f1']
    ])
    
    im = plt.imshow(performance_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.yticks([0, 1, 2, 3], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.xlabel('Threshold Index')
    plt.title('Performance Heatmap')
    
    plt.tight_layout()
    plt.savefig('./logs/threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(metrics, best_idx):
    """최적 임계값에서의 상세 결과 출력"""
    
    print("\n" + "="*60)
    print("🎯 최적 임계값 분석 결과")
    print("="*60)
    
    threshold = metrics['threshold'][best_idx]
    accuracy = metrics['accuracy'][best_idx]
    precision = metrics['precision'][best_idx]
    recall = metrics['recall'][best_idx]
    f1 = metrics['f1'][best_idx]
    
    tn = metrics['tn'][best_idx]
    fp = metrics['fp'][best_idx]
    fn = metrics['fn'][best_idx]
    tp = metrics['tp'][best_idx]
    
    print(f"📊 최적 임계값: {threshold:.3f}")
    print(f"")
    print(f"🎯 성능 지표:")
    print(f"   정확도 (Accuracy): {accuracy:.3f}")
    print(f"   정밀도 (Precision): {precision:.3f}")
    print(f"   재현율 (Recall): {recall:.3f}")
    print(f"   F1 점수: {f1:.3f}")
    print(f"")
    print(f"📋 혼동 행렬:")
    print(f"                  예측")
    print(f"              정상    낙상")
    print(f"   실제 정상   {tn:4d}   {fp:4d}")
    print(f"        낙상   {fn:4d}   {tp:4d}")
    print(f"")
    print(f"💡 해석:")
    print(f"   - 정상을 낙상으로 오분류: {fp}건 ({fp/(tn+fp)*100:.1f}%)")
    print(f"   - 낙상을 정상으로 오분류: {fn}건 ({fn/(fn+tp)*100:.1f}%)")
    print(f"   - 낙상 감지 성공률: {tp}건 ({tp/(fn+tp)*100:.1f}%)")
    
    # 개선 방향 제시
    print(f"")
    print(f"🚀 개선 방향:")
    if precision < 0.5:
        print(f"   ⚠️  정밀도가 낮음 → 거짓 양성 감소 필요")
    if recall < 0.7:
        print(f"   ⚠️  재현율이 낮음 → 놓치는 낙상 감소 필요")
    if f1 < 0.6:
        print(f"   ⚠️  전체적인 성능 개선 필요")
    
    print("="*60)

def apply_optimal_threshold(optimal_threshold):
    """최적 임계값을 설정 파일에 저장"""
    
    try:
        # config.py 업데이트
        config_path = "./config.py"
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # FALL_THRESHOLD 값 업데이트
        updated_content = content.replace(
            'FALL_THRESHOLD = 0.5',
            f'FALL_THRESHOLD = {optimal_threshold:.3f}'
        )
        
        with open(config_path, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ 설정 파일 업데이트 완료: FALL_THRESHOLD = {optimal_threshold:.3f}")
        
        # 실시간 감지기용 설정 파일도 생성
        threshold_config = {
            'optimal_threshold': optimal_threshold,
            'model_name': 'hybrid_20250607_143159_best',
            'updated_time': str(np.datetime64('now'))
        }
        
        import json
        with open('./models/optimal_threshold.json', 'w') as f:
            json.dump(threshold_config, f, indent=2)
        
        print("✅ 임계값 설정 파일 생성 완료: ./models/optimal_threshold.json")
        
    except Exception as e:
        print(f"❌ 설정 파일 업데이트 실패: {e}")

def compare_thresholds():
    """현재 임계값 vs 최적 임계값 비교"""
    
    model_name = "hybrid_20250607_143159_best"
    current_threshold = 0.5  # 현재 사용중인 임계값
    
    # 최적 임계값 찾기
    optimal_threshold, metrics = find_optimal_threshold(model_name)
    
    if optimal_threshold is None:
        return
    
    # 비교 결과 출력
    print("\n" + "="*70)
    print("📊 임계값 비교 분석")
    print("="*70)
    
    # 현재 임계값에서의 성능
    current_idx = None
    for i, t in enumerate(metrics['threshold']):
        if abs(t - current_threshold) < 0.01:
            current_idx = i
            break
    
    if current_idx is not None:
        optimal_idx = np.argmax(metrics['f1'])
        
        print(f"🔹 현재 임계값 ({current_threshold:.2f}):")
        print(f"   F1: {metrics['f1'][current_idx]:.3f}")
        print(f"   정확도: {metrics['accuracy'][current_idx]:.3f}")
        print(f"   정밀도: {metrics['precision'][current_idx]:.3f}")
        print(f"   재현율: {metrics['recall'][current_idx]:.3f}")
        
        print(f"\n🏆 최적 임계값 ({optimal_threshold:.2f}):")
        print(f"   F1: {metrics['f1'][optimal_idx]:.3f}")
        print(f"   정확도: {metrics['accuracy'][optimal_idx]:.3f}")
        print(f"   정밀도: {metrics['precision'][optimal_idx]:.3f}")
        print(f"   재현율: {metrics['recall'][optimal_idx]:.3f}")
        
        # 개선량 계산
        f1_improvement = metrics['f1'][optimal_idx] - metrics['f1'][current_idx]
        acc_improvement = metrics['accuracy'][optimal_idx] - metrics['accuracy'][current_idx]
        
        print(f"\n📈 예상 개선량:")
        print(f"   F1 점수: +{f1_improvement:.3f} ({f1_improvement/metrics['f1'][current_idx]*100:+.1f}%)")
        print(f"   정확도: +{acc_improvement:.3f} ({acc_improvement/metrics['accuracy'][current_idx]*100:+.1f}%)")
        
        # 적용 여부 확인
        if f1_improvement > 0.01:  # 1% 이상 개선되는 경우
            response = input(f"\n🤔 최적 임계값을 적용하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                apply_optimal_threshold(optimal_threshold)
            else:
                print("임계값 변경을 취소했습니다.")
        else:
            print("💡 현재 임계값이 이미 충분히 최적화되어 있습니다.")
    
    print("="*70)

if __name__ == "__main__":
    print("🎯 CSI 낙상 감지 임계값 최적화")
    print("=" * 50)
    
    try:
        # 임계값 비교 및 최적화
        compare_thresholds()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
