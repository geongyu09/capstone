"""
CSI 낙상 감지 v4 유틸리티 함수들
"""

import os
import glob
import logging
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명에 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"csi_fall_detection_{timestamp}.log")
    
    # 로거 설정
    logger = logging.getLogger("CSIFallDetection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 핸들러가 이미 있다면 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"로깅 시작 - 로그 파일: {log_file}")
    return logger


def collect_csv_files(data_paths: List[str], pattern: str = "*.csv") -> List[str]:
    """지정된 경로들에서 CSV 파일 수집"""
    csv_files = []
    
    for path in data_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, pattern))
            csv_files.extend(files)
        else:
            print(f"⚠️ 경로를 찾을 수 없습니다: {path}")
    
    return csv_files


def analyze_data_distribution(csv_files: List[str]) -> Dict[str, Any]:
    """데이터 분포 분석"""
    print("📊 데이터 분포 분석 중...")
    
    stats = {
        'total_files': len(csv_files),
        'file_sizes': [],
        'sample_counts': [],
        'label_distributions': {},
        'amplitude_stats': {},
        'file_info': []
    }
    
    for file_path in csv_files:
        try:
            # 파일 크기
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            stats['file_sizes'].append(file_size)
            
            # 데이터 로드 (샘플링)
            df = pd.read_csv(file_path)
            sample_count = len(df)
            stats['sample_counts'].append(sample_count)
            
            # 라벨 분포 (있는 경우)
            if 'label' in df.columns:
                label_dist = df['label'].value_counts().to_dict()
                file_name = os.path.basename(file_path)
                stats['label_distributions'][file_name] = label_dist
            
            # Amplitude 데이터 통계 (샘플링)
            if len(df.columns) > 253:
                amplitude_data = df.iloc[:100, 8:253]  # 처음 100행만 샘플링
                amp_stats = {
                    'mean': amplitude_data.values.mean(),
                    'std': amplitude_data.values.std(),
                    'min': amplitude_data.values.min(),
                    'max': amplitude_data.values.max()
                }
                stats['amplitude_stats'][os.path.basename(file_path)] = amp_stats
            
            # 파일 정보
            stats['file_info'].append({
                'file': os.path.basename(file_path),
                'size_mb': file_size,
                'samples': sample_count,
                'columns': len(df.columns)
            })
            
        except Exception as e:
            print(f"⚠️ 파일 분석 실패: {file_path} - {e}")
    
    # 전체 통계 계산
    if stats['file_sizes']:
        stats['avg_file_size_mb'] = np.mean(stats['file_sizes'])
        stats['total_samples'] = sum(stats['sample_counts'])
        stats['avg_samples_per_file'] = np.mean(stats['sample_counts'])
    
    return stats


def print_data_analysis(stats: Dict[str, Any]) -> None:
    """데이터 분석 결과 출력"""
    print("\n📈 데이터 분석 결과")
    print("=" * 50)
    
    print(f"📁 파일 정보:")
    print(f"   - 총 파일 수: {stats['total_files']:,}개")
    if stats.get('avg_file_size_mb'):
        print(f"   - 평균 파일 크기: {stats['avg_file_size_mb']:.1f} MB")
        print(f"   - 총 샘플 수: {stats['total_samples']:,}개")
        print(f"   - 파일당 평균 샘플: {stats['avg_samples_per_file']:.0f}개")
    
    # 라벨 분포
    if stats['label_distributions']:
        print(f"\n🏷️ 라벨 분포:")
        total_fall = 0
        total_normal = 0
        for file_name, dist in stats['label_distributions'].items():
            fall_count = dist.get(1, 0)
            normal_count = dist.get(0, 0)
            total_fall += fall_count
            total_normal += normal_count
        
        total_labeled = total_fall + total_normal
        if total_labeled > 0:
            print(f"   - 낙상 샘플: {total_fall:,}개 ({total_fall/total_labeled*100:.1f}%)")
            print(f"   - 정상 샘플: {total_normal:,}개 ({total_normal/total_labeled*100:.1f}%)")
    
    # Amplitude 데이터 통계
    if stats['amplitude_stats']:
        print(f"\n📊 Amplitude 데이터 통계:")
        all_means = [stat['mean'] for stat in stats['amplitude_stats'].values()]
        all_stds = [stat['std'] for stat in stats['amplitude_stats'].values()]
        all_mins = [stat['min'] for stat in stats['amplitude_stats'].values()]
        all_maxs = [stat['max'] for stat in stats['amplitude_stats'].values()]
        
        print(f"   - 평균값 범위: {np.min(all_means):.3f} ~ {np.max(all_means):.3f}")
        print(f"   - 표준편차 범위: {np.min(all_stds):.3f} ~ {np.max(all_stds):.3f}")
        print(f"   - 최소값 범위: {np.min(all_mins):.3f} ~ {np.max(all_mins):.3f}")
        print(f"   - 최대값 범위: {np.min(all_maxs):.3f} ~ {np.max(all_maxs):.3f}")


def save_metadata(metadata: Dict[str, Any], file_path: str) -> None:
    """메타데이터 저장"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def load_metadata(file_path: str) -> Dict[str, Any]:
    """메타데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model_artifacts(model, scaler, metadata: Dict[str, Any], 
                        model_dir: str, model_name: str) -> Dict[str, str]:
    """모델 관련 파일들 저장"""
    os.makedirs(model_dir, exist_ok=True)
    
    # 파일 경로들
    paths = {
        'model': os.path.join(model_dir, f"{model_name}.keras"),
        'scaler': os.path.join(model_dir, f"{model_name}_scaler.pkl"),
        'metadata': os.path.join(model_dir, f"{model_name}_metadata.json")
    }
    
    # 모델 저장
    model.save(paths['model'])
    
    # 스케일러 저장
    with open(paths['scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    
    # 메타데이터 저장
    save_metadata(metadata, paths['metadata'])
    
    return paths


def load_model_artifacts(model_dir: str, model_name: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """모델 관련 파일들 로드"""
    from tensorflow.keras.models import load_model
    
    paths = {
        'model': os.path.join(model_dir, f"{model_name}.keras"),
        'scaler': os.path.join(model_dir, f"{model_name}_scaler.pkl"),
        'metadata': os.path.join(model_dir, f"{model_name}_metadata.json")
    }
    
    # 모델 로드
    model = load_model(paths['model'])
    
    # 스케일러 로드
    with open(paths['scaler'], 'rb') as f:
        scaler = pickle.load(f)
    
    # 메타데이터 로드
    metadata = load_metadata(paths['metadata'])
    
    return model, scaler, metadata


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """학습 히스토리 플롯"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    if 'accuracy' in history:
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history:
        axes[1, 0].plot(history['precision'], label='Training Precision')
        if 'val_precision' in history:
            axes[1, 0].plot(history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history:
        axes[1, 1].plot(history['recall'], label='Training Recall')
        if 'val_recall' in history:
            axes[1, 1].plot(history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 학습 히스토리 저장: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: List[str] = None,
                         save_path: Optional[str] = None) -> None:
    """혼동 행렬 플롯"""
    if labels is None:
        labels = ['Normal', 'Fall']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 혼동 행렬 저장: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                  save_path: Optional[str] = None) -> float:
    """ROC 커브 플롯"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC 커브 저장: {save_path}")
    
    plt.show()
    
    return roc_auc


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                  threshold: float = 0.5) -> Dict[str, Any]:
    """모델 평가"""
    # 예측
    y_scores = model.predict(X_test)
    y_pred = (y_scores > threshold).astype(int)
    
    # 분류 리포트
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC
    roc_auc = auc(*roc_curve(y_test, y_scores)[:2])
    
    # 결과 정리
    results = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'threshold': threshold,
        'predictions': {
            'y_true': y_test.tolist(),
            'y_scores': y_scores.flatten().tolist(),
            'y_pred': y_pred.flatten().tolist()
        }
    }
    
    return results


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """평가 결과 출력"""
    print("\n📊 모델 평가 결과")
    print("=" * 50)
    
    print(f"🎯 전체 성능:")
    print(f"   - 정확도: {results['accuracy']:.4f}")
    print(f"   - ROC AUC: {results['roc_auc']:.4f}")
    
    print(f"\n🔍 낙상 감지 성능:")
    print(f"   - 정밀도: {results['precision']:.4f}")
    print(f"   - 재현율: {results['recall']:.4f}")
    print(f"   - F1 점수: {results['f1_score']:.4f}")
    
    print(f"\n📋 혼동 행렬:")
    cm = np.array(results['confusion_matrix'])
    print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    print(f"\n⚙️ 임계값: {results['threshold']}")


def create_timestamp() -> str:
    """타임스탬프 생성"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """클래스 가중치 계산"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def split_data_by_files(file_paths: List[str], 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """파일 단위로 데이터 분할"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "비율의 합이 1이 되어야 합니다."
    
    np.random.seed(random_seed)
    
    # 파일들을 셔플
    shuffled_files = file_paths.copy()
    np.random.shuffle(shuffled_files)
    
    n_files = len(shuffled_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]
    
    return train_files, val_files, test_files


def memory_usage_mb() -> float:
    """현재 메모리 사용량 (MB) 반환"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def format_time(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    print("🧪 유틸리티 함수 테스트")
    print("=" * 40)
    
    # 로깅 테스트
    logger = setup_logging()
    logger.info("로깅 테스트 메시지")
    
    # 타임스탬프 테스트
    timestamp = create_timestamp()
    print(f"현재 타임스탬프: {timestamp}")
    
    # 메모리 사용량 테스트
    memory = memory_usage_mb()
    print(f"현재 메모리 사용량: {memory:.1f} MB")
    
    # 시간 포맷 테스트
    test_seconds = 3661
    formatted_time = format_time(test_seconds)
    print(f"시간 포맷 테스트: {test_seconds}초 -> {formatted_time}")
    
    print("✅ 유틸리티 함수 테스트 완료!")
