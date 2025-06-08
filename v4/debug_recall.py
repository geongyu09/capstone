"""
Recall 문제 디버깅 스크립트
낙상 감지 성능이 0에 가까운 문제를 분석합니다.
(디버깅만을 위한 파일. 제거 예정)
"""

import numpy as np
import pandas as pd
import glob
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data_generator import create_data_generators
from utils import setup_logging

def analyze_data_distribution():
    """데이터 분포 상세 분석"""
    print("🔍 데이터 분포 상세 분석")
    print("=" * 50)
    
    # 전처리된 파일들 확인
    processed_files = glob.glob(f"{Config.PROCESSED_DATA_DIR}/*_processed.csv")
    
    if not processed_files:
        print("❌ 전처리된 파일이 없습니다!")
        return
    
    print(f"📁 전처리된 파일: {len(processed_files)}개")
    
    total_samples = 0
    total_fall_samples = 0
    file_stats = []
    
    for file_path in processed_files[:5]:  # 처음 5개 파일만 분석
        try:
            df = pd.read_csv(file_path)
            
            if 'label' in df.columns:
                labels = df['label'].values
                fall_count = np.sum(labels == 1)
                normal_count = np.sum(labels == 0)
                fall_ratio = fall_count / len(labels) if len(labels) > 0 else 0
                
                file_stats.append({
                    'file': file_path.split('/')[-1],
                    'total': len(labels),
                    'fall': fall_count,
                    'normal': normal_count,
                    'fall_ratio': fall_ratio
                })
                
                total_samples += len(labels)
                total_fall_samples += fall_count
                
                print(f"📄 {file_path.split('/')[-1]}:")
                print(f"   총 샘플: {len(labels):,}")
                print(f"   낙상: {fall_count:,} ({fall_ratio*100:.1f}%)")
                print(f"   정상: {normal_count:,} ({(1-fall_ratio)*100:.1f}%)")
            else:
                print(f"⚠️ {file_path}: 라벨 컬럼이 없습니다!")
                
        except Exception as e:
            print(f"❌ {file_path} 분석 실패: {e}")
    
    if total_samples > 0:
        overall_fall_ratio = total_fall_samples / total_samples
        print(f"\n📊 전체 통계:")
        print(f"   총 샘플: {total_samples:,}")
        print(f"   낙상 샘플: {total_fall_samples:,} ({overall_fall_ratio*100:.2f}%)")
        print(f"   정상 샘플: {total_samples-total_fall_samples:,} ({(1-overall_fall_ratio)*100:.2f}%)")
        
        # 클래스 불균형 정도
        imbalance_ratio = (total_samples - total_fall_samples) / total_fall_samples if total_fall_samples > 0 else float('inf')
        print(f"   불균형 비율: 1:{imbalance_ratio:.1f} (정상:낙상)")
        
        if overall_fall_ratio < 0.05:
            print("🚨 심각한 클래스 불균형! 낙상 데이터가 5% 미만입니다.")
        elif overall_fall_ratio < 0.1:
            print("⚠️ 클래스 불균형 문제 있음. 낙상 데이터가 10% 미만입니다.")
    
    return file_stats

def analyze_sequence_labeling():
    """시퀀스 라벨링 분석"""
    print(f"\n🔍 시퀀스 라벨링 분석")
    print("=" * 50)
    
    print(f"현재 설정:")
    print(f"   WINDOW_SIZE: {Config.WINDOW_SIZE}")
    print(f"   STRIDE: {Config.STRIDE}")
    print(f"   OVERLAP_THRESHOLD: {Config.OVERLAP_THRESHOLD}")
    
    try:
        # 데이터 제너레이터로 실제 시퀀스 라벨 확인
        train_gen, val_gen, test_gen = create_data_generators()
        
        print(f"\n📊 제너레이터 통계:")
        print(f"   훈련 시퀀스: {train_gen.total_sequences:,}개")
        print(f"   검증 시퀀스: {val_gen.total_sequences:,}개")
        print(f"   테스트 시퀀스: {test_gen.total_sequences:,}개")
        
        # 샘플 배치로 실제 라벨 분포 확인
        print(f"\n🔍 실제 시퀀스 라벨 분포 확인 (샘플링):")
        
        sample_labels = []
        for i in range(min(10, len(train_gen))):  # 처음 10개 배치 확인
            X_batch, y_batch = train_gen[i]
            sample_labels.extend(y_batch)
            
            fall_count = np.sum(y_batch == 1)
            batch_fall_ratio = fall_count / len(y_batch) if len(y_batch) > 0 else 0
            
            print(f"   배치 {i+1}: 낙상 {fall_count}/{len(y_batch)} ({batch_fall_ratio*100:.1f}%)")
        
        # 전체 샘플 라벨 분포
        if sample_labels:
            sample_labels = np.array(sample_labels)
            fall_count = np.sum(sample_labels == 1)
            total_count = len(sample_labels)
            fall_ratio = fall_count / total_count
            
            print(f"\n📈 샘플 시퀀스 전체 분포:")
            print(f"   총 시퀀스: {total_count}")
            print(f"   낙상 시퀀스: {fall_count} ({fall_ratio*100:.2f}%)")
            print(f"   정상 시퀀스: {total_count-fall_count} ({(1-fall_ratio)*100:.2f}%)")
            
            if fall_ratio < 0.01:
                print("🚨 치명적! 낙상 시퀀스가 1% 미만입니다!")
                return "critical"
            elif fall_ratio < 0.05:
                print("🚨 심각한 문제! 낙상 시퀀스가 5% 미만입니다!")
                return "severe"
            elif fall_ratio < 0.1:
                print("⚠️ 문제 있음! 낙상 시퀀스가 10% 미만입니다!")
                return "moderate"
            else:
                print("✅ 라벨 분포는 양호합니다.")
                return "good"
        
    except Exception as e:
        print(f"❌ 시퀀스 분석 실패: {e}")
        return "error"

def suggest_fixes(severity):
    """문제 해결 방안 제시"""
    print(f"\n💡 문제 해결 방안")
    print("=" * 50)
    
    if severity in ["critical", "severe"]:
        print("🚨 긴급 수정 필요:")
        print("1. OVERLAP_THRESHOLD 낮추기:")
        print("   Config.OVERLAP_THRESHOLD = 0.1  # 0.3 → 0.1")
        print()
        print("2. 더 강한 클래스 가중치:")
        print("   class_weights = {0: 1.0, 1: 20.0}  # 낙상에 20배 가중치")
        print()
        print("3. 데이터 증강:")
        print("   - 낙상 시퀀스 복제")
        print("   - 시간 워핑")
        print("   - 노이즈 추가")
        
    elif severity == "moderate":
        print("⚠️ 권장 수정사항:")
        print("1. OVERLAP_THRESHOLD 조정:")
        print("   Config.OVERLAP_THRESHOLD = 0.2  # 0.3 → 0.2")
        print()
        print("2. 클래스 가중치 강화:")
        print("   class_weights = {0: 1.0, 1: 10.0}")
        print()
        print("3. Focal Loss 파라미터 조정:")
        print("   FocalLoss(alpha=0.75, gamma=3.0)  # alpha 증가, gamma 증가")
    
    else:
        print("✅ 현재 설정으로도 학습 가능하지만 다음을 고려:")
        print("1. 학습률 조정")
        print("2. 에포크 증가")
        print("3. 정규화 줄이기")

def create_quick_fix_config():
    """빠른 수정을 위한 설정 파일 생성"""
    print(f"\n🛠️ 빠른 수정 설정 파일 생성")
    
    fix_config = """
# quick_fix_config.py
# Recall 문제 해결을 위한 수정된 설정

# 시퀀스 라벨링 완화
OVERLAP_THRESHOLD = 0.1  # 0.3 → 0.1로 낮춤

# 강화된 클래스 가중치
CLASS_WEIGHTS = {
    0: 1.0,   # 정상
    1: 15.0   # 낙상 (15배 가중치)
}

# Focal Loss 파라미터
FOCAL_LOSS_ALPHA = 0.75  # 낙상 클래스에 더 집중
FOCAL_LOSS_GAMMA = 3.0   # 어려운 샘플에 더 집중

# 학습 설정
LEARNING_RATE = 0.0005   # 낮은 학습률
PATIENCE = 20           # 더 긴 patience
EPOCHS = 60            # 더 많은 에포크

# 배치 크기 (필요시 줄이기)
BATCH_SIZE = 16        # 32 → 16
"""
    
    with open("quick_fix_config.py", "w", encoding="utf-8") as f:
        f.write(fix_config)
    
    print("✅ quick_fix_config.py 생성 완료")

def main():
    """메인 디버깅 함수"""
    print("🔍 CSI 낙상 감지 Recall 문제 디버깅")
    print("=" * 60)
    
    # 1. 데이터 분포 분석
    file_stats = analyze_data_distribution()
    
    # 2. 시퀀스 라벨링 분석
    severity = analyze_sequence_labeling()
    
    # 3. 해결 방안 제시
    suggest_fixes(severity)
    
    # 4. 빠른 수정 설정 생성
    create_quick_fix_config()
    
    print(f"\n📋 다음 단계:")
    print("1. quick_fix_config.py의 설정을 config.py에 적용")
    print("2. train_improved.py 수정하여 새 설정 사용")
    print("3. 작은 배치로 테스트 학습 실행")
    print("4. Recall 개선 확인")

if __name__ == "__main__":
    main()
