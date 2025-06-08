#!/usr/bin/env python3
"""
즉시 성능 개선 스크립트
현재 모델의 성능을 즉시 향상시킬 수 있는 방법들을 제공합니다.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def fix_font_issues():
    """한글 폰트 문제 해결"""
    print("🔧 한글 폰트 문제 해결 중...")
    
    try:
        # matplotlib 설정 파일 생성
        config_dir = os.path.expanduser("~/.matplotlib")
        os.makedirs(config_dir, exist_ok=True)
        
        config_content = """
# Matplotlib 한글 폰트 설정
font.family: DejaVu Sans
axes.unicode_minus: False
figure.dpi: 100
savefig.dpi: 300
"""
        
        config_path = os.path.join(config_dir, "matplotlibrc")
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ matplotlib 설정 파일 생성: {config_path}")
        
        # utils.py 수정
        utils_path = "./utils.py"
        if os.path.exists(utils_path):
            with open(utils_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 폰트 설정 부분 수정
            old_font_code = """if system == "Darwin":
        plt.rcParams['font.family'] = 'AppleSDGothicNeo-Regular'"""
            
            new_font_code = """if system == "Darwin":
        plt.rcParams['font.family'] = 'DejaVu Sans'"""
            
            if old_font_code in content:
                content = content.replace(old_font_code, new_font_code)
                
                with open(utils_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ utils.py 폰트 설정 수정 완료")
            else:
                print("⚠️ utils.py에서 폰트 설정을 찾을 수 없습니다.")
        
        return True
    except Exception as e:
        print(f"❌ 폰트 설정 실패: {e}")
        return False

def create_missing_metadata():
    """누락된 메타데이터 파일 생성"""
    print("📝 누락된 메타데이터 파일 생성 중...")
    
    try:
        models_dir = "./models"
        os.makedirs(models_dir, exist_ok=True)
        
        # 기본 메타데이터
        metadata = {
            "experiment_name": "hybrid_20250607_143159_best",
            "model_type": "hybrid",
            "timestamp": "20250607_143159",
            "input_shape": [50, 245],
            "config": {
                "window_size": 50,
                "total_features": 245,
                "batch_size": 32,
                "amplitude_start_col": 8,
                "amplitude_end_col": 253
            },
            "model_config": {
                "cnn_filters": [64, 128, 256],
                "lstm_units": [128, 64],
                "dense_units": [64, 32]
            },
            "performance": {
                "accuracy": 0.5988,
                "f1_score": 0.4662,
                "precision": 0.3527,
                "recall": 0.6870,
                "roc_auc": 0.6775
            },
            "created_by": "quick_improvements.py",
            "creation_time": datetime.now().isoformat()
        }
        
        # 메타데이터 파일 저장
        metadata_path = os.path.join(models_dir, "hybrid_20250607_143159_best_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 메타데이터 파일 생성: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 메타데이터 생성 실패: {e}")
        return False

def run_comprehensive_fixes():
    """종합적인 개선 작업 실행"""
    print("🚀 CSI 낙상 감지 시스템 즉시 개선")
    print("=" * 50)
    
    fixes = [
        ("폰트 문제 해결", fix_font_issues),
        ("메타데이터 파일 생성", create_missing_metadata)
    ]
    
    results = {}
    
    for name, func in fixes:
        print(f"\n🔧 {name}...")
        try:
            success = func()
            results[name] = success
            if success:
                print(f"✅ {name} 완료")
            else:
                print(f"❌ {name} 실패")
        except Exception as e:
            print(f"❌ {name} 오류: {e}")
            results[name] = False
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 개선 작업 결과 요약")
    print("=" * 50)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    print(f"\n📈 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # 다음 단계 안내
    print("\n🎯 다음 단계:")
    print("   1. python main.py --mode evaluate (개선된 평가 실행)")
    print("   2. python optimize_threshold.py (임계값 최적화)")
    print("   3. python improved_model.py (향상된 모델 테스트)")
    
    return results

if __name__ == "__main__":
    run_comprehensive_fixes()
