"""
multi_file_trainer로 학습한 모델 테스트하는 코드드
"""


# Python 코드 실행 전에 추가
import matplotlib.pyplot as plt
import platform

# Windows
if platform.system() == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False


# direct_test.py
from model_tester import CSIModelTester

def run_direct_test():
    """찾은 파일로 직접 테스트"""
    print("🚀 직접 모델 테스트 실행")
    print("=" * 40)
    
    # 추천 파일들 (크기가 큰 순서)
    test_files = [
        # "../csi_data/case1/12_labeled.csv",
        # "../csi_data/case2/11_labeled.csv", 
        # "../csi_data/case3/10_labeled.csv",
        "../csi_data/non/1.csv"  # 정상 데이터
    ]
    
    print("🎯 테스트할 파일들:")
    for i, file_path in enumerate(test_files):
        import os
        exists = "✅" if os.path.exists(file_path) else "❌"
        try:
            size = os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
            print(f"   {i+1}. {exists} {file_path} ({size:.1f} KB)")
        except:
            print(f"   {i+1}. {exists} {file_path}")
    
    # 첫 번째 존재하는 파일로 테스트
    tester = CSIModelTester()
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n🔥 테스트 파일: {file_path}")
            
            try:
                result = tester.quick_test(file_path)
                if result:
                    print("✅ 테스트 성공!")
                    break
                else:
                    print("❌ 테스트 실패, 다음 파일 시도...")
                    
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                print("다음 파일로 시도...")
                continue
    else:
        print("❌ 모든 파일 테스트 실패")

def test_multiple_files():
    """여러 파일로 연속 테스트"""
    print("🔄 다중 파일 테스트")
    print("=" * 30)
    
    test_files = [
        ("../csi_data/case1/12_labeled.csv", "Case1 - 낙상 데이터"),
        ("../csi_data/case2/11_labeled.csv", "Case2 - 낙상 데이터"),
        ("../csi_data/case3/10_labeled.csv", "Case3 - 낙상 데이터"),
        ("../csi_data/non/1.csv", "Non - 정상 데이터")
    ]
    
    tester = CSIModelTester()
    
    # 첫 번째 파일로 모델 로드
    model_path = tester.find_latest_model()
    if not model_path or not tester.load_model_and_preprocessors(model_path):
        print("❌ 모델 로드 실패")
        return
    
    results = []
    
    for file_path, description in test_files:
        if os.path.exists(file_path):
            print(f"\n📄 테스트: {description}")
            print(f"   파일: {os.path.basename(file_path)}")
            
            try:
                pred_probs, pred_labels, true_labels = tester.test_on_csv(file_path)
                
                if pred_probs is not None:
                    fall_predictions = np.sum(pred_labels == 1)
                    max_prob = np.max(pred_probs)
                    avg_prob = np.mean(pred_probs)
                    
                    result = {
                        'file': description,
                        'fall_predictions': fall_predictions,
                        'max_prob': max_prob,
                        'avg_prob': avg_prob,
                        'total_sequences': len(pred_probs)
                    }
                    
                    results.append(result)
                    
                    print(f"   결과: {fall_predictions}개 낙상 예측 (최대 확률: {max_prob:.3f})")
                    
                    if max_prob > 0.8:
                        print(f"   🚨 강한 낙상 신호!")
                    elif max_prob > 0.5:
                        print(f"   ⚠️ 낙상 가능성 있음")
                    else:
                        print(f"   ✅ 정상으로 판단")
                
            except Exception as e:
                print(f"   ❌ 테스트 실패: {e}")
    
    # 결과 요약
    if results:
        print(f"\n📊 전체 테스트 요약:")
        print(f"{'파일':<20} {'낙상예측':<8} {'최대확률':<8} {'평균확률':<8}")
        print("-" * 50)
        
        for result in results:
            print(f"{result['file']:<20} {result['fall_predictions']:<8} "
                  f"{result['max_prob']:<8.3f} {result['avg_prob']:<8.3f}")

if __name__ == "__main__":
    import sys
    import numpy as np
    import os
    
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        test_multiple_files()
    else:
        run_direct_test()