"""
CSI 낙상 감지 v4 - 간편 실행 스크립트
"""

import os
import sys

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import setup_logging


def quick_start():
    """빠른 시작 가이드"""
    print("🚀 CSI 낙상 감지 v4 - 빠른 시작")
    print("=" * 50)
    
    while True:
        print(f"\n📋 작업 선택:")
        print(f"1. 📊 데이터 구조 확인")
        print(f"2. 🔧 데이터 전처리")
        print(f"3. 🤖 모델 학습")
        print(f"4. 📈 모델 평가")
        print(f"5. ⚙️ 설정 확인")
        print(f"6. 🔄 전체 파이프라인 (전처리 + 학습 + 평가)")
        print(f"7. 🧪 테스트 모드")
        print(f"8. 📁 프로젝트 정보")
        print(f"9. ❌ 종료")
        
        choice = input("\n선택하세요 (1-9): ").strip()
        
        if choice == '1':
            check_data_structure()
        elif choice == '2':
            run_preprocessing()
        elif choice == '3':
            run_training()
        elif choice == '4':
            run_evaluation()
        elif choice == '5':
            show_config()
        elif choice == '6':
            run_full_pipeline()
        elif choice == '7':
            run_test_mode()
        elif choice == '8':
            show_project_info()
        elif choice == '9':
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1-9 중에서 선택하세요.")


def check_data_structure():
    """데이터 구조 확인"""
    print("\n🔍 데이터 구조 확인")
    print("-" * 30)
    
    try:
        from test_preprocessing import test_data_structure
        test_data_structure()
    except Exception as e:
        print(f"❌ 데이터 구조 확인 실패: {e}")


def run_preprocessing():
    """데이터 전처리 실행"""
    print("\n🔧 데이터 전처리")
    print("-" * 30)
    
    print("전처리 옵션:")
    print("1. 테스트 전처리 (소규모)")
    print("2. 전체 배치 전처리")
    
    choice = input("선택하세요 (1-2): ").strip()
    
    if choice == '1':
        try:
            from test_preprocessing import test_preprocessing
            test_preprocessing()
        except Exception as e:
            print(f"❌ 테스트 전처리 실패: {e}")
    elif choice == '2':
        try:
            os.system("python main.py --mode preprocess")
        except Exception as e:
            print(f"❌ 배치 전처리 실패: {e}")
    else:
        print("❌ 잘못된 선택입니다.")


def run_training():
    """모델 학습 실행"""
    print("\n🤖 모델 학습")
    print("-" * 30)
    
    print("학습 옵션:")
    print("1. 빠른 테스트 학습 (3 에포크)")
    print("2. 정상 학습 (기본 설정)")
    print("3. 커스텀 학습")
    
    choice = input("선택하세요 (1-3): ").strip()
    
    if choice == '1':
        try:
            from trainer import train_model
            train_model(
                model_type='simple',
                epochs=3,
                patience=2
            )
        except Exception as e:
            print(f"❌ 테스트 학습 실패: {e}")
    elif choice == '2':
        try:
            os.system("python main.py --mode train")
        except Exception as e:
            print(f"❌ 정상 학습 실패: {e}")
    elif choice == '3':
        run_custom_training()
    else:
        print("❌ 잘못된 선택입니다.")


def run_custom_training():
    """커스텀 학습 설정"""
    print("\n⚙️ 커스텀 학습 설정")
    
    try:
        # 모델 타입 선택
        print("모델 타입:")
        print("1. Simple (간단한 LSTM)")
        print("2. CNN (CNN 전용)")
        print("3. Hybrid (CNN + LSTM)")
        
        model_choice = input("모델 타입 선택 (1-3): ").strip()
        model_types = {'1': 'simple', '2': 'cnn', '3': 'hybrid'}
        model_type = model_types.get(model_choice, 'hybrid')
        
        # 에포크 수 설정
        epochs_input = input(f"에포크 수 (기본값: {Config.EPOCHS}): ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else Config.EPOCHS
        
        # 학습 실행
        from trainer import train_model
        train_model(
            model_type=model_type,
            epochs=epochs,
            patience=max(3, epochs//10)
        )
        
    except Exception as e:
        print(f"❌ 커스텀 학습 실패: {e}")


def run_evaluation():
    """모델 평가 실행"""
    print("\n📈 모델 평가")
    print("-" * 30)
    
    try:
        from evaluator import list_available_models, evaluate_saved_model
        
        available_models = list_available_models()
        
        if not available_models:
            print("❌ 평가할 모델이 없습니다. 먼저 모델을 학습하세요.")
            return
        
        print(f"📂 사용 가능한 모델 ({len(available_models)}개):")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
        
        if len(available_models) == 1:
            # 모델이 하나만 있으면 자동 선택
            selected_model = available_models[0]
            print(f"\n🎯 자동 선택: {selected_model}")
        else:
            # 사용자 선택
            choice = input(f"\n평가할 모델 번호 (1-{len(available_models)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                selected_model = available_models[int(choice) - 1]
            else:
                print("❌ 잘못된 선택입니다.")
                return
        
        # 평가 실행
        evaluate_saved_model(selected_model, detailed=True)
        
    except Exception as e:
        print(f"❌ 모델 평가 실패: {e}")


def show_config():
    """설정 확인"""
    print("\n⚙️ 현재 설정")
    print("-" * 30)
    
    Config.print_config()
    
    # 디렉토리 상태 확인
    print(f"\n📁 디렉토리 상태:")
    dirs_to_check = [
        Config.DATA_DIR,
        Config.PROCESSED_DATA_DIR,
        Config.MODEL_DIR,
        Config.LOG_DIR
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"   ✅ {dir_path}: {file_count}개 파일")
        else:
            print(f"   ❌ {dir_path}: 존재하지 않음")


def run_full_pipeline():
    """전체 파이프라인 실행"""
    print("\n🔄 전체 파이프라인")
    print("-" * 30)
    
    print("⚠️ 전체 파이프라인을 실행합니다:")
    print("1. 데이터 전처리")
    print("2. 모델 학습")
    print("3. 모델 평가")
    print("\n예상 소요 시간: 30분 ~ 2시간")
    
    confirm = input("계속하시겠습니까? (y/n): ").lower().strip()
    
    if confirm == 'y':
        try:
            os.system("python main.py --mode all")
        except Exception as e:
            print(f"❌ 파이프라인 실행 실패: {e}")
    else:
        print("🔄 파이프라인 실행을 취소했습니다.")


def run_test_mode():
    """테스트 모드"""
    print("\n🧪 테스트 모드")
    print("-" * 30)
    
    try:
        os.system("python test_preprocessing.py")
    except Exception as e:
        print(f"❌ 테스트 모드 실행 실패: {e}")


def show_project_info():
    """프로젝트 정보"""
    print("\n📁 프로젝트 정보")
    print("-" * 30)
    
    print("🎯 CSI 낙상 감지 v4")
    print("   Channel State Information을 이용한 실시간 낙상 감지 시스템")
    
    print(f"\n📂 프로젝트 구조:")
    
    files_info = [
        ("main.py", "메인 실행 스크립트"),
        ("config.py", "설정 파일"),
        ("data_preprocessing.py", "데이터 전처리 모듈"),
        ("data_generator.py", "데이터 제너레이터"),
        ("model_builder.py", "모델 아키텍처"),
        ("trainer.py", "모델 학습기"),
        ("evaluator.py", "모델 평가기"),
        ("utils.py", "유틸리티 함수들"),
        ("test_preprocessing.py", "전처리 테스트"),
        ("quick_start.py", "간편 실행 스크립트 (현재 파일)"),
        ("README.md", "프로젝트 문서")
    ]
    
    for filename, description in files_info:
        if os.path.exists(filename):
            print(f"   ✅ {filename:<20} - {description}")
        else:
            print(f"   ❌ {filename:<20} - {description} (없음)")
    
    print(f"\n🔗 주요 기능:")
    print(f"   • 이동 평균 필터 기반 노이즈 제거")
    print(f"   • Z-score 기반 이상치 제거")
    print(f"   • 다양한 정규화 옵션 (MinMax, Standard, Robust)")
    print(f"   • CNN + LSTM 하이브리드 모델")
    print(f"   • 메모리 효율적 배치 처리")
    print(f"   • 포괄적인 모델 평가 및 시각화")


if __name__ == "__main__":
    quick_start()
