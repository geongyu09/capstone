# main.py
"""
CSI 기반 낙상 감지 시스템 통합 실행 스크립트
모든 모듈을 통합하여 사용하기 쉬운 인터페이스 제공
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 로컬 모듈 import
from config import CSIConfig
from trainer import CSITrainer
from analyzer import FallTimelineAnalyzer
from model_builder import CSIModelBuilder, recommend_model

def setup_main_logger():
    """메인 로거 설정"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(CSIConfig.LOG_DIR, f'main_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def train_command(args):
    """훈련 명령 실행"""
    logger = setup_main_logger()
    logger.info("🚀 CSI 낙상 감지 모델 훈련 시작")
    
    try:
        # 트레이너 생성
        trainer = CSITrainer(
            data_directory=args.data_dir,
            model_type=args.model_type
        )
        
        # 전체 학습 또는 빠른 학습
        if args.quick and args.csv_file:
            logger.info(f"🧪 빠른 학습 모드: {args.csv_file}")
            history, metrics = trainer.quick_train(args.csv_file, epochs=args.epochs)
        else:
            logger.info(f"🎓 전체 학습 모드")
            history = trainer.train_model(epochs=args.epochs)
        
        # 훈련 요약 출력
        trainer.print_training_summary()
        
        logger.info("✅ 훈련 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 훈련 실패: {e}")
        return False

def analyze_command(args):
    """분석 명령 실행"""
    logger = setup_main_logger()
    logger.info("🔍 CSI 데이터 분석 시작")
    
    try:
        # 분석기 생성
        analyzer = FallTimelineAnalyzer(
            model_path=args.model_path,
            confidence_threshold=args.confidence
        )
        
        # 분석 실행
        if args.visualize:
            fall_events = analyzer.analyze_and_visualize(
                args.csv_file, 
                save_results=args.save_results
            )
        else:
            fall_events = analyzer.analyze_csv_timeline(args.csv_file)
            
            if args.save_results and fall_events:
                analyzer.export_fall_events()
        
        logger.info(f"✅ 분석 완료! 감지된 이벤트: {len(fall_events)}개")
        return True
        
    except Exception as e:
        logger.error(f"❌ 분석 실패: {e}")
        return False

def config_command(args):
    """설정 정보 출력"""
    print("⚙️ CSI 낙상 감지 시스템 설정")
    print("=" * 60)
    
    CSIConfig.print_config()
    
    # 추가 정보
    print(f"\n📁 경로 상태:")
    paths_to_check = [
        ('데이터 디렉토리', CSIConfig.DEFAULT_DATA_DIR),
        ('모델 저장소', CSIConfig.MODEL_SAVE_DIR),
        ('로그 디렉토리', CSIConfig.LOG_DIR),
        ('결과 디렉토리', CSIConfig.RESULTS_DIR)
    ]
    
    for name, path in paths_to_check:
        status = "✅ 존재" if os.path.exists(path) else "❌ 없음"
        print(f"   {name}: {path} ({status})")
    
    # 사용 가능한 모델 타입
    print(f"\n🏗️ 사용 가능한 모델 타입:")
    model_types = ['basic_lstm', 'cnn_lstm_hybrid', 'attention', 'multi_scale', 'lightweight']
    for model_type in model_types:
        recommended = " (권장)" if model_type == recommend_model() else ""
        print(f"   • {model_type}{recommended}")
    
    return True

def test_command(args):
    """시스템 테스트 실행"""
    logger = setup_main_logger()
    logger.info("🧪 시스템 테스트 시작")
    
    try:
        # 1. 설정 테스트
        print("1️⃣ 설정 테스트...")
        CSIConfig.create_directories()
        print("   ✅ 디렉토리 생성 완료")
        
        # 2. 모델 빌더 테스트
        print("\n2️⃣ 모델 빌더 테스트...")
        builder = CSIModelBuilder()
        builder.print_model_comparison()
        print("   ✅ 모델 빌더 테스트 완료")
        
        # 3. 데이터 파일 확인
        print("\n3️⃣ 데이터 파일 확인...")
        test_files = []
        
        # 기본 테스트 파일들
        default_test_files = ['35.csv', 'case32.csv', 'test.csv']
        for test_file in default_test_files:
            if os.path.exists(test_file):
                test_files.append(test_file)
                print(f"   ✅ 발견: {test_file}")
        
        # 데이터 디렉토리 확인
        if os.path.exists(CSIConfig.DEFAULT_DATA_DIR):
            import glob
            data_files = glob.glob(os.path.join(CSIConfig.DEFAULT_DATA_DIR, "*.csv"))
            test_files.extend(data_files[:3])  # 최대 3개
            print(f"   📁 데이터 디렉토리: {len(data_files)}개 CSV 파일 발견")
        
        if not test_files:
            print("   ⚠️ 테스트용 CSV 파일이 없습니다")
            return False
        
        # 4. 빠른 훈련 테스트 (선택적)
        if args.include_training:
            print(f"\n4️⃣ 빠른 훈련 테스트...")
            test_file = test_files[0]
            print(f"   📄 테스트 파일: {os.path.basename(test_file)}")
            
            trainer = CSITrainer(model_type='lightweight')  # 빠른 테스트용 경량 모델
            history, metrics = trainer.quick_train(test_file, epochs=3)
            
            print(f"   ✅ 빠른 훈련 완료: 정확도 {metrics.get('accuracy', 0):.1%}")
            
            # 5. 분석 테스트
            print(f"\n5️⃣ 분석 테스트...")
            analyzer = FallTimelineAnalyzer(confidence_threshold=0.3)
            fall_events = analyzer.analyze_csv_timeline(test_file)
            
            print(f"   ✅ 분석 완료: {len(fall_events)}개 이벤트 감지")
        
        else:
            print(f"\n   ℹ️ 훈련 테스트 스킵 (--include-training 옵션으로 활성화)")
        
        logger.info("✅ 시스템 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 테스트 실패: {e}")
        return False

def info_command(args):
    """시스템 정보 출력"""
    print("ℹ️ CSI 낙상 감지 시스템 정보")
    print("=" * 60)
    
    # 버전 정보
    print("📋 시스템 정보:")
    print("   버전: 2.0")
    print("   제작: CSI Fall Detection System")
    print("   최적화: 288Hz 고주파 CSI 데이터")
    
    # 설치된 모델 확인
    print(f"\n🤖 설치된 모델:")
    import glob
    model_files = glob.glob(os.path.join(CSIConfig.MODEL_SAVE_DIR, "*.keras"))
    
    if model_files:
        for model_file in model_files[-5:]:  # 최근 5개만
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
            print(f"   • {os.path.basename(model_file)} ({file_size:.1f}MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
        
        if len(model_files) > 5:
            print(f"   ... 외 {len(model_files)-5}개 모델")
    else:
        print("   ❌ 설치된 모델이 없습니다")
    
    # 데이터 디렉토리 상태
    print(f"\n📊 데이터 상태:")
    if os.path.exists(CSIConfig.DEFAULT_DATA_DIR):
        import glob
        csv_files = glob.glob(os.path.join(CSIConfig.DEFAULT_DATA_DIR, "**", "*.csv"), recursive=True)
        total_size = sum(os.path.getsize(f) for f in csv_files if os.path.exists(f)) / (1024 * 1024)  # MB
        
        print(f"   📁 CSV 파일: {len(csv_files)}개")
        print(f"   💾 총 크기: {total_size:.1f}MB")
    else:
        print(f"   ❌ 데이터 디렉토리 없음: {CSIConfig.DEFAULT_DATA_DIR}")
    
    # 성능 벤치마크 (간단한)
    print(f"\n⚡ 성능 정보:")
    try:
        import tensorflow as tf
        print(f"   TensorFlow: {tf.__version__}")
        
        # GPU 확인
        if tf.config.list_physical_devices('GPU'):
            print("   🎮 GPU: 사용 가능")
        else:
            print("   💻 GPU: CPU 모드")
            
        # 메모리 정보
        import psutil
        memory = psutil.virtual_memory()
        print(f"   🧠 메모리: {memory.available / (1024**3):.1f}GB 사용 가능")
        
    except ImportError:
        print("   ⚠️ 성능 정보 수집 불가")
    
    return True

def create_parser():
    """명령행 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description='CSI 기반 낙상 감지 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 모델 훈련
  python main.py train --data-dir ./csi_data --epochs 50
  
  # 빠른 테스트 훈련
  python main.py train --quick --csv-file 35.csv --epochs 10
  
  # 데이터 분석
  python main.py analyze 35.csv --confidence 0.5 --visualize
  
  # 시스템 테스트
  python main.py test --include-training
  
  # 설정 정보
  python main.py config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령')
    
    # Train 명령
    train_parser = subparsers.add_parser('train', help='모델 훈련')
    train_parser.add_argument('--data-dir', default=CSIConfig.DEFAULT_DATA_DIR,
                             help='데이터 디렉토리 경로')
    train_parser.add_argument('--model-type', default='cnn_lstm_hybrid',
                             choices=['basic_lstm', 'cnn_lstm_hybrid', 'attention', 'multi_scale', 'lightweight'],
                             help='사용할 모델 타입')
    train_parser.add_argument('--epochs', type=int, default=CSIConfig.EPOCHS,
                             help='훈련 에포크 수')
    train_parser.add_argument('--quick', action='store_true',
                             help='빠른 훈련 모드 (단일 파일)')
    train_parser.add_argument('--csv-file',
                             help='빠른 훈련용 CSV 파일 (--quick과 함께 사용)')
    
    # Analyze 명령
    analyze_parser = subparsers.add_parser('analyze', help='데이터 분석')
    analyze_parser.add_argument('csv_file', help='분석할 CSV 파일')
    analyze_parser.add_argument('--model-path',
                               help='사용할 모델 파일 경로 (없으면 자동 탐지)')
    analyze_parser.add_argument('--confidence', type=float, default=CSIConfig.CONFIDENCE_THRESHOLD,
                               help='낙상 감지 신뢰도 임계값')
    analyze_parser.add_argument('--visualize', action='store_true',
                               help='결과 시각화')
    analyze_parser.add_argument('--save-results', action='store_true', default=True,
                               help='결과 저장')
    
    # Test 명령
    test_parser = subparsers.add_parser('test', help='시스템 테스트')
    test_parser.add_argument('--include-training', action='store_true',
                            help='훈련 테스트 포함 (시간 오래 걸림)')
    
    # Config 명령
    config_parser = subparsers.add_parser('config', help='설정 정보 출력')
    
    # Info 명령
    info_parser = subparsers.add_parser('info', help='시스템 정보 출력')
    
    return parser

def main():
    """메인 실행 함수"""
    # 파서 생성
    parser = create_parser()
    args = parser.parse_args()
    
    # 명령이 없으면 도움말 출력
    if not args.command:
        parser.print_help()
        return
    
    # 시작 메시지
    print("🏠 CSI 기반 낙상 감지 시스템 v2.0")
    print("=" * 60)
    
    # 명령 실행
    success = False
    
    if args.command == 'train':
        success = train_command(args)
    elif args.command == 'analyze':
        success = analyze_command(args)
    elif args.command == 'test':
        success = test_command(args)
    elif args.command == 'config':
        success = config_command(args)
    elif args.command == 'info':
        success = info_command(args)
    else:
        print(f"❌ 알 수 없는 명령: {args.command}")
        parser.print_help()
    
    # 결과 출력
    if success:
        print(f"\n🎉 {args.command} 명령이 성공적으로 완료되었습니다!")
    else:
        print(f"\n❌ {args.command} 명령이 실패했습니다.")
        sys.exit(1)

def quick_demo():
    """빠른 데모 실행"""
    print("🎪 CSI 낙상 감지 시스템 빠른 데모")
    print("=" * 50)
    
    # 데모용 파일 찾기
    demo_files = ['35.csv', 'case32.csv', 'test.csv', 'data.csv']
    demo_file = None
    
    for file in demo_files:
        if os.path.exists(file):
            demo_file = file
            break
    
    if not demo_file:
        print("❌ 데모용 CSV 파일이 없습니다!")
        print("💡 다음 파일 중 하나를 준비해주세요:")
        for file in demo_files:
            print(f"   • {file}")
        return False
    
    print(f"📄 데모 파일: {demo_file}")
    
    try:
        # 1. 빠른 훈련
        print("\n1️⃣ 빠른 모델 훈련...")
        trainer = CSITrainer(model_type='lightweight')
        history, metrics = trainer.quick_train(demo_file, epochs=3)
        
        print(f"   ✅ 훈련 완료: 정확도 {metrics.get('accuracy', 0):.1%}")
        
        # 2. 분석 및 시각화
        print("\n2️⃣ 데이터 분석 및 시각화...")
        analyzer = FallTimelineAnalyzer(confidence_threshold=0.3)
        fall_events = analyzer.analyze_and_visualize(demo_file)
        
        print(f"   ✅ 분석 완료: {len(fall_events)}개 낙상 이벤트 감지")
        
        # 3. 결과 요약
        print("\n3️⃣ 데모 결과 요약:")
        print(f"   📊 훈련 데이터: {demo_file}")
        print(f"   🎯 모델 정확도: {metrics.get('accuracy', 0):.1%}")
        print(f"   🚨 감지된 낙상: {len(fall_events)}개")
        
        if fall_events:
            high_conf = sum(1 for e in fall_events if e['confidence_level'] == 'high')
            print(f"   🔴 고신뢰도 이벤트: {high_conf}개")
        
        print("\n🎉 데모 완료! 그래프 창에서 결과를 확인하세요.")
        return True
        
    except Exception as e:
        print(f"❌ 데모 실패: {e}")
        return False

if __name__ == "__main__":
    # 명령행 인자가 없으면 빠른 데모 실행
    if len(sys.argv) == 1:
        print("💡 명령행 인자가 없습니다. 빠른 데모를 실행합니다...")
        print("   전체 기능은 'python main.py --help'를 참조하세요.\n")
        
        quick_demo()
    else:
        # 일반 명령 실행
        main()