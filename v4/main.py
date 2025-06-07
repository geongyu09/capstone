"""
CSI 낙상 감지 v4 메인 실행 스크립트
"""

import os
import sys
import argparse
from typing import List

# 현재 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, ModelConfig
from utils import (
    setup_logging, collect_csv_files, analyze_data_distribution, 
    print_data_analysis, create_timestamp
)
from data_preprocessing import CSIPreprocessor


def setup_environment():
    """환경 설정"""
    # 필요한 디렉토리 생성
    Config.ensure_directories()
    
    # 로깅 설정
    logger = setup_logging(Config.LOG_DIR, Config.LOG_LEVEL)
    
    return logger


def preprocess_data(logger):
    """데이터 전처리 실행"""
    logger.info("🔧 데이터 전처리 시작")
    
    # CSV 파일 수집
    data_paths = Config.get_data_paths()
    csv_files = collect_csv_files(data_paths)
    
    logger.info(f"수집된 CSV 파일: {len(csv_files)}개")
    
    if not csv_files:
        logger.error("처리할 CSV 파일이 없습니다.")
        return False
    
    # 데이터 분석
    logger.info("📊 데이터 분포 분석 중...")
    stats = analyze_data_distribution(csv_files)
    print_data_analysis(stats)
    
    # 전처리기 초기화
    preprocessor = CSIPreprocessor(
        amplitude_start_col=Config.AMPLITUDE_START_COL,
        amplitude_end_col=Config.AMPLITUDE_END_COL,
        scaler_type=Config.SCALER_TYPE,
        logger=logger
    )
    
    # 배치 전처리 실행
    logger.info("⚡ 배치 전처리 시작...")
    
    results = preprocessor.process_multiple_files(
        file_paths=csv_files,
        output_dir=Config.PROCESSED_DATA_DIR,
        moving_avg_window=Config.MOVING_AVERAGE_WINDOW,
        outlier_threshold=Config.OUTLIER_THRESHOLD,
        fit_scaler_on_first=True
    )
    
    # 결과 출력
    logger.info(f"✅ 전처리 완료: {len(results['processed_files'])}개 성공, {len(results['failed_files'])}개 실패")
    
    if results['processing_stats']:
        report = preprocessor.generate_processing_report(results['processing_stats'])
        print(report)
    
    # 실패한 파일들 로깅
    if results['failed_files']:
        logger.warning("실패한 파일들:")
        for failed in results['failed_files']:
            logger.warning(f"  {failed['file']}: {failed['error']}")
    
    return len(results['processed_files']) > 0


def train_model(logger):
    """모델 학습 실행"""
    logger.info("🤖 모델 학습 시작")
    
    try:
        from trainer import train_model as run_training
        
        # 학습 실행
        results = run_training(
            model_type='hybrid',  # 기본값으로 hybrid 모델 사용
            epochs=Config.EPOCHS,
            patience=10
        )
        
        if results:
            logger.info(f"✅ 학습 완료: {results.get('experiment_name')}")
            logger.info(f"최고 검증 정확도: {results.get('best_val_accuracy', 0):.4f}")
            
            if 'test_results' in results:
                logger.info(f"테스트 정확도: {results['test_results'].get('accuracy', 0):.4f}")
            
            return True
        else:
            logger.error("학습이 완료되지 않았습니다.")
            return False
            
    except Exception as e:
        logger.error(f"모델 학습 실패: {e}")
        return False


def evaluate_model(logger):
    """모델 평가 실행"""
    logger.info("📊 모델 평가 시작")
    
    try:
        from evaluator import list_available_models, evaluate_saved_model
        
        # 사용 가능한 모델 확인
        available_models = list_available_models()
        
        if not available_models:
            logger.error("평가할 모델이 없습니다. 먼저 모델을 학습하세요.")
            return False
        
        # 가장 최근 모델 선택 (이름 기준)
        latest_model = sorted(available_models)[-1]
        logger.info(f"평가 대상 모델: {latest_model}")
        
        # 평가 실행
        results = evaluate_saved_model(latest_model, detailed=True)
        
        if results:
            logger.info(f"✅ 평가 완료: {results['model_info']['experiment_name']}")
            logger.info(f"정확도: {results['basic_metrics'].get('accuracy', 0):.4f}")
            
            if 'best_f1_score' in results:
                logger.info(f"최고 F1 점수: {results['best_f1_score']:.4f}")
            
            return True
        else:
            logger.error("평가가 완료되지 않았습니다.")
            return False
            
    except Exception as e:
        logger.error(f"모델 평가 실패: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="CSI 낙상 감지 v4")
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate', 'all'], 
                       default='preprocess', help='실행 모드')
    parser.add_argument('--config', action='store_true', 
                       help='설정 정보만 출력하고 종료')
    
    args = parser.parse_args()
    
    # 설정 정보 출력
    if args.config:
        Config.print_config()
        print()
        ModelConfig.print_model_config()
        return
    
    # 환경 설정
    logger = setup_environment()
    
    # 시작 메시지
    timestamp = create_timestamp()
    logger.info("🚀 CSI 낙상 감지 v4 시작")
    logger.info(f"실행 시간: {timestamp}")
    logger.info(f"실행 모드: {args.mode}")
    
    try:
        success = True
        
        if args.mode in ['preprocess', 'all']:
            success &= preprocess_data(logger)
        
        if args.mode in ['train', 'all'] and success:
            success &= train_model(logger)
        
        if args.mode in ['evaluate', 'all'] and success:
            success &= evaluate_model(logger)
        
        if success:
            logger.info("✅ 모든 작업이 성공적으로 완료되었습니다.")
        else:
            logger.error("❌ 일부 작업이 실패했습니다.")
            
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        end_timestamp = create_timestamp()
        logger.info(f"🏁 프로그램 종료: {end_timestamp}")


if __name__ == "__main__":
    main()
