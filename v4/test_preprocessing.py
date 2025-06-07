"""
CSI 데이터 전처리 테스트 스크립트
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
from data_preprocessing import CSIPreprocessor

def test_data_structure():
    """데이터 구조 확인"""
    print("🔍 데이터 구조 확인")
    print("=" * 50)
    
    # 테스트 파일 찾기
    data_paths = [
        "../csi_data/case1",
        "../csi_data/case2", 
        "../csi_data/case3",
        "../labeled"
    ]
    
    test_file = None
    for path in data_paths:
        if os.path.exists(path):
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            if csv_files:
                test_file = csv_files[0]
                break
    
    if not test_file:
        print("❌ 테스트할 CSV 파일을 찾을 수 없습니다.")
        return None
    
    print(f"📁 테스트 파일: {test_file}")
    
    try:
        # 파일 크기 확인
        file_size = os.path.getsize(test_file)
        print(f"📏 파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        # 첫 몇 줄만 읽어서 구조 확인
        df_sample = pd.read_csv(test_file, nrows=10)
        print(f"📊 데이터 형태 (샘플): {df_sample.shape}")
        print(f"📋 컬럼 정보:")
        print(f"   - 총 컬럼 수: {len(df_sample.columns)}")
        print(f"   - 처음 10개 컬럼: {list(df_sample.columns[:10])}")
        
        # 8번부터 253번 컬럼 확인
        if len(df_sample.columns) > 253:
            amplitude_cols = df_sample.columns[8:253]
            print(f"   - Amplitude 컬럼 범위 (8:253): {len(amplitude_cols)}개")
            print(f"   - Amplitude 첫 5개: {list(amplitude_cols[:5])}")
            print(f"   - Amplitude 마지막 5개: {list(amplitude_cols[-5:])}")
        else:
            print(f"   ⚠️ 컬럼 수가 253개보다 적습니다: {len(df_sample.columns)}개")
        
        # 라벨 컬럼 확인
        if 'label' in df_sample.columns:
            print(f"   ✅ 라벨 컬럼 발견")
        else:
            print(f"   ⚠️ 라벨 컬럼이 없습니다. 사용 가능한 컬럼: {list(df_sample.columns)}")
        
        # 샘플 데이터 값 확인
        print(f"\n📈 샘플 데이터 값:")
        if len(df_sample.columns) > 15:
            sample_data = df_sample.iloc[0, 8:15]  # 첫 번째 행의 8-14번 컬럼
            print(f"   첫 번째 행 (8-14번 컬럼): {sample_data.values}")
            print(f"   데이터 타입: {sample_data.dtype}")
            print(f"   값 범위: {sample_data.min():.3f} ~ {sample_data.max():.3f}")
        
        return test_file
        
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None

def test_preprocessing():
    """전처리 기능 테스트"""
    print("\n🧪 전처리 기능 테스트")
    print("=" * 50)
    
    # 테스트 파일 확인
    test_file = test_data_structure()
    if not test_file:
        return
    
    try:
        # 전처리기 초기화
        preprocessor = CSIPreprocessor(
            amplitude_start_col=8,
            amplitude_end_col=253,
            scaler_type='minmax'
        )
        
        print(f"\n⚙️ 전처리 시작...")
        
        # 작은 샘플로 테스트 (메모리 절약)
        df_full = pd.read_csv(test_file)
        df_sample = df_full.head(100)  # 처음 100행만 사용
        
        # 임시 파일로 저장
        temp_file = "temp_sample.csv"
        df_sample.to_csv(temp_file, index=False)
        
        # 전처리 실행
        processed_df, stats = preprocessor.process_single_file(
            file_path=temp_file,
            moving_avg_window=5,
            outlier_threshold=3.0,
            fit_scaler=True,
            save_processed=False
        )
        
        print(f"✅ 전처리 완료!")
        print(f"📊 처리 결과:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        # 전처리 전후 비교
        print(f"\n📈 전처리 전후 비교:")
        
        # 원본 amplitude 데이터
        original_amplitude = df_sample.iloc[:, 8:253]
        processed_amplitude = processed_df.iloc[:, 8:253]
        
        print(f"   원본 데이터:")
        print(f"     - 형태: {original_amplitude.shape}")
        print(f"     - 값 범위: {original_amplitude.values.min():.3f} ~ {original_amplitude.values.max():.3f}")
        print(f"     - 평균: {original_amplitude.values.mean():.3f}")
        print(f"     - 표준편차: {original_amplitude.values.std():.3f}")
        
        print(f"   전처리된 데이터:")
        print(f"     - 형태: {processed_amplitude.shape}")
        print(f"     - 값 범위: {processed_amplitude.values.min():.3f} ~ {processed_amplitude.values.max():.3f}")
        print(f"     - 평균: {processed_amplitude.values.mean():.3f}")
        print(f"     - 표준편차: {processed_amplitude.values.std():.3f}")
        
        # 임시 파일 삭제
        os.remove(temp_file)
        
        print(f"\n✅ 전처리 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 전처리 테스트 실패: {e}")
        if os.path.exists("temp_sample.csv"):
            os.remove("temp_sample.csv")

def collect_csi_files():
    """모든 CSI 파일 수집"""
    print("\n📂 CSI 파일 수집")
    print("=" * 50)
    
    csv_files = []
    data_paths = [
        "../csi_data/case1",
        "../csi_data/case2", 
        "../csi_data/case3",
        "../labeled"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.csv"))
            csv_files.extend(files)
            print(f"📁 {path}: {len(files)}개 파일")
        else:
            print(f"❌ {path}: 경로를 찾을 수 없습니다")
    
    print(f"\n📊 총 수집된 파일: {len(csv_files)}개")
    
    if csv_files:
        print(f"📋 파일 예시:")
        for i, file_path in enumerate(csv_files[:5]):
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            print(f"   {i+1}. {os.path.basename(file_path)} ({file_size:.1f} MB)")
        
        if len(csv_files) > 5:
            print(f"   ... 외 {len(csv_files)-5}개 파일")
    
    return csv_files

def run_batch_preprocessing():
    """배치 전처리 실행"""
    print("\n🚀 배치 전처리 실행")
    print("=" * 50)
    
    # 파일 수집
    csv_files = collect_csi_files()
    
    if not csv_files:
        print("❌ 처리할 CSV 파일이 없습니다.")
        return
    
    # 사용자 확인
    print(f"\n⚠️ {len(csv_files)}개 파일을 전처리합니다.")
    print(f"   예상 소요 시간: 약 {len(csv_files)*2}분")
    
    choice = input("계속하시겠습니까? (y/n): ").lower().strip()
    
    if choice != 'y':
        print("🔄 배치 처리를 취소했습니다.")
        return
    
    try:
        # 전처리기 초기화
        preprocessor = CSIPreprocessor(
            amplitude_start_col=8,
            amplitude_end_col=253,
            scaler_type='minmax'
        )
        
        # 출력 디렉토리 설정
        output_dir = "./processed_data"
        
        print(f"\n⚡ 배치 전처리 시작...")
        print(f"📁 출력 디렉토리: {output_dir}")
        
        # 배치 처리 실행
        results = preprocessor.process_multiple_files(
            file_paths=csv_files,
            output_dir=output_dir,
            moving_avg_window=5,
            outlier_threshold=3.0,
            fit_scaler_on_first=True
        )
        
        # 결과 출력
        print(f"\n📊 배치 처리 결과:")
        print(f"   ✅ 성공: {len(results['processed_files'])}개")
        print(f"   ❌ 실패: {len(results['failed_files'])}개")
        print(f"   📁 출력 위치: {output_dir}")
        
        # 처리 보고서 생성
        if results['processing_stats']:
            report = preprocessor.generate_processing_report(results['processing_stats'])
            print(report)
        
        # 실패한 파일 목록
        if results['failed_files']:
            print(f"\n❌ 실패한 파일들:")
            for failed in results['failed_files']:
                print(f"   - {os.path.basename(failed['file'])}: {failed['error']}")
        
        print(f"\n✅ 배치 전처리 완료!")
        
    except Exception as e:
        print(f"❌ 배치 처리 실패: {e}")

def main():
    """메인 함수"""
    print("🔧 CSI 데이터 전처리 테스트 v4")
    print("=" * 60)
    
    while True:
        print(f"\n📋 메뉴:")
        print(f"1. 데이터 구조 확인")
        print(f"2. 전처리 기능 테스트")
        print(f"3. 파일 수집 확인")
        print(f"4. 배치 전처리 실행")
        print(f"5. 종료")
        
        choice = input("\n선택하세요 (1-5): ").strip()
        
        if choice == '1':
            test_data_structure()
        elif choice == '2':
            test_preprocessing()
        elif choice == '3':
            collect_csi_files()
        elif choice == '4':
            run_batch_preprocessing()
        elif choice == '5':
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1-5 중에서 선택하세요.")

if __name__ == "__main__":
    main()
