"""
CSI 데이터 전처리 모듈 v4
낙상 감지를 위한 CSI 데이터 전처리 파이프라인

개선사항:
- 모듈화된 전처리 함수들
- 배치 처리 지원
- 메모리 효율적 처리
- 다양한 정규화 옵션
- 데이터 검증 기능
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
import logging
from typing import Tuple, Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class CSIPreprocessor:
    """CSI 데이터 전처리를 위한 클래스"""
    
    def __init__(self, 
                 amplitude_start_col: int = 8,
                 amplitude_end_col: int = 253,
                 scaler_type: str = 'minmax',
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            amplitude_start_col: amplitude 데이터 시작 컬럼 인덱스
            amplitude_end_col: amplitude 데이터 종료 컬럼 인덱스 (exclusive)
            scaler_type: 정규화 방법 ('minmax', 'standard', 'robust')
            logger: 로거 객체
        """
        self.amplitude_start_col = amplitude_start_col
        self.amplitude_end_col = amplitude_end_col
        self.scaler_type = scaler_type
        self.logger = logger or self._setup_logger()
        
        # 스케일러 초기화
        self.scaler = self._init_scaler()
        self.fitted = False
        
        self.logger.info(f"CSI Preprocessor 초기화 완료")
        self.logger.info(f"  Amplitude 컬럼 범위: {amplitude_start_col}:{amplitude_end_col}")
        self.logger.info(f"  스케일러 타입: {scaler_type}")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger('CSIPreprocessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_scaler(self):
        """스케일러 초기화"""
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"지원하지 않는 스케일러 타입: {self.scaler_type}")
    
    def apply_moving_average_2d(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        2D 데이터에 대해 시간 축(행 방향)으로 이동 평균 필터를 적용
        
        Args:
            data: (패킷 수, 피처 수) 형태의 2D 데이터
            window_size: 이동 평균 창 크기
            
        Returns:
            이동 평균 처리된 결과
        """
        self.logger.debug(f"이동 평균 필터 적용 (window_size={window_size})")
        
        df = pd.DataFrame(data)
        filtered = df.rolling(window=window_size, min_periods=1).mean()
        return filtered.values
    
    def remove_outliers_zscore(self, data: np.ndarray, threshold: float = 3.0) -> pd.DataFrame:
        """
        z-score 기반 이상치 제거
        이상치인 값은 NaN으로 바꾸고 선형 보간으로 채움
        
        Args:
            data: 입력 데이터
            threshold: z-score 임계값
            
        Returns:
            이상치가 제거된 DataFrame
        """
        self.logger.debug(f"Z-score 기반 이상치 제거 (threshold={threshold})")
        
        df = pd.DataFrame(data)
        z_scores = np.abs(zscore(df, nan_policy='omit'))
        
        # 이상치를 NaN으로 설정
        df[z_scores > threshold] = np.nan
        
        # 선형 보간으로 결측치 채우기
        df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')
        
        # 여전히 NaN이 있는 경우 forward fill / backward fill
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        데이터 정규화
        
        Args:
            df: 입력 DataFrame
            fit: True이면 스케일러를 학습, False이면 기존 스케일러 사용
            
        Returns:
            정규화된 데이터 배열
        """
        self.logger.debug(f"데이터 정규화 ({self.scaler_type}, fit={fit})")
        
        if fit:
            scaled = self.scaler.fit_transform(df)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("스케일러가 학습되지 않았습니다. fit=True로 먼저 호출하세요.")
            scaled = self.scaler.transform(df)
        
        return scaled
    
    def process_single_file(self, 
                          file_path: str,
                          moving_avg_window: int = 5,
                          outlier_threshold: float = 3.0,
                          fit_scaler: bool = True,
                          save_processed: bool = False,
                          output_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        단일 파일 전처리
        
        Args:
            file_path: CSV 파일 경로
            moving_avg_window: 이동 평균 창 크기
            outlier_threshold: 이상치 임계값
            fit_scaler: 스케일러 학습 여부
            save_processed: 처리된 데이터 저장 여부
            output_path: 출력 파일 경로
            
        Returns:
            전처리된 DataFrame과 처리 통계
        """
        self.logger.info(f"파일 전처리 시작: {file_path}")
        
        try:
            # 데이터 로드
            df = pd.read_csv(file_path)
            original_shape = df.shape
            self.logger.debug(f"원본 데이터 형태: {original_shape}")
            
            # amplitude 데이터 추출
            amplitude_cols = df.columns[self.amplitude_start_col:self.amplitude_end_col]
            amplitude_data = df.iloc[:, self.amplitude_start_col:self.amplitude_end_col]
            
            self.logger.debug(f"Amplitude 데이터 형태: {amplitude_data.shape}")
            
            # 1. 이동 평균 필터 적용
            amplitude_filtered = self.apply_moving_average_2d(
                amplitude_data.values, 
                window_size=moving_avg_window
            )
            
            # 2. 이상치 제거
            amplitude_no_outliers = self.remove_outliers_zscore(
                amplitude_filtered, 
                threshold=outlier_threshold
            )
            
            # 3. 정규화
            amplitude_normalized = self.normalize_data(
                amplitude_no_outliers, 
                fit=fit_scaler
            )
            
            # 전처리 결과를 DataFrame으로 변환
            processed_df = pd.DataFrame(
                amplitude_normalized, 
                columns=amplitude_cols,
                index=amplitude_data.index
            )
            
            # 원본 데이터의 다른 컬럼들과 결합
            result_df = df.copy()
            result_df.loc[:, amplitude_cols] = processed_df
            
            # 처리 통계
            stats = {
                'original_shape': original_shape,
                'processed_shape': result_df.shape,
                'amplitude_features': len(amplitude_cols),
                'moving_avg_window': moving_avg_window,
                'outlier_threshold': outlier_threshold,
                'scaler_type': self.scaler_type,
                'scaler_fitted': self.fitted
            }
            
            # 파일 저장
            if save_processed:
                if output_path is None:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_path = f"{base_name}_processed.csv"
                
                result_df.to_csv(output_path, index=False)
                self.logger.info(f"전처리된 데이터 저장: {output_path}")
                stats['output_path'] = output_path
            
            self.logger.info(f"파일 전처리 완료: {file_path}")
            return result_df, stats
            
        except Exception as e:
            self.logger.error(f"파일 전처리 실패: {file_path} - {e}")
            raise
    
    def process_multiple_files(self, 
                             file_paths: List[str],
                             output_dir: str,
                             moving_avg_window: int = 5,
                             outlier_threshold: float = 3.0,
                             fit_scaler_on_first: bool = True) -> Dict[str, Any]:
        """
        여러 파일 배치 전처리
        
        Args:
            file_paths: 처리할 파일 경로 리스트
            output_dir: 출력 디렉토리
            moving_avg_window: 이동 평균 창 크기
            outlier_threshold: 이상치 임계값
            fit_scaler_on_first: 첫 번째 파일로 스케일러 학습 여부
            
        Returns:
            처리 결과 통계
        """
        self.logger.info(f"배치 전처리 시작: {len(file_paths)}개 파일")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'processed_files': [],
            'failed_files': [],
            'total_files': len(file_paths),
            'processing_stats': []
        }
        
        for i, file_path in enumerate(file_paths):
            try:
                # 첫 번째 파일에서만 스케일러 학습
                fit_scaler = fit_scaler_on_first and (i == 0)
                
                # 출력 파일 경로 생성
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_processed.csv")
                
                # 파일 처리
                processed_df, stats = self.process_single_file(
                    file_path=file_path,
                    moving_avg_window=moving_avg_window,
                    outlier_threshold=outlier_threshold,
                    fit_scaler=fit_scaler,
                    save_processed=True,
                    output_path=output_path
                )
                
                results['processed_files'].append(file_path)
                results['processing_stats'].append(stats)
                
                self.logger.info(f"진행률: {i+1}/{len(file_paths)} ({(i+1)/len(file_paths)*100:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"파일 처리 실패: {file_path} - {e}")
                results['failed_files'].append({'file': file_path, 'error': str(e)})
        
        self.logger.info(f"배치 전처리 완료: {len(results['processed_files'])}개 성공, {len(results['failed_files'])}개 실패")
        return results
    
    def visualize_preprocessing_effects(self, 
                                      original_data: np.ndarray,
                                      processed_data: np.ndarray,
                                      sample_features: int = 5,
                                      sample_length: int = 100) -> None:
        """
        전처리 효과 시각화
        
        Args:
            original_data: 원본 데이터
            processed_data: 전처리된 데이터
            sample_features: 시각화할 특성 수
            sample_length: 시각화할 샘플 길이
        """
        self.logger.info("전처리 효과 시각화")
        
        # 샘플 데이터 선택
        sample_data_orig = original_data[:sample_length, :sample_features]
        sample_data_proc = processed_data[:sample_length, :sample_features]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 원본 데이터 시계열
        axes[0, 0].plot(sample_data_orig)
        axes[0, 0].set_title('원본 데이터 (시계열)')
        axes[0, 0].set_xlabel('시간')
        axes[0, 0].set_ylabel('값')
        axes[0, 0].legend([f'Feature {i}' for i in range(sample_features)])
        
        # 전처리된 데이터 시계열
        axes[0, 1].plot(sample_data_proc)
        axes[0, 1].set_title('전처리된 데이터 (시계열)')
        axes[0, 1].set_xlabel('시간')
        axes[0, 1].set_ylabel('정규화된 값')
        axes[0, 1].legend([f'Feature {i}' for i in range(sample_features)])
        
        # 원본 데이터 분포
        axes[1, 0].hist(original_data.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title('원본 데이터 분포')
        axes[1, 0].set_xlabel('값')
        axes[1, 0].set_ylabel('밀도')
        
        # 전처리된 데이터 분포
        axes[1, 1].hist(processed_data.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 1].set_title('전처리된 데이터 분포')
        axes[1, 1].set_xlabel('정규화된 값')
        axes[1, 1].set_ylabel('밀도')
        
        plt.tight_layout()
        plt.show()
    
    def generate_processing_report(self, stats_list: List[Dict[str, Any]]) -> str:
        """
        전처리 결과 보고서 생성
        
        Args:
            stats_list: 처리 통계 리스트
            
        Returns:
            보고서 문자열
        """
        if not stats_list:
            return "처리된 파일이 없습니다."
        
        # 통계 계산
        total_files = len(stats_list)
        total_original_samples = sum(stat['original_shape'][0] for stat in stats_list)
        total_features = stats_list[0]['amplitude_features']
        
        report = f"""
        📊 CSI 데이터 전처리 보고서
        ===============================================
        
        📁 처리 결과:
        - 총 파일 수: {total_files:,}개
        - 총 샘플 수: {total_original_samples:,}개
        - Amplitude 특성 수: {total_features}개
        
        ⚙️ 전처리 설정:
        - 이동 평균 창 크기: {stats_list[0]['moving_avg_window']}
        - 이상치 임계값: {stats_list[0]['outlier_threshold']}
        - 정규화 방법: {stats_list[0]['scaler_type']}
        
        📈 처리 통계:
        - 평균 파일 크기: {total_original_samples/total_files:.0f}개 샘플
        - 데이터 형태: (샘플수, {total_features}개 특성)
        
        ✅ 모든 파일이 성공적으로 전처리되었습니다.
        """
        
        return report


def main():
    """메인 함수 - 사용 예제"""
    import glob
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 CSI 데이터 전처리 v4")
    print("=" * 50)
    
    # CSI 데이터 파일 찾기
    data_dir = "../csi_data"
    csv_files = []
    
    # case별로 파일 수집
    for case in ['case1', 'case2', 'case3']:
        case_dir = os.path.join(data_dir, case)
        if os.path.exists(case_dir):
            case_files = glob.glob(os.path.join(case_dir, "*.csv"))
            csv_files.extend(case_files)
    
    if not csv_files:
        print("❌ CSV 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 발견된 CSV 파일: {len(csv_files)}개")
    
    # 전처리기 초기화
    preprocessor = CSIPreprocessor(
        amplitude_start_col=8,
        amplitude_end_col=253,
        scaler_type='minmax'
    )
    
    # 출력 디렉토리 설정
    output_dir = "./processed_data"
    
    # 샘플 파일로 단일 파일 테스트
    print("\n🧪 단일 파일 전처리 테스트")
    test_file = csv_files[0]
    print(f"테스트 파일: {test_file}")
    
    try:
        processed_df, stats = preprocessor.process_single_file(
            file_path=test_file,
            moving_avg_window=5,
            outlier_threshold=3.0,
            fit_scaler=True,
            save_processed=False
        )
        
        print(f"✅ 전처리 완료!")
        print(f"   원본 형태: {stats['original_shape']}")
        print(f"   처리 후 형태: {stats['processed_shape']}")
        print(f"   Amplitude 특성 수: {stats['amplitude_features']}")
        
        # 전처리 효과 시각화 (처음 100개 샘플, 5개 특성)
        original_data = pd.read_csv(test_file).iloc[:, 8:253].values
        processed_data = processed_df.iloc[:, 8:253].values
        
        print("\n📊 전처리 효과 시각화")
        preprocessor.visualize_preprocessing_effects(
            original_data=original_data[:100, :5],
            processed_data=processed_data[:100, :5],
            sample_features=5,
            sample_length=100
        )
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return
    
    # 사용자 선택
    print(f"\n🚀 전체 파일 배치 처리를 진행하시겠습니까?")
    print(f"   처리할 파일 수: {len(csv_files)}개")
    print(f"   출력 디렉토리: {output_dir}")
    
    choice = input("계속하려면 'y'를 입력하세요 (y/n): ").lower()
    
    if choice != 'y':
        print("🔄 배치 처리를 취소했습니다.")
        return
    
    # 배치 처리 실행
    print(f"\n⚡ 배치 전처리 시작...")
    
    try:
        results = preprocessor.process_multiple_files(
            file_paths=csv_files,
            output_dir=output_dir,
            moving_avg_window=5,
            outlier_threshold=3.0,
            fit_scaler_on_first=True
        )
        
        # 결과 보고서 생성
        report = preprocessor.generate_processing_report(results['processing_stats'])
        print(report)
        
        # 실패한 파일이 있는 경우 출력
        if results['failed_files']:
            print("\n❌ 처리 실패 파일들:")
            for failed in results['failed_files']:
                print(f"   {failed['file']}: {failed['error']}")
        
        print(f"\n✅ 배치 전처리 완료!")
        print(f"   성공: {len(results['processed_files'])}개")
        print(f"   실패: {len(results['failed_files'])}개")
        print(f"   출력 위치: {output_dir}")
        
    except Exception as e:
        print(f"❌ 배치 처리 실패: {e}")


if __name__ == "__main__":
    main()
