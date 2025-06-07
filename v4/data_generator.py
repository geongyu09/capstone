"""
CSI 낙상 감지 v4 - 데이터 제너레이터
메모리 효율적인 시퀀스 데이터 생성
"""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
import os
import glob
from typing import List, Tuple, Optional
from config import Config
from utils import setup_logging


class CSIDataGenerator(Sequence):
    """CSI 데이터를 위한 시퀀스 제너레이터"""
    
    def __init__(self, 
                 file_paths: List[str],
                 batch_size: int = Config.BATCH_SIZE,
                 window_size: int = Config.WINDOW_SIZE,
                 stride: int = Config.STRIDE,
                 shuffle: bool = True,
                 scaler=None,
                 logger=None):
        """
        Args:
            file_paths: 전처리된 CSV 파일 경로 리스트
            batch_size: 배치 크기
            window_size: 시퀀스 윈도우 크기
            stride: 윈도우 이동 간격
            shuffle: 에포크마다 셔플 여부
            scaler: 전처리에 사용된 스케일러 (이미 적용된 경우 None)
            logger: 로거 객체
        """
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        self.shuffle = shuffle
        self.scaler = scaler
        self.logger = logger or setup_logging()
        
        # 각 파일의 시퀀스 정보 계산
        self.sequences = self._calculate_sequences()
        self.total_sequences = len(self.sequences)
        
        # 인덱스 초기화
        self.indices = np.arange(self.total_sequences)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.logger.info(f"📊 데이터 제너레이터 초기화 완료")
        self.logger.info(f"   파일 수: {len(self.file_paths)}")
        self.logger.info(f"   총 시퀀스: {self.total_sequences:,}개")
        self.logger.info(f"   배치 수: {len(self)}개")
    
    def _calculate_sequences(self) -> List[Tuple[str, int, int]]:
        """각 파일에서 생성 가능한 시퀀스들을 미리 계산"""
        sequences = []
        
        for file_path in self.file_paths:
            try:
                # 파일 크기만 확인 (전체 로드 없이)
                df_info = pd.read_csv(file_path, nrows=0)  # 헤더만 읽기
                
                # 실제 데이터 행 수 확인을 위해 한 번은 읽어야 함
                with open(file_path, 'r') as f:
                    n_rows = sum(1 for line in f) - 1  # 헤더 제외
                
                if n_rows < self.window_size:
                    self.logger.warning(f"파일 크기 부족으로 스킵: {file_path} ({n_rows} < {self.window_size})")
                    continue
                
                # 가능한 시퀀스 시작점들 계산
                max_start = n_rows - self.window_size
                for start_idx in range(0, max_start + 1, self.stride):
                    end_idx = start_idx + self.window_size
                    sequences.append((file_path, start_idx, end_idx))
                
            except Exception as e:
                self.logger.warning(f"파일 처리 실패: {file_path} - {e}")
        
        return sequences
    
    def __len__(self) -> int:
        """배치 수 반환"""
        return int(np.ceil(self.total_sequences / self.batch_size))
    
    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """배치 데이터 생성"""
        # 현재 배치의 인덱스들
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_sequences)
        batch_indices = self.indices[start_idx:end_idx]
        
        # 배치 데이터 수집
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            file_path, seq_start, seq_end = self.sequences[idx]
            
            try:
                X_seq, y_seq = self._load_sequence(file_path, seq_start, seq_end)
                X_batch.append(X_seq)
                y_batch.append(y_seq)
            except Exception as e:
                self.logger.warning(f"시퀀스 로드 실패: {file_path}[{seq_start}:{seq_end}] - {e}")
                # 더미 데이터로 대체
                X_batch.append(np.zeros((self.window_size, Config.TOTAL_FEATURES)))
                y_batch.append(0)
        
        return np.array(X_batch), np.array(y_batch)
    
    def _load_sequence(self, file_path: str, start_idx: int, end_idx: int) -> Tuple[np.ndarray, int]:
        """특정 시퀀스 로드"""
        # 필요한 행만 로드
        df = pd.read_csv(file_path, skiprows=range(1, start_idx + 1), nrows=self.window_size)
        
        # Amplitude 데이터 추출
        amplitude_cols = df.columns[Config.AMPLITUDE_START_COL:Config.AMPLITUDE_END_COL]
        X = df[amplitude_cols].values
        
        # 라벨 처리
        if 'label' in df.columns:
            labels = df['label'].values
            # 시퀀스 라벨링: 낙상이 일정 비율 이상 포함되면 낙상 시퀀스로 분류
            y = 1 if np.mean(labels) >= Config.OVERLAP_THRESHOLD else 0
        else:
            y = 0  # 라벨이 없는 경우 정상으로 분류
        
        # 데이터 검증
        if X.shape[0] != self.window_size:
            raise ValueError(f"시퀀스 길이 불일치: {X.shape[0]} != {self.window_size}")
        
        if X.shape[1] != Config.TOTAL_FEATURES:
            raise ValueError(f"특성 수 불일치: {X.shape[1]} != {Config.TOTAL_FEATURES}")
        
        return X, y
    
    def on_epoch_end(self):
        """에포크 종료 시 인덱스 셔플"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_class_distribution(self) -> dict:
        """클래스 분포 계산"""
        labels = []
        
        # 모든 시퀀스의 라벨 수집 (시간이 오래 걸릴 수 있음)
        self.logger.info("클래스 분포 계산 중...")
        
        for i, (file_path, start_idx, end_idx) in enumerate(self.sequences):
            try:
                _, y = self._load_sequence(file_path, start_idx, end_idx)
                labels.append(y)
                
                if (i + 1) % 1000 == 0:
                    self.logger.info(f"진행률: {i+1}/{len(self.sequences)}")
                    
            except Exception as e:
                self.logger.warning(f"라벨 로드 실패: {file_path}[{start_idx}:{end_idx}] - {e}")
                labels.append(0)
        
        labels = np.array(labels)
        distribution = {
            'normal': np.sum(labels == 0),
            'fall': np.sum(labels == 1),
            'total': len(labels)
        }
        
        if distribution['total'] > 0:
            distribution['fall_ratio'] = distribution['fall'] / distribution['total']
        else:
            distribution['fall_ratio'] = 0.0
        
        return distribution
    
    def print_statistics(self):
        """통계 정보 출력"""
        print(f"📊 데이터 제너레이터 통계:")
        print(f"   파일 수: {len(self.file_paths)}")
        print(f"   총 시퀀스: {self.total_sequences:,}개")
        print(f"   배치 크기: {self.batch_size}")
        print(f"   배치 수: {len(self)}개")
        print(f"   윈도우 크기: {self.window_size}")
        print(f"   스트라이드: {self.stride}")
        print(f"   특성 수: {Config.TOTAL_FEATURES}")


def create_data_generators(processed_data_dir: str = Config.PROCESSED_DATA_DIR,
                          train_ratio: float = Config.TRAIN_RATIO,
                          val_ratio: float = Config.VAL_RATIO,
                          test_ratio: float = Config.TEST_RATIO,
                          random_seed: int = 42) -> Tuple[CSIDataGenerator, CSIDataGenerator, CSIDataGenerator]:
    """학습/검증/테스트 데이터 제너레이터 생성"""
    
    logger = setup_logging()
    
    # 전처리된 파일들 수집
    processed_files = glob.glob(os.path.join(processed_data_dir, "*_processed.csv"))
    
    if not processed_files:
        raise ValueError(f"전처리된 파일을 찾을 수 없습니다: {processed_data_dir}")
    
    logger.info(f"전처리된 파일 {len(processed_files)}개 발견")
    
    # 파일 단위로 데이터 분할
    from utils import split_data_by_files
    train_files, val_files, test_files = split_data_by_files(
        processed_files, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    logger.info(f"데이터 분할: 훈련 {len(train_files)}개, 검증 {len(val_files)}개, 테스트 {len(test_files)}개")
    
    # 제너레이터 생성
    train_gen = CSIDataGenerator(
        file_paths=train_files,
        batch_size=Config.BATCH_SIZE,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        shuffle=True,
        logger=logger
    )
    
    val_gen = CSIDataGenerator(
        file_paths=val_files,
        batch_size=Config.BATCH_SIZE,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        shuffle=False,  # 검증 데이터는 셔플하지 않음
        logger=logger
    )
    
    test_gen = CSIDataGenerator(
        file_paths=test_files,
        batch_size=Config.BATCH_SIZE,
        window_size=Config.WINDOW_SIZE,
        stride=Config.STRIDE,
        shuffle=False,  # 테스트 데이터는 셔플하지 않음
        logger=logger
    )
    
    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 CSI 데이터 제너레이터 테스트")
    print("=" * 50)
    
    try:
        # 제너레이터 생성
        train_gen, val_gen, test_gen = create_data_generators()
        
        # 통계 출력
        print("\n📊 훈련 데이터:")
        train_gen.print_statistics()
        
        print("\n📊 검증 데이터:")
        val_gen.print_statistics()
        
        print("\n📊 테스트 데이터:")
        test_gen.print_statistics()
        
        # 첫 번째 배치 테스트
        print(f"\n🔍 첫 번째 배치 테스트:")
        X_batch, y_batch = train_gen[0]
        print(f"   X 형태: {X_batch.shape}")
        print(f"   y 형태: {y_batch.shape}")
        print(f"   y 분포: {np.bincount(y_batch)}")
        
        print("\n✅ 데이터 제너레이터 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
