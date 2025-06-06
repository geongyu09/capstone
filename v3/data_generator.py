# data_generator.py
"""
메모리 효율적 CSI 데이터 제너레이터
대용량 파일 처리를 위한 배치 단위 로딩
"""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
import logging
from config import CSIConfig

class CSIDataGenerator(Sequence):
    """메모리 효율적 CSI 데이터 제너레이터 클래스"""
    
    def __init__(self, file_list, batch_size=None, window_size=None, stride=None,
                 scaler=None, active_range=None, shuffle=True, logger=None):
        """
        Args:
            file_list: CSV 파일 경로 리스트
            batch_size: 배치 크기
            window_size: 윈도우 크기 (샘플 수)
            stride: 스트라이드 (샘플 수)
            scaler: 사전 학습된 스케일러
            active_range: 활성 특성 범위 (start_idx, end_idx)
            shuffle: 에포크마다 셔플 여부
            logger: 로거 객체
        """
        # 기본값 설정
        self.file_list = file_list
        self.batch_size = batch_size or CSIConfig.BATCH_SIZE
        self.window_size = window_size or CSIConfig.WINDOW_SIZE
        self.stride = stride or CSIConfig.STRIDE
        self.scaler = scaler
        self.active_range = active_range or CSIConfig.ACTIVE_FEATURE_RANGE
        self.shuffle = shuffle
        
        # 로깅 설정
        self.logger = logger or logging.getLogger(__name__)
        
        # 파일별 시퀀스 인덱스 미리 계산
        self.file_sequences = self._calculate_file_sequences()
        self.total_sequences = sum(len(seq) for seq in self.file_sequences.values())
        
        # 전체 인덱스 생성 (파일경로, 시작인덱스)
        self.sequence_indices = self._generate_sequence_indices()
        
        if self.shuffle:
            np.random.shuffle(self.sequence_indices)
        
        self.logger.info(f"📊 데이터 제너레이터 초기화 완료")
        self.logger.info(f"   파일 수: {len(self.file_list)}")
        self.logger.info(f"   총 시퀀스: {self.total_sequences:,}개")
        self.logger.info(f"   배치 수: {len(self)}개")
    
    def _calculate_file_sequences(self):
        """각 파일의 시퀀스 인덱스 미리 계산"""
        file_sequences = {}
        
        for file_path in self.file_list:
            try:
                # 파일 크기만 확인 (실제 로드는 나중에)
                df = pd.read_csv(file_path)
                n_samples = len(df)
                
                # 윈도우 크기보다 작은 파일은 스킵
                if n_samples < self.window_size:
                    self.logger.warning(f"파일 크기 부족으로 스킵: {file_path} ({n_samples} < {self.window_size})")
                    continue
                
                # 가능한 시퀀스 시작점들
                sequence_starts = list(range(0, n_samples - self.window_size + 1, self.stride))
                file_sequences[file_path] = sequence_starts
                
            except Exception as e:
                self.logger.warning(f"파일 읽기 실패로 스킵: {file_path} - {e}")
                continue
        
        return file_sequences
    
    def _generate_sequence_indices(self):
        """전역 시퀀스 인덱스 생성"""
        indices = []
        for file_path, starts in self.file_sequences.items():
            for start_idx in starts:
                indices.append((file_path, start_idx))
        return indices
    
    def __len__(self):
        """배치 수 반환"""
        return (self.total_sequences + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, batch_idx):
        """배치 데이터 생성"""
        # 현재 배치의 시퀀스 인덱스들
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_sequences)
        
        batch_sequences = []
        batch_labels = []
        
        for i in range(start_idx, end_idx):
            file_path, seq_start = self.sequence_indices[i]
            
            try:
                X_seq, y_seq = self._load_sequence(file_path, seq_start)
                batch_sequences.append(X_seq)
                batch_labels.append(y_seq)
            except Exception as e:
                self.logger.warning(f"시퀀스 로드 실패: {file_path}[{seq_start}] - {e}")
                # 더미 데이터로 대체
                dummy_features = self.active_range[1] - self.active_range[0] + 1
                batch_sequences.append(np.zeros((self.window_size, dummy_features)))
                batch_labels.append(0)
        
        return np.array(batch_sequences), np.array(batch_labels)
    
    def _load_sequence(self, file_path, start_idx):
        """파일에서 특정 시퀀스만 로드"""
        # 필요한 행만 로드 (메모리 절약)
        end_idx = start_idx + self.window_size
        
        try:
            # pandas로 특정 범위만 읽기
            df = pd.read_csv(file_path, skiprows=range(1, start_idx + 1), nrows=self.window_size)
            
            # 데이터가 부족한 경우 패딩
            if len(df) < self.window_size:
                padding_rows = self.window_size - len(df)
                if len(df) > 0:
                    last_row = df.iloc[-1:].copy()
                    for _ in range(padding_rows):
                        df = pd.concat([df, last_row], ignore_index=True)
                else:
                    # 완전히 빈 경우 더미 데이터
                    raise ValueError("빈 데이터")
            
            # 특성 추출
            feature_cols = [col for col in df.columns if col.startswith('feat_')]
            if not feature_cols:
                raise ValueError("특성 컬럼이 없음")
            
            X = df[feature_cols].values
            y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
            
            # 활성 특성만 선택
            start_feat, end_feat = self.active_range
            if X.shape[1] > end_feat:
                X_active = X[:, start_feat:end_feat+1]
            else:
                # 특성 수가 부족한 경우 처리
                available_features = min(X.shape[1], end_feat - start_feat + 1)
                X_active = X[:, start_feat:start_feat + available_features]
                
                # 부족한 특성은 0으로 패딩
                if X_active.shape[1] < (end_feat - start_feat + 1):
                    padding_features = (end_feat - start_feat + 1) - X_active.shape[1]
                    padding = np.zeros((X_active.shape[0], padding_features))
                    X_active = np.hstack([X_active, padding])
            
            # 정규화 (사전 학습된 스케일러 사용)
            if self.scaler:
                X_normalized = self.scaler.transform(X_active)
            else:
                X_normalized = X_active
            
            # 고급 라벨링
            sequence_label = self._generate_sequence_label(y)
            
            return X_normalized, sequence_label
            
        except Exception as e:
            # 에러 시 더미 데이터 반환
            self.logger.debug(f"시퀀스 로드 에러: {file_path}[{start_idx}] - {e}")
            dummy_features = self.active_range[1] - self.active_range[0] + 1
            return np.zeros((self.window_size, dummy_features)), 0
    
    def _generate_sequence_label(self, y):
        """고급 시퀀스 라벨링"""
        fall_ratio = np.mean(y == 1)
        fall_positions = np.where(y == 1)[0]
        
        sequence_label = 0
        
        if len(fall_positions) > 0:
            # 1. 충분한 낙상 비율
            if fall_ratio >= CSIConfig.OVERLAP_THRESHOLD:
                sequence_label = 1
            
            # 2. 낙상 시작 패턴 (윈도우 후반부에 낙상 시작)
            elif fall_positions[-1] >= len(y) * 0.7:
                sequence_label = 1
            
            # 3. 낙상 진행 중 패턴 (윈도우 전반부에 낙상)
            elif fall_positions[0] <= len(y) * 0.3 and fall_ratio >= 0.1:
                sequence_label = 1
            
            # 4. 연속성 고려
            elif (fall_ratio >= 0.05 and 
                  len(fall_positions) > 0 and
                  (fall_positions[-1] - fall_positions[0] + 1) >= len(fall_positions) * 0.8):
                sequence_label = 1
        
        return sequence_label
    
    def on_epoch_end(self):
        """에포크 종료 시 셔플"""
        if self.shuffle:
            np.random.shuffle(self.sequence_indices)
    
    def get_statistics(self):
        """데이터 제너레이터 통계 정보"""
        file_sizes = []
        sequence_counts = []
        
        for file_path, sequences in self.file_sequences.items():
            try:
                file_size = len(pd.read_csv(file_path))
                file_sizes.append(file_size)
                sequence_counts.append(len(sequences))
            except:
                continue
        
        stats = {
            'total_files': len(self.file_list),
            'valid_files': len(self.file_sequences),
            'total_sequences': self.total_sequences,
            'avg_file_size': np.mean(file_sizes) if file_sizes else 0,
            'avg_sequences_per_file': np.mean(sequence_counts) if sequence_counts else 0,
            'batch_count': len(self)
        }
        
        return stats
    
    def print_statistics(self):
        """통계 정보 출력"""
        stats = self.get_statistics()
        
        print(f"📊 데이터 제너레이터 통계:")
        print(f"   총 파일: {stats['total_files']}개")
        print(f"   유효 파일: {stats['valid_files']}개")
        print(f"   총 시퀀스: {stats['total_sequences']:,}개")
        print(f"   평균 파일 크기: {stats['avg_file_size']:.0f}개 샘플")
        print(f"   파일당 평균 시퀀스: {stats['avg_sequences_per_file']:.0f}개")
        print(f"   총 배치: {stats['batch_count']}개")

if __name__ == "__main__":
    # 테스트 코드
    import glob
    
    # 테스트 파일 찾기
    test_files = glob.glob("*.csv")[:3]  # 최대 3개 파일로 테스트
    
    if test_files:
        print("🧪 CSI 데이터 제너레이터 테스트")
        print("=" * 40)
        
        # 제너레이터 생성
        generator = CSIDataGenerator(
            file_list=test_files,
            batch_size=4,
            shuffle=True
        )
        
        # 통계 출력
        generator.print_statistics()
        
        # 첫 번째 배치 테스트
        print(f"\n🔍 첫 번째 배치 테스트:")
        X_batch, y_batch = generator[0]
        print(f"   배치 형태: X={X_batch.shape}, y={y_batch.shape}")
        print(f"   라벨 분포: {np.bincount(y_batch)}")
        
        print("✅ 테스트 완료!")
    else:
        print("❌ 테스트할 CSV 파일이 없습니다!")