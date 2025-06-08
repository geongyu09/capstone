"""
CSI 낙상 감지 v4 - 모델 아키텍처
CNN + LSTM 하이브리드 모델
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling1D, Concatenate,
    TimeDistributed, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Tuple, Dict, Any
import numpy as np

from config import Config, ModelConfig
from utils import setup_logging


# TensorFlow 2.x 호환 커스텀 손실 함수들
def weighted_binary_crossentropy_factory(class_weights):
    """
    클래스 가중치를 적용한 binary crossentropy 손실 함수를 반환하는 팩토리 함수
    
    Args:
        class_weights: {0: weight_for_class_0, 1: weight_for_class_1} 형태의 딕셔너리
    
    Returns:
        가중치가 적용된 손실 함수
    """
    def loss_function(y_true, y_pred):
        # 클래스 가중치 적용
        weight_0 = class_weights.get(0, 1.0)
        weight_1 = class_weights.get(1, 1.0)
        
        # 가중치 적용
        weights = y_true * weight_1 + (1 - y_true) * weight_0
        
        # Binary crossentropy with weights
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce * weights
    
    # 함수에 이름 속성 추가 (저장/로드를 위해)
    loss_function.__name__ = 'weighted_binary_crossentropy'
    return loss_function


def simple_weighted_binary_crossentropy(y_true, y_pred):
    """
    간단한 가중치 적용 binary crossentropy
    기본적으로 클래스 1에 더 높은 가중치를 적용 (낙상 감지를 위해)
    """
    # 낙상(1)에 더 높은 가중치 적용
    class_weight_1 = 2.0  # 낙상 클래스에 2배 가중치
    class_weight_0 = 1.0  # 정상 클래스
    
    weights = y_true * class_weight_1 + (1 - y_true) * class_weight_0
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce * weights


# 전역 커스텀 객체 딕셔너리 (모델 로드시 사용)
CUSTOM_OBJECTS = {
    'weighted_binary_crossentropy': simple_weighted_binary_crossentropy,
    'simple_weighted_binary_crossentropy': simple_weighted_binary_crossentropy
}


class CSIFallDetectionModel:
    """CSI 낙상 감지 모델 빌더"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 model_config: Dict[str, Any] = None,
                 logger=None):
        """
        Args:
            input_shape: (window_size, n_features) 형태
            model_config: 모델 설정 딕셔너리
            logger: 로거 객체
        """
        self.input_shape = input_shape
        self.model_config = model_config or self._get_default_config()
        self.logger = logger or setup_logging()
        self.model = None
        
        self.logger.info(f"🧠 모델 빌더 초기화")
        self.logger.info(f"   입력 형태: {input_shape}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 모델 설정"""
        return {
            'cnn_filters': ModelConfig.CNN_FILTERS,
            'cnn_kernel_size': ModelConfig.CNN_KERNEL_SIZE,
            'cnn_dropout': ModelConfig.CNN_DROPOUT,
            'lstm_units': ModelConfig.LSTM_UNITS,
            'lstm_dropout': ModelConfig.LSTM_DROPOUT,
            'lstm_recurrent_dropout': ModelConfig.LSTM_RECURRENT_DROPOUT,
            'dense_units': ModelConfig.DENSE_UNITS,
            'dense_dropout': ModelConfig.DENSE_DROPOUT,
            'output_activation': ModelConfig.OUTPUT_ACTIVATION
        }
    
    def build_cnn_lstm_model(self) -> Model:
        """CNN + LSTM 하이브리드 모델 구축"""
        
        # 입력 레이어
        inputs = Input(shape=self.input_shape, name='input')
        
        # CNN 브랜치
        cnn_branch = self._build_cnn_branch(inputs)
        
        # LSTM 브랜치
        lstm_branch = self._build_lstm_branch(inputs)
        
        # 브랜치 결합
        if cnn_branch is not None and lstm_branch is not None:
            combined = Concatenate(name='concat')([cnn_branch, lstm_branch])
        elif cnn_branch is not None:
            combined = cnn_branch
        else:
            combined = lstm_branch
        
        # 최종 분류 레이어
        outputs = self._build_classifier(combined)
        
        # 모델 생성
        model = Model(inputs=inputs, outputs=outputs, name='CSI_Fall_Detection')
        
        self.model = model
        self.logger.info(f"✅ 모델 구축 완료")
        
        return model
    
    def _build_cnn_branch(self, inputs) -> tf.Tensor:
        """CNN 브랜치 구축"""
        x = inputs
        
        # CNN 레이어들
        for i, filters in enumerate(self.model_config['cnn_filters']):
            x = Conv1D(
                filters=filters,
                kernel_size=self.model_config['cnn_kernel_size'],
                activation='relu',
                padding='same',
                name=f'cnn_conv_{i+1}'
            )(x)
            
            x = BatchNormalization(name=f'cnn_bn_{i+1}')(x)
            
            x = MaxPooling1D(
                pool_size=2,
                name=f'cnn_pool_{i+1}'
            )(x)
            
            x = Dropout(
                self.model_config['cnn_dropout'],
                name=f'cnn_dropout_{i+1}'
            )(x)
        
        # Global Average Pooling
        cnn_output = GlobalAveragePooling1D(name='cnn_gap')(x)
        
        return cnn_output
    
    def _build_lstm_branch(self, inputs) -> tf.Tensor:
        """LSTM 브랜치 구축"""
        x = inputs
        
        # LSTM 레이어들
        for i, units in enumerate(self.model_config['lstm_units']):
            return_sequences = (i < len(self.model_config['lstm_units']) - 1)
            
            x = Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.model_config['lstm_dropout'],
                    recurrent_dropout=self.model_config['lstm_recurrent_dropout'],
                    name=f'lstm_{i+1}'
                ),
                name=f'bi_lstm_{i+1}'
            )(x)
            
            if return_sequences:
                x = BatchNormalization(name=f'lstm_bn_{i+1}')(x)
        
        return x
    
    def _build_classifier(self, features) -> tf.Tensor:
        """분류기 구축"""
        x = features
        
        # Dense 레이어들
        for i, units in enumerate(self.model_config['dense_units']):
            x = Dense(
                units=units,
                activation='relu',
                name=f'dense_{i+1}'
            )(x)
            
            x = BatchNormalization(name=f'dense_bn_{i+1}')(x)
            
            x = Dropout(
                self.model_config['dense_dropout'],
                name=f'dense_dropout_{i+1}'
            )(x)
        
        # 출력 레이어
        outputs = Dense(
            units=1,
            activation=self.model_config['output_activation'],
            name='output'
        )(x)
        
        return outputs
    
    def compile_model(self, 
                     learning_rate: float = Config.LEARNING_RATE,
                     class_weights: Dict[int, float] = None) -> None:
        """모델 컴파일"""
        
        if self.model is None:
            raise ValueError("모델이 아직 구축되지 않았습니다. build_model()를 먼저 호출하세요.")
        
        # 옵티마이저
        optimizer = Adam(learning_rate=learning_rate)
        
        # 손실 함수 (클래스 불균형 고려)
        if class_weights:
            # 팩토리 함수 사용
            loss = weighted_binary_crossentropy_factory(class_weights)
            loss_name = "weighted_binary_crossentropy"
        else:
            # 간단한 가중치 적용 함수 사용
            loss = simple_weighted_binary_crossentropy
            loss_name = "simple_weighted_binary_crossentropy"
        
        # 메트릭
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"✅ 모델 컴파일 완료")
        self.logger.info(f"   옵티마이저: Adam (lr={learning_rate})")
        self.logger.info(f"   손실 함수: {loss_name}")
        self.logger.info(f"   메트릭: {[m.name if hasattr(m, 'name') else str(m) for m in metrics]}")
    
    def get_callbacks(self, 
                     model_name: str,
                     monitor: str = 'val_loss',
                     patience: int = 10) -> list:
        """학습 콜백 생성"""
        
        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint
            ModelCheckpoint(
                filepath=f"{Config.MODEL_DIR}/{model_name}_best.keras",
                monitor=monitor,
                save_best_only=True,
                verbose=1
            ),
            
            # Learning Rate Reduction
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def print_model_summary(self) -> None:
        """모델 요약 출력"""
        if self.model is None:
            self.logger.warning("모델이 구축되지 않았습니다.")
            return
        
        print("\n🧠 모델 아키텍처 요약")
        print("=" * 50)
        self.model.summary()
        
        # 파라미터 수 계산
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n📊 파라미터 정보:")
        print(f"   총 파라미터: {total_params:,}")
        print(f"   학습 가능: {trainable_params:,}")
        print(f"   고정: {non_trainable_params:,}")
    
    def create_simple_model(self) -> Model:
        """간단한 baseline 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 간단한 LSTM
        x = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(inputs)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='Simple_CSI_Model')
        self.model = model
        
        return model
    
    def create_cnn_only_model(self) -> Model:
        """CNN만 사용하는 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        x = inputs
        for i, filters in enumerate([64, 128, 256]):
            x = Conv1D(filters, 3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(0.3)(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNN_Only_Model')
        self.model = model
        
        return model


def create_model(model_type: str = 'hybrid',
                input_shape: Tuple[int, int] = None,
                learning_rate: float = Config.LEARNING_RATE,
                class_weights: Dict[int, float] = None) -> Tuple[Model, CSIFallDetectionModel]:
    """모델 생성 헬퍼 함수"""
    
    if input_shape is None:
        input_shape = (Config.WINDOW_SIZE, Config.TOTAL_FEATURES)
    
    logger = setup_logging()
    
    # 모델 빌더 생성
    model_builder = CSIFallDetectionModel(input_shape=input_shape, logger=logger)
    
    # 모델 타입에 따라 구축
    if model_type == 'hybrid':
        model = model_builder.build_cnn_lstm_model()
    elif model_type == 'simple':
        model = model_builder.create_simple_model()
    elif model_type == 'cnn':
        model = model_builder.create_cnn_only_model()
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 모델 컴파일
    model_builder.compile_model(learning_rate=learning_rate, class_weights=class_weights)
    
    logger.info(f"✅ {model_type} 모델 생성 완료")
    
    return model, model_builder


# 모델 로드를 위한 커스텀 객체 getter 함수
def get_custom_objects():
    """모델 로드시 사용할 커스텀 객체들 반환"""
    return CUSTOM_OBJECTS.copy()


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 CSI 모델 아키텍처 테스트")
    print("=" * 50)
    
    # 입력 형태 설정
    input_shape = (Config.WINDOW_SIZE, Config.TOTAL_FEATURES)
    print(f"입력 형태: {input_shape}")
    
    # 다양한 모델 테스트
    model_types = ['simple', 'cnn', 'hybrid']
    
    for model_type in model_types:
        print(f"\n🔍 {model_type.upper()} 모델 테스트:")
        
        try:
            model, builder = create_model(model_type=model_type, input_shape=input_shape)
            builder.print_model_summary()
            
            # 더미 데이터로 예측 테스트
            dummy_input = np.random.randn(1, *input_shape)
            output = model.predict(dummy_input, verbose=0)
            print(f"✅ 예측 테스트 성공: 출력 형태 {output.shape}")
            
        except Exception as e:
            print(f"❌ {model_type} 모델 테스트 실패: {e}")
    
    print(f"\n✅ 모델 아키텍처 테스트 완료!")
