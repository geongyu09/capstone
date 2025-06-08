"""
향상된 CSI 낙상 감지 모델 (수정된 버전)
어텐션 메커니즘과 잔차 연결을 포함한 개선된 아키텍처
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling1D, Concatenate,
    Bidirectional, Add, Lambda, Layer
)
from tensorflow.keras.optimizers import Adam
import numpy as np

from config import Config, ModelConfig
from utils import setup_logging

class AttentionLayer(Layer):
    """개선된 셀프 어텐션 레이어"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: (batch_size, sequence_length, features)
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, -1)
        weighted_input = inputs * a
        output = tf.reduce_sum(weighted_input, axis=1)
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss 클래스"""
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # 클리핑으로 수치적 안정성 확보
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Focal Loss 계산
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.pow((1 - p_t), self.gamma) * tf.math.log(p_t)
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config

def residual_block(x, filters, kernel_size=3, dropout_rate=0.3):
    """잔차 블록"""
    shortcut = x
    
    # 첫 번째 컨볼루션
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # 두 번째 컨볼루션
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # 차원이 다르면 shortcut 조정
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    
    # 잔차 연결
    x = Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def multi_scale_cnn_branch(inputs):
    """다중 스케일 CNN 브랜치"""
    cnn_outputs = []
    
    # 다양한 커널 크기로 특성 추출
    for kernel_size in [3, 5, 7]:
        x = inputs
        
        # 첫 번째 잔차 블록
        x = residual_block(x, 64, kernel_size)
        x = MaxPooling1D(2)(x)
        
        # 두 번째 잔차 블록
        x = residual_block(x, 128, kernel_size)
        x = MaxPooling1D(2)(x)
        
        # 전역 평균 풀링
        x = GlobalAveragePooling1D()(x)
        
        cnn_outputs.append(x)
    
    # 다중 스케일 특성 결합
    if len(cnn_outputs) > 1:
        combined = Concatenate()(cnn_outputs)
    else:
        combined = cnn_outputs[0]
    
    return combined

def attention_lstm_branch(inputs):
    """어텐션이 적용된 LSTM 브랜치"""
    
    # Bidirectional LSTM
    lstm_out = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(inputs)
    
    # 커스텀 어텐션 적용
    attention_layer = AttentionLayer(units=128)
    context_vector = attention_layer(lstm_out)
    
    return context_vector

def build_improved_model(input_shape=(50, 245)):
    """향상된 하이브리드 모델 구축"""
    
    logger = setup_logging()
    logger.info("🧠 향상된 모델 아키텍처 구축 시작")
    
    inputs = Input(shape=input_shape, name='input')
    
    # 다중 스케일 CNN 브랜치
    cnn_features = multi_scale_cnn_branch(inputs)
    
    # 어텐션 LSTM 브랜치
    lstm_features = attention_lstm_branch(inputs)
    
    # 특성 결합
    combined_features = Concatenate()([cnn_features, lstm_features])
    
    # 분류기
    x = Dense(256, activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ImprovedCSIFallDetection')
    
    logger.info("✅ 향상된 모델 아키텍처 구축 완료")
    
    return model

def compile_improved_model(model, learning_rate=0.001, use_focal_loss=True):
    """향상된 모델 컴파일"""
    
    # 옵티마이저
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # 손실 함수
    if use_focal_loss:
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss_name = "focal_loss"
    else:
        # 가중치 적용 binary crossentropy
        def weighted_bce(y_true, y_pred):
            class_weight_1 = 2.0
            class_weight_0 = 1.0
            weights = y_true * class_weight_1 + (1 - y_true) * class_weight_0
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            return bce * weights
        
        loss = weighted_bce
        loss_name = "weighted_binary_crossentropy"
    
    # 메트릭
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"✅ 모델 컴파일 완료 (손실 함수: {loss_name})")
    
    return model

def create_lightweight_model(input_shape=(50, 245)):
    """실시간 추론을 위한 경량화 모델"""
    
    inputs = Input(shape=input_shape, name='input')
    
    # 깊이별 분리 가능한 컨볼루션
    x = tf.keras.layers.SeparableConv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    x = tf.keras.layers.SeparableConv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    # 단순한 LSTM
    x = LSTM(64, dropout=0.2)(x)
    
    # 압축된 분류기
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LightweightCSIFallDetection')
    
    return model

def create_baseline_model(input_shape=(50, 245)):
    """기본 비교 모델"""
    
    inputs = Input(shape=input_shape, name='input')
    
    # 간단한 LSTM
    x = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='BaselineCSIModel')
    
    return model

def model_comparison_test():
    """모델 아키텍처 비교 테스트"""
    
    print("🧪 모델 아키텍처 비교 테스트")
    print("=" * 50)
    
    input_shape = (Config.WINDOW_SIZE, Config.TOTAL_FEATURES)
    
    models = {
        'Baseline': create_baseline_model(input_shape),
        'Improved': build_improved_model(input_shape),
        'Lightweight': create_lightweight_model(input_shape)
    }
    
    # 모델별 정보 출력
    for name, model in models.items():
        print(f"\n📊 {name} 모델:")
        print(f"   총 파라미터: {model.count_params():,}")
        
        # 더미 입력으로 추론 시간 테스트
        dummy_input = np.random.randn(1, *input_shape)
        
        import time
        start_time = time.time()
        for _ in range(10):  # 10번 추론
            _ = model.predict(dummy_input, verbose=0)
        inference_time = (time.time() - start_time) / 10
        
        print(f"   추론 시간: {inference_time*1000:.2f}ms")
        print(f"   모델 크기 (추정): {model.count_params() * 4 / 1024 / 1024:.1f}MB")
        
        # 모델 요약 (간단히)
        print(f"   레이어 수: {len(model.layers)}")
    
    return models

def save_improved_model(model, model_name="improved_csi_model"):
    """향상된 모델 저장"""
    
    from utils import create_timestamp, save_model_artifacts
    
    experiment_name = f"{model_name}_{create_timestamp()}"
    
    # 메타데이터 생성
    metadata = {
        'experiment_name': experiment_name,
        'model_type': 'improved_hybrid',
        'timestamp': create_timestamp(),
        'architecture': 'CNN+LSTM with Attention',
        'config': {
            'window_size': Config.WINDOW_SIZE,
            'total_features': Config.TOTAL_FEATURES,
            'batch_size': Config.BATCH_SIZE
        },
        'improvements': [
            'Multi-scale CNN',
            'Bidirectional LSTM with Attention',
            'Residual connections',
            'Focal Loss'
        ],
        'created_by': 'improved_model.py'
    }
    
    try:
        # 모델 저장
        model_path = f"./models/{experiment_name}.keras"
        model.save(model_path)
        
        # 메타데이터 저장
        import json
        metadata_path = f"./models/{experiment_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 모델 저장 완료:")
        print(f"   모델: {model_path}")
        print(f"   메타데이터: {metadata_path}")
        
        return experiment_name
        
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")
        return None

def train_improved_model():
    """향상된 모델 학습을 위한 준비"""
    print("🚀 향상된 모델 학습 준비")
    
    # 모델 생성 및 컴파일
    model = build_improved_model()
    model = compile_improved_model(model, use_focal_loss=True)
    
    # 모델 요약 출력
    print("\n📋 모델 아키텍처:")
    model.summary()
    
    # 모델 저장
    experiment_name = save_improved_model(model)
    
    if experiment_name:
        print(f"\n✅ 향상된 모델 준비 완료: {experiment_name}")
        print("\n📝 다음 단계:")
        print("1. 기존 trainer.py를 수정하여 새 모델 사용")
        print("2. 또는 다음 명령으로 학습 실행:")
        print(f"   # model = load_model('./models/{experiment_name}.keras')")
        
    return model, experiment_name

if __name__ == "__main__":
    print("🧠 향상된 CSI 낙상 감지 모델")
    print("=" * 50)
    
    try:
        # 모델 비교 테스트
        models = model_comparison_test()
        
        # 사용자 선택
        print("\n🤔 어떤 작업을 수행하시겠습니까?")
        print("1. 향상된 모델 아키텍처만 확인")
        print("2. 향상된 모델 저장 및 학습 준비")
        print("3. 경량화 모델 테스트")
        print("4. 모든 모델 비교")
        
        choice = input("선택 (1-4): ").strip()
        
        if choice == "1":
            improved_model = build_improved_model()
            print("\n📋 향상된 모델 아키텍처:")
            improved_model.summary()
            
        elif choice == "2":
            model, experiment_name = train_improved_model()
            
        elif choice == "3":
            lightweight_model = create_lightweight_model()
            print("\n📋 경량화 모델 아키텍처:")
            lightweight_model.summary()
            
        elif choice == "4":
            print("\n📊 모든 모델이 성공적으로 생성되었습니다!")
            for name in models.keys():
                print(f"   ✅ {name} 모델")
            
        else:
            print("올바른 선택이 아닙니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
