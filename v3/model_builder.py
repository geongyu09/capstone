# model_builder.py
"""
CNN+LSTM 하이브리드 모델 구축기
고주파 CSI 데이터에 최적화된 아키텍처
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Dropout, 
    BatchNormalization, GlobalAveragePooling1D, Concatenate,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging
from config import CSIConfig

class CSIModelBuilder:
    """CSI 낙상 감지 모델 구축 클래스"""
    
    def __init__(self, input_shape=None, logger=None):
        """
        Args:
            input_shape: 입력 형태 (window_size, feature_count)
            logger: 로거 객체
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 입력 형태 설정
        if input_shape is None:
            feature_count = CSIConfig.ACTIVE_FEATURE_COUNT
            self.input_shape = (CSIConfig.WINDOW_SIZE, feature_count)
        else:
            self.input_shape = input_shape
        
        self.model = None
        self.model_config = CSIConfig.get_model_config()
        
        self.logger.info(f"🏗️ 모델 빌더 초기화: 입력 형태 {self.input_shape}")
    
    def build_basic_lstm(self):
        """기본 LSTM 모델"""
        self.logger.info("📦 기본 LSTM 모델 구축...")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=self.input_shape, name='lstm_1'),
            Dropout(0.4),
            
            LSTM(32, return_sequences=False, name='lstm_2'),
            Dropout(0.4),
            
            Dense(16, activation='relu', name='dense_1'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid', name='output')
        ])
        
        self.model = model
        self._compile_model()
        
        self.logger.info(f"✅ 기본 LSTM 모델 구축 완료: {model.count_params():,} 파라미터")
        return model
    
    def build_cnn_lstm_hybrid(self):
        """CNN+LSTM 하이브리드 모델 (권장)"""
        self.logger.info("🚀 CNN+LSTM 하이브리드 모델 구축...")
        
        model = Sequential([
            # CNN 부분 - 지역적 패턴 추출
            Conv1D(filters=self.model_config['cnn_filters'][0], 
                   kernel_size=self.model_config['cnn_kernel_sizes'][0], 
                   activation='relu', 
                   input_shape=self.input_shape, 
                   name='conv1d_1'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(self.model_config['dropout_rates'][0]),
            
            Conv1D(filters=self.model_config['cnn_filters'][1], 
                   kernel_size=self.model_config['cnn_kernel_sizes'][1], 
                   activation='relu', 
                   name='conv1d_2'),
            BatchNormalization(),
            Dropout(self.model_config['dropout_rates'][0]),
            
            # LSTM 부분 - 시계열 패턴 학습
            LSTM(self.model_config['lstm_units'][0], 
                 return_sequences=True, 
                 name='lstm_1'),
            Dropout(self.model_config['dropout_rates'][1]),
            
            LSTM(self.model_config['lstm_units'][1], 
                 return_sequences=False, 
                 name='lstm_2'),
            Dropout(self.model_config['dropout_rates'][1]),
            
            # 분류 부분
            Dense(self.model_config['dense_units'][0], 
                  activation='relu', 
                  name='dense_1'),
            BatchNormalization(),
            Dropout(self.model_config['dropout_rates'][2]),
            
            Dense(1, activation='sigmoid', name='output')
        ])
        
        self.model = model
        self._compile_model()
        
        self.logger.info(f"✅ CNN+LSTM 하이브리드 모델 구축 완료: {model.count_params():,} 파라미터")
        return model
    
    def build_attention_model(self):
        """Attention 메커니즘 포함 고급 모델"""
        self.logger.info("🧠 Attention 모델 구축...")
        
        # 함수형 API 사용
        inputs = Input(shape=self.input_shape, name='input')
        
        # CNN 레이어
        x = Conv1D(64, kernel_size=5, activation='relu', name='conv1d_1')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.25)(x)
        
        x = Conv1D(32, kernel_size=3, activation='relu', name='conv1d_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        
        # LSTM 레이어
        lstm_out = LSTM(64, return_sequences=True, name='lstm_1')(x)
        lstm_out = Dropout(0.4)(lstm_out)
        
        # Multi-Head Attention
        attention_out = MultiHeadAttention(
            num_heads=4, 
            key_dim=16, 
            name='multi_head_attention'
        )(lstm_out, lstm_out)
        
        # Residual connection + Layer Normalization
        attention_out = LayerNormalization()(attention_out + lstm_out)
        
        # Global Average Pooling
        pooled = GlobalAveragePooling1D()(attention_out)
        
        # 분류 레이어
        dense = Dense(16, activation='relu', name='dense_1')(pooled)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        outputs = Dense(1, activation='sigmoid', name='output')(dense)
        
        model = Model(inputs=inputs, outputs=outputs, name='CSI_Attention_Model')
        
        self.model = model
        self._compile_model()
        
        self.logger.info(f"✅ Attention 모델 구축 완료: {model.count_params():,} 파라미터")
        return model
    
    def build_multi_scale_model(self):
        """다중 스케일 특성 추출 모델"""
        self.logger.info("🔬 다중 스케일 모델 구축...")
        
        inputs = Input(shape=self.input_shape, name='input')
        
        # 다중 스케일 CNN 브랜치
        # 브랜치 1: 짧은 패턴 (kernel_size=3)
        conv1_1 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_1 = MaxPooling1D(pool_size=2)(conv1_1)
        
        # 브랜치 2: 중간 패턴 (kernel_size=5)
        conv1_2 = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
        conv1_2 = BatchNormalization()(conv1_2)
        conv1_2 = MaxPooling1D(pool_size=2)(conv1_2)
        
        # 브랜치 3: 긴 패턴 (kernel_size=7)
        conv1_3 = Conv1D(32, kernel_size=7, activation='relu', padding='same')(inputs)
        conv1_3 = BatchNormalization()(conv1_3)
        conv1_3 = MaxPooling1D(pool_size=2)(conv1_3)
        
        # 브랜치 결합
        merged = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        merged = Dropout(0.25)(merged)
        
        # 추가 CNN 레이어
        x = Conv1D(64, kernel_size=3, activation='relu')(merged)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        
        # LSTM 레이어
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.4)(x)
        
        # 분류 레이어
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CSI_MultiScale_Model')
        
        self.model = model
        self._compile_model()
        
        self.logger.info(f"✅ 다중 스케일 모델 구축 완료: {model.count_params():,} 파라미터")
        return model
    
    def build_lightweight_model(self):
        """경량화 모델 (실시간 처리용)"""
        self.logger.info("⚡ 경량화 모델 구축...")
        
        model = Sequential([
            # 경량 CNN
            Conv1D(16, kernel_size=5, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=4),  # 더 큰 풀링
            Dropout(0.2),
            
            Conv1D(8, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # 경량 LSTM
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            
            # 간단한 분류기
            Dense(8, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        self.model = model
        self._compile_model()
        
        self.logger.info(f"✅ 경량화 모델 구축 완료: {model.count_params():,} 파라미터")
        return model
    
    def _compile_model(self):
        """모델 컴파일"""
        optimizer = Adam(learning_rate=self.model_config['learning_rate'])
        
        try:
            # TensorFlow 2.x 호환 메트릭
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
        except Exception as e:
            self.logger.warning(f"고급 메트릭 설정 실패, 기본 설정 사용: {e}")
            # 백업: 기본 메트릭만 사용
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
    
    def get_model_summary(self):
        """모델 요약 정보"""
        if self.model is None:
            return "모델이 구축되지 않았습니다."
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def visualize_model(self, filename='model_architecture.png'):
        """모델 아키텍처 시각화"""
        if self.model is None:
            self.logger.error("모델이 구축되지 않았습니다!")
            return None
        
        try:
            tf.keras.utils.plot_model(
                self.model, 
                to_file=filename,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=150
            )
            self.logger.info(f"📊 모델 아키텍처 저장: {filename}")
            return filename
        except Exception as e:
            self.logger.warning(f"모델 시각화 실패: {e}")
            return None
    
    def create_callbacks(self, model_save_path, monitor='val_loss'):
        """훈련 콜백 생성"""
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=CSIConfig.PATIENCE,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=CSIConfig.PATIENCE // 2,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1,
                mode='min'
            )
        ]
        
        return callbacks
    
    def build_model(self, model_type='cnn_lstm_hybrid'):
        """지정된 타입의 모델 구축"""
        model_builders = {
            'basic_lstm': self.build_basic_lstm,
            'cnn_lstm_hybrid': self.build_cnn_lstm_hybrid,
            'attention': self.build_attention_model,
            'multi_scale': self.build_multi_scale_model,
            'lightweight': self.build_lightweight_model
        }
        
        if model_type not in model_builders:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        self.logger.info(f"🏗️ 모델 구축 시작: {model_type}")
        model = model_builders[model_type]()
        
        # 모델 요약 출력
        self.logger.info(f"📋 모델 요약:\n{self.get_model_summary()}")
        
        return model
    
    def compare_models(self):
        """다양한 모델들의 파라미터 수 비교"""
        models_info = {}
        
        for model_type in ['basic_lstm', 'cnn_lstm_hybrid', 'attention', 'multi_scale', 'lightweight']:
            try:
                # 임시로 모델 구축
                temp_builder = CSIModelBuilder(self.input_shape)
                temp_model = temp_builder.build_model(model_type)
                
                models_info[model_type] = {
                    'params': temp_model.count_params(),
                    'trainable_params': temp_model.count_params(),
                    'layers': len(temp_model.layers)
                }
                
                # 메모리 정리
                del temp_model, temp_builder
                
            except Exception as e:
                models_info[model_type] = {'error': str(e)}
        
        return models_info
    
    def print_model_comparison(self):
        """모델 비교 정보 출력"""
        print("🔍 모델 아키텍처 비교")
        print("=" * 60)
        
        models_info = self.compare_models()
        
        for model_type, info in models_info.items():
            print(f"\n📦 {model_type.upper().replace('_', ' ')}:")
            if 'error' in info:
                print(f"   ❌ 구축 실패: {info['error']}")
            else:
                print(f"   파라미터 수: {info['params']:,}개")
                print(f"   레이어 수: {info['layers']}개")
                
                # 복잡도 평가
                if info['params'] < 50000:
                    complexity = "가벼움 ⚡"
                elif info['params'] < 200000:
                    complexity = "보통 ⚖️"
                else:
                    complexity = "무거움 🏋️"
                
                print(f"   복잡도: {complexity}")

# 모델 추천 함수
def recommend_model(use_case='general'):
    """사용 목적에 따른 모델 추천"""
    recommendations = {
        'general': 'cnn_lstm_hybrid',
        'high_accuracy': 'attention',
        'real_time': 'lightweight',
        'research': 'multi_scale',
        'simple': 'basic_lstm'
    }
    
    return recommendations.get(use_case, 'cnn_lstm_hybrid')

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 CSI 모델 빌더 테스트")
    print("=" * 40)
    
    # 모델 빌더 생성
    builder = CSIModelBuilder()
    
    # 모델 비교
    builder.print_model_comparison()
    
    # 추천 모델 구축
    print(f"\n🎯 추천 모델 구축...")
    recommended_type = recommend_model('general')
    model = builder.build_model(recommended_type)
    
    print(f"✅ {recommended_type} 모델 구축 완료!")
    print(f"   파라미터: {model.count_params():,}개")
    
    # 아키텍처 시각화 (선택적)
    try:
        builder.visualize_model('test_model.png')
    except:
        print("   ⚠️ 모델 시각화 스킵 (graphviz 미설치)")