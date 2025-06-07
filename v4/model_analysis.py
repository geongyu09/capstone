"""
CSI 낙상 감지 모델 상세 분석 및 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config, ModelConfig
from model_builder import create_model
import tensorflow as tf

def analyze_model_architecture():
    """모델 아키텍처 상세 분석"""
    
    print("🧠 CSI 낙상 감지 모델 아키텍처 분석")
    print("=" * 60)
    
    # 입력 형태
    input_shape = (Config.WINDOW_SIZE, Config.TOTAL_FEATURES)
    print(f"📊 입력 데이터 형태: {input_shape}")
    print(f"   - 시퀀스 길이: {Config.WINDOW_SIZE}개 시간 스텝")
    print(f"   - 특성 수: {Config.TOTAL_FEATURES}개 CSI amplitude 값")
    print(f"   - 데이터 크기: {Config.WINDOW_SIZE * Config.TOTAL_FEATURES:,}개 값")
    
    # 세 가지 모델 비교
    model_types = ['simple', 'cnn', 'hybrid']
    model_info = {}
    
    for model_type in model_types:
        print(f"\n🔍 {model_type.upper()} 모델 분석:")
        print("-" * 40)
        
        try:
            model, builder = create_model(model_type=model_type, input_shape=input_shape)
            
            # 모델 정보 수집
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            model_info[model_type] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'layers': len(model.layers),
                'model': model
            }
            
            print(f"📈 모델 정보:")
            print(f"   - 총 레이어 수: {len(model.layers)}")
            print(f"   - 총 파라미터: {total_params:,}")
            print(f"   - 학습 가능 파라미터: {trainable_params:,}")
            print(f"   - 메모리 사용량 (추정): {total_params * 4 / 1024 / 1024:.1f} MB")
            
            # 레이어별 상세 정보
            print(f"📋 주요 레이어:")
            for i, layer in enumerate(model.layers[:10]):  # 처음 10개 레이어만
                if hasattr(layer, 'output_shape'):
                    print(f"   {i+1:2d}. {layer.name:<20} {str(layer.output_shape):<20}")
            
            if len(model.layers) > 10:
                print(f"   ... 외 {len(model.layers)-10}개 레이어")
            
        except Exception as e:
            print(f"❌ {model_type} 모델 분석 실패: {e}")
    
    # 모델 비교 차트
    print(f"\n📊 모델 비교:")
    print("-" * 40)
    
    comparison_data = []
    for model_type, info in model_info.items():
        comparison_data.append([
            model_type.upper(),
            f"{info['total_params']:,}",
            f"{info['layers']}",
            f"{info['total_params'] * 4 / 1024 / 1024:.1f} MB"
        ])
    
    print(f"{'모델':<10} {'파라미터':<12} {'레이어':<8} {'메모리':<10}")
    print("-" * 42)
    for data in comparison_data:
        print(f"{data[0]:<10} {data[1]:<12} {data[2]:<8} {data[3]:<10}")
    
    return model_info

def explain_cnn_component():
    """CNN 컴포넌트 설명"""
    
    print(f"\n🔍 CNN (Convolutional Neural Network) 컴포넌트")
    print("=" * 50)
    
    print(f"📋 역할:")
    print(f"   - CSI 데이터의 공간적 패턴 추출")
    print(f"   - 낙상 시 특징적인 amplitude 변화 감지")
    print(f"   - 노이즈 필터링 및 중요 특성 강조")
    
    print(f"\n🏗️ 구조:")
    for i, filters in enumerate(ModelConfig.CNN_FILTERS):
        print(f"   Layer {i+1}: Conv1D({filters} filters) → BatchNorm → MaxPool → Dropout")
    
    print(f"\n⚙️ 설정:")
    print(f"   - 필터 수: {ModelConfig.CNN_FILTERS}")
    print(f"   - 커널 크기: {ModelConfig.CNN_KERNEL_SIZE}")
    print(f"   - 드롭아웃: {ModelConfig.CNN_DROPOUT}")
    
    print(f"\n💡 왜 CNN을 사용하나요?")
    print(f"   - CSI amplitude는 시간에 따라 연속적으로 변화")
    print(f"   - 낙상 시 특정 패턴이 나타남 (급격한 변화, 특정 주파수)")
    print(f"   - CNN이 이런 국소적 패턴을 효과적으로 감지")

def explain_lstm_component():
    """LSTM 컴포넌트 설명"""
    
    print(f"\n🔄 LSTM (Long Short-Term Memory) 컴포넌트")
    print("=" * 50)
    
    print(f"📋 역할:")
    print(f"   - 시간 순서에 따른 패턴 학습")
    print(f"   - 장기 의존성 포착 (낙상 전후 상황 이해)")
    print(f"   - 순차적 행동 패턴 분석")
    
    print(f"\n🏗️ 구조:")
    for i, units in enumerate(ModelConfig.LSTM_UNITS):
        direction = "Bidirectional" 
        print(f"   Layer {i+1}: {direction} LSTM({units} units)")
    
    print(f"\n⚙️ 설정:")
    print(f"   - 유닛 수: {ModelConfig.LSTM_UNITS}")
    print(f"   - 드롭아웃: {ModelConfig.LSTM_DROPOUT}")
    print(f"   - 순환 드롭아웃: {ModelConfig.LSTM_RECURRENT_DROPOUT}")
    
    print(f"\n💡 왜 LSTM을 사용하나요?")
    print(f"   - 낙상은 시간의 흐름에 따른 연속적 과정")
    print(f"   - 정상 상태 → 불안정 → 낙상 → 충격의 순서")
    print(f"   - LSTM이 이런 시계열 패턴을 기억하고 학습")

def explain_hybrid_approach():
    """하이브리드 접근법 설명"""
    
    print(f"\n🔗 하이브리드 (CNN + LSTM) 접근법")
    print("=" * 50)
    
    print(f"🎯 결합 전략:")
    print(f"   1. CNN: 공간적 특성 추출 → GlobalAveragePooling")
    print(f"   2. LSTM: 시간적 특성 추출 → 최종 은닉 상태")
    print(f"   3. Concatenate: 두 특성을 결합")
    print(f"   4. Dense: 최종 분류 결정")
    
    print(f"\n🔄 데이터 흐름:")
    print(f"   입력 (50×245)")
    print(f"   ├─ CNN 브랜치 → (64차원 특성)")
    print(f"   ├─ LSTM 브랜치 → (128차원 특성)")
    print(f"   └─ 결합 → (192차원) → Dense → 출력 (1차원)")
    
    print(f"\n✨ 장점:")
    print(f"   - CNN: 순간적 변화 패턴 감지 (낙상 순간)")
    print(f"   - LSTM: 연속적 행동 흐름 이해 (낙상 과정)")
    print(f"   - 두 정보를 종합하여 더 정확한 판단")

def visualize_data_flow():
    """데이터 플로우 시각화"""
    
    print(f"\n📊 데이터 처리 과정 시각화")
    print("=" * 50)
    
    # 가상의 CSI 데이터 생성
    time_steps = 50
    features = 10  # 시각화용으로 축소
    
    # 정상 상태 (안정적)
    normal_data = np.random.normal(0, 0.5, (time_steps, features))
    
    # 낙상 상태 (중간에 급격한 변화)
    fall_data = np.random.normal(0, 0.5, (time_steps, features))
    fall_start = 25
    fall_end = 35
    fall_data[fall_start:fall_end] += np.random.normal(2, 1, (fall_end - fall_start, features))
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 정상 상태 CSI 패턴
    axes[0, 0].imshow(normal_data.T, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('정상 상태 CSI 패턴')
    axes[0, 0].set_xlabel('시간 (Time Steps)')
    axes[0, 0].set_ylabel('CSI 특성')
    
    # 낙상 상태 CSI 패턴
    axes[0, 1].imshow(fall_data.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('낙상 상태 CSI 패턴')
    axes[0, 1].set_xlabel('시간 (Time Steps)')
    axes[0, 1].set_ylabel('CSI 특성')
    axes[0, 1].axvline(x=fall_start, color='red', linestyle='--', label='낙상 시작')
    axes[0, 1].axvline(x=fall_end, color='red', linestyle='--', label='낙상 종료')
    axes[0, 1].legend()
    
    # 시간에 따른 변화 (한 특성만)
    axes[1, 0].plot(normal_data[:, 0], label='정상', color='blue')
    axes[1, 0].plot(fall_data[:, 0], label='낙상', color='red')
    axes[1, 0].set_title('시간에 따른 CSI 변화 (특성 1)')
    axes[1, 0].set_xlabel('시간')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 모델 아키텍처 개념도
    axes[1, 1].text(0.1, 0.9, '🧠 하이브리드 모델', fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.8, '📊 입력: CSI 데이터 (50×245)', fontsize=10)
    axes[1, 1].text(0.1, 0.7, '├─ 🔍 CNN: 공간적 패턴', fontsize=10)
    axes[1, 1].text(0.1, 0.6, '├─ 🔄 LSTM: 시간적 패턴', fontsize=10)
    axes[1, 1].text(0.1, 0.5, '└─ 🔗 결합 → 분류', fontsize=10)
    axes[1, 1].text(0.1, 0.3, '📈 출력: 낙상 확률 (0~1)', fontsize=10)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{Config.LOG_DIR}/model_architecture_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 시각화 저장됨: {Config.LOG_DIR}/model_architecture_explanation.png")

def explain_training_process():
    """학습 과정 설명"""
    
    print(f"\n🏋️ 모델 학습 과정")
    print("=" * 50)
    
    print(f"📋 1단계: 데이터 준비")
    print(f"   - 전처리된 CSI 파일들을 시퀀스로 변환")
    print(f"   - 윈도우 크기: {Config.WINDOW_SIZE}")
    print(f"   - 스트라이드: {Config.STRIDE}")
    print(f"   - 훈련/검증/테스트: {Config.TRAIN_RATIO}/{Config.VAL_RATIO}/{Config.TEST_RATIO}")
    
    print(f"\n📋 2단계: 모델 설정")
    print(f"   - 배치 크기: {Config.BATCH_SIZE}")
    print(f"   - 학습률: {Config.LEARNING_RATE}")
    print(f"   - 손실 함수: Binary Crossentropy (클래스 가중치 적용)")
    print(f"   - 옵티마이저: Adam")
    
    print(f"\n📋 3단계: 학습 실행")
    print(f"   - 최대 에포크: {Config.EPOCHS}")
    print(f"   - 조기 종료: validation loss 기준")
    print(f"   - 학습률 감소: plateau 시 0.5배")
    print(f"   - 최고 성능 모델 자동 저장")
    
    print(f"\n📋 4단계: 성능 평가")
    print(f"   - 테스트 데이터로 최종 평가")
    print(f"   - 다양한 임계값으로 성능 측정")
    print(f"   - 혼동 행렬, ROC 커브 생성")

def main():
    """메인 실행 함수"""
    
    # 모델 아키텍처 분석
    model_info = analyze_model_architecture()
    
    # 각 컴포넌트 상세 설명
    explain_cnn_component()
    explain_lstm_component()
    explain_hybrid_approach()
    
    # 학습 과정 설명
    explain_training_process()
    
    # 시각화
    print(f"\n📊 데이터 플로우 시각화를 생성하시겠습니까?")
    choice = input("y/n: ").lower().strip()
    
    if choice == 'y':
        try:
            visualize_data_flow()
            print(f"✅ 시각화 완료!")
        except Exception as e:
            print(f"❌ 시각화 실패: {e}")
    
    print(f"\n🎯 요약:")
    print(f"   - 하이브리드 모델이 가장 복잡하지만 성능이 우수")
    print(f"   - Simple 모델은 빠른 테스트용")
    print(f"   - CNN 모델은 공간적 패턴에 특화")
    print(f"   - 클래스 불균형 문제 해결을 위한 가중치 적용")
    print(f"   - 실시간 처리 가능한 경량화된 구조")

if __name__ == "__main__":
    main()
