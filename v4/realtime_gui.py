"""
CSI 낙상 감지 v4 - 실시간 감지 GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import threading
import time
from datetime import datetime

from realtime_detector import RealTimeCSIDetector, CSIDataSimulator
from evaluator import list_available_models


class RealTimeDetectionGUI:
    """실시간 낙상 감지 GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚨 CSI 실시간 낙상 감지 시스템")
        self.root.geometry("1200x800")
        
        # 감지기 및 시뮬레이터
        self.detector = None
        self.simulator = None
        self.is_running = False
        
        # 데이터 저장용
        self.prediction_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
        # GUI 구성
        self.setup_gui()
        self.setup_plots()
        
        # 애니메이션
        self.animation = None
        
    def setup_gui(self):
        """GUI 구성"""
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 컨트롤 패널
        control_frame = ttk.LabelFrame(main_frame, text="🎛️ 제어 패널", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 모델 선택
        ttk.Label(control_frame, text="모델:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                       width=40, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=(5, 10))
        
        # 임계값 설정
        ttk.Label(control_frame, text="임계값:").grid(row=0, column=2, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, 
                                        variable=self.threshold_var, orient=tk.HORIZONTAL,
                                        length=150, command=self.on_threshold_changed)
        self.threshold_scale.grid(row=0, column=3, padx=(5, 10))
        
        self.threshold_label = ttk.Label(control_frame, text="0.5")
        self.threshold_label.grid(row=0, column=4)
        
        # 제어 버튼들
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="🟢 시작", 
                                      command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="🔴 중지", 
                                     command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.reset_button = ttk.Button(button_frame, text="🔄 리셋", 
                                      command=self.reset_system)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 상태 패널
        status_frame = ttk.LabelFrame(main_frame, text="📊 상태 정보", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 상태 라벨들
        self.status_labels = {}
        status_items = [
            ("상태", "대기 중"),
            ("마지막 예측", "0.000"),
            ("낙상 감지", "0회"),
            ("총 예측", "0회"),
            ("버퍼 크기", "0/32")
        ]
        
        for i, (key, default_value) in enumerate(status_items):
            ttk.Label(status_frame, text=f"{key}:").grid(row=i, column=0, sticky=tk.W)
            label = ttk.Label(status_frame, text=default_value, font=("Arial", 10, "bold"))
            label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0))
            self.status_labels[key] = label
        
        # 로그 패널
        log_frame = ttk.LabelFrame(main_frame, text="📝 로그", padding="10")
        log_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 로그 텍스트
        self.log_text = tk.Text(log_frame, height=15, width=50)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 그래프 프레임
        self.plot_frame = ttk.LabelFrame(main_frame, text="📈 실시간 그래프", padding="10")
        self.plot_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 사용 가능한 모델 로드
        self.load_available_models()
        
        # 그리드 가중치 설정
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=2)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def setup_plots(self):
        """그래프 설정"""
        # Matplotlib 그래프
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 예측 확률 그래프
        self.ax1.set_title("실시간 낙상 예측 확률")
        self.ax1.set_ylabel("확률")
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True)
        
        # 임계값 라인
        self.threshold_line = self.ax1.axhline(y=0.5, color='r', linestyle='--', 
                                              label='임계값')
        self.ax1.legend()
        
        # 감지 상태 그래프
        self.ax2.set_title("낙상 감지 상태")
        self.ax2.set_ylabel("감지")
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlabel("시간 (초)")
        self.ax2.grid(True)
        
        # 캔버스 생성
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_available_models(self):
        """사용 가능한 모델 로드"""
        try:
            models = list_available_models()
            self.model_combo['values'] = models
            if models:
                self.model_combo.current(len(models) - 1)  # 최신 모델 선택
        except Exception as e:
            self.log_message(f"❌ 모델 목록 로드 실패: {e}")
    
    def on_threshold_changed(self, value):
        """임계값 변경 처리"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        
        # 그래프 업데이트
        if hasattr(self, 'threshold_line'):
            self.threshold_line.set_ydata([threshold, threshold])
            self.canvas.draw_idle()
        
        # 감지기에 적용
        if self.detector:
            self.detector.set_threshold(threshold)
    
    def start_detection(self):
        """감지 시작"""
        try:
            model_name = self.model_var.get()
            if not model_name:
                messagebox.showerror("오류", "모델을 선택하세요.")
                return
            
            threshold = self.threshold_var.get()
            
            # 감지기 초기화
            self.detector = RealTimeCSIDetector(model_name, threshold=threshold)
            self.detector.on_fall_detected = self.on_fall_detected
            self.detector.on_prediction_updated = self.on_prediction_updated
            
            # 모델 로드
            self.log_message("📂 모델 로딩 중...")
            self.detector.load_model()
            
            # 시뮬레이터 시작
            self.simulator = CSIDataSimulator()
            
            # 감지 시작
            self.detector.start_detection()
            
            # 데이터 스트리밍 시작
            self.is_running = True
            self.stream_thread = threading.Thread(target=self.data_streaming)
            self.stream_thread.start()
            
            # 애니메이션 시작
            self.animation = FuncAnimation(self.fig, self.update_plots, 
                                         interval=100, blit=False)
            
            # 버튼 상태 변경
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            self.log_message("🟢 실시간 감지 시작!")
            
        except Exception as e:
            messagebox.showerror("오류", f"감지 시작 실패: {e}")
            self.log_message(f"❌ 감지 시작 실패: {e}")
    
    def stop_detection(self):
        """감지 중지"""
        try:
            self.is_running = False
            
            if self.detector:
                self.detector.stop_detection()
            
            if self.animation:
                self.animation.event_source.stop()
            
            # 버튼 상태 변경
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            self.log_message("🔴 실시간 감지 중지")
            
        except Exception as e:
            self.log_message(f"❌ 감지 중지 실패: {e}")
    
    def reset_system(self):
        """시스템 리셋"""
        if self.is_running:
            self.stop_detection()
        
        # 데이터 초기화
        self.prediction_history.clear()
        self.time_history.clear()
        
        # 상태 라벨 초기화
        self.status_labels["상태"].config(text="대기 중")
        self.status_labels["마지막 예측"].config(text="0.000")
        self.status_labels["낙상 감지"].config(text="0회")
        self.status_labels["총 예측"].config(text="0회")
        self.status_labels["버퍼 크기"].config(text="0/32")
        
        # 그래프 초기화
        self.ax1.clear()
        self.ax2.clear()
        self.setup_plots()
        
        self.log_message("🔄 시스템 리셋 완료")
    
    def data_streaming(self):
        """데이터 스트리밍 (별도 스레드)"""
        while self.is_running:
            try:
                sample = self.simulator.get_next_sample()
                if sample is not None:
                    self.detector.add_data_point(sample)
                
                time.sleep(0.1)  # 100ms 간격
                
            except Exception as e:
                self.log_message(f"❌ 데이터 스트리밍 오류: {e}")
                break
    
    def on_fall_detected(self, prediction, timestamp):
        """낙상 감지 콜백"""
        message = f"🚨 낙상 감지! {timestamp.strftime('%H:%M:%S')} (확률: {prediction:.1%})"
        self.log_message(message)
        
        # 상태 업데이트
        if self.detector:
            status = self.detector.get_status()
            self.status_labels["낙상 감지"].config(text=f"{status['detection_count']}회")
    
    def on_prediction_updated(self, prediction, threshold):
        """예측 업데이트 콜백"""
        # 히스토리에 추가
        current_time = time.time()
        self.prediction_history.append(prediction)
        self.time_history.append(current_time)
        
        # 상태 업데이트
        if self.detector:
            status = self.detector.get_status()
            self.status_labels["상태"].config(text="실행 중" if status['is_running'] else "대기 중")
            self.status_labels["마지막 예측"].config(text=f"{prediction:.3f}")
            self.status_labels["총 예측"].config(text=f"{status['total_predictions']}회")
            self.status_labels["버퍼 크기"].config(text=f"{status['buffer_size']}/{self.detector.window_size}")
    
    def update_plots(self, frame):
        """그래프 업데이트"""
        if not self.prediction_history:
            return
        
        # 시간 축 계산 (최근 30초)
        current_time = time.time()
        time_window = 30  # 30초
        
        # 시간 범위 내 데이터만 선택
        recent_times = []
        recent_predictions = []
        recent_detections = []
        
        for i, t in enumerate(self.time_history):
            if current_time - t <= time_window:
                recent_times.append(t - current_time)  # 상대 시간
                recent_predictions.append(self.prediction_history[i])
                recent_detections.append(1 if self.prediction_history[i] > self.threshold_var.get() else 0)
        
        if not recent_times:
            return
        
        # 예측 확률 그래프 업데이트
        self.ax1.clear()
        self.ax1.plot(recent_times, recent_predictions, 'b-', linewidth=2, label='예측 확률')
        self.ax1.axhline(y=self.threshold_var.get(), color='r', linestyle='--', label='임계값')
        self.ax1.set_title("실시간 낙상 예측 확률")
        self.ax1.set_ylabel("확률")
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlim(-time_window, 0)
        self.ax1.grid(True)
        self.ax1.legend()
        
        # 감지 상태 그래프 업데이트
        self.ax2.clear()
        self.ax2.plot(recent_times, recent_detections, 'ro-', markersize=4, label='낙상 감지')
        self.ax2.set_title("낙상 감지 상태")
        self.ax2.set_ylabel("감지")
        self.ax2.set_xlabel("시간 (초 전)")
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_xlim(-time_window, 0)
        self.ax2.grid(True)
        self.ax2.legend()
        
        # 캔버스 업데이트
        self.canvas.draw_idle()
    
    def log_message(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    def run(self):
        """GUI 실행"""
        self.root.mainloop()


def main():
    """메인 함수"""
    print("🚀 CSI 실시간 낙상 감지 GUI 시작")
    
    try:
        app = RealTimeDetectionGUI()
        app.run()
    except Exception as e:
        print(f"❌ GUI 실행 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
