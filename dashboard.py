import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import socket
import cv2
import numpy as np
import threading
import queue
import time
from database import Database

class VideoStream:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.thread = None
        self.last_frame = None
        self.frame_counter = 0
        self.last_frame_time = 0
        
    def start_stream(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._receive_frames, daemon=True)
        self.thread.start()
        
    def stop_stream(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def _receive_frames(self):
        sock = None
        reconnect_delay = 1
        
        while self.is_running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((self.host, self.port))
                st.success(f"✅ Подключено к видео-серверу {self.host}:{self.port}")
                reconnect_delay = 1
                
                while self.is_running:
                    try:
                        size_data = b''
                        while len(size_data) < 4:
                            chunk = sock.recv(4 - len(size_data))
                            if not chunk:
                                break
                            size_data += chunk
                        
                        if len(size_data) != 4:
                            break
                            
                        frame_size = int.from_bytes(size_data, byteorder='big')
                        
                        frame_data = b''
                        while len(frame_data) < frame_size:
                            chunk = sock.recv(min(4096, frame_size - len(frame_data)))
                            if not chunk:
                                break
                            frame_data += chunk
                        
                        if len(frame_data) != frame_size:
                            break
                        
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            self.last_frame = frame_rgb
                            self.frame_counter += 1
                            self.last_frame_time = time.time()
                            
                            if not self.frame_queue.full():
                                self.frame_queue.put(frame_rgb)
                                
                    except socket.timeout:
                        continue
                    except Exception as e:
                        st.error(f"Ошибка получения кадра: {e}")
                        break
                        
            except Exception as e:
                error_msg = f"Ошибка подключения к видео-серверу: {e}"
                if sock:
                    sock.close()
                
                if self.is_running:
                    st.warning(f"🔄 Переподключение через {reconnect_delay} сек...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)
            finally:
                if sock:
                    sock.close()
    
    def get_frame(self):
        try:
            new_frame = self.frame_queue.get_nowait()
            self.last_frame = new_frame
            return new_frame
        except queue.Empty:
            pass
        
        if self.last_frame is not None:
            if time.time() - self.last_frame_time < 30:
                return self.last_frame
            else:
                st.warning("📹 Видеопоток прервался. Ожидание новых кадров...")
        
        return None


def main():
    st.set_page_config(page_title="Factory Analytics", layout="wide")
    st.title("🏭 Система анализа видео на предприятии")

    if 'video_stream' not in st.session_state:
        st.session_state.video_stream = VideoStream()
        
    db = Database()

    st.sidebar.header("Настройки")
    hours = st.sidebar.slider("Период анализа (часы):", 1, 168, 24)
    
    st.sidebar.header("Настройки видео")
    video_host = st.sidebar.text_input("Хост видео-сервера:", "localhost")
    video_port = st.sidebar.number_input("Порт видео-сервера:", min_value=1000, max_value=65535, value=5000)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Запустить видео"):
            st.session_state.video_stream.host = video_host
            st.session_state.video_stream.port = video_port
            st.session_state.video_stream.start_stream()
            st.success("Видеопоток запущен")
            
    with col2:
        if st.button("⏹️ Остановить видео"):
            st.session_state.video_stream.stop_stream()
            st.session_state.video_stream.last_frame = None  # Очищаем последний кадр
            st.info("Видеопоток остановлен")

    # Кнопка очистки БД в сайдбаре
    st.sidebar.header("Управление базой данных")
    if st.sidebar.button("🗑️ Очистить базу данных", type="secondary"):
        if st.sidebar.checkbox("Подтвердить очистку всех данных"):
            if db.clear_database():
                st.rerun()

    action_stats, train_stats, hourly_activity = db.get_stats(hours)

    # Верхние метрики
    col1, col2, col3, col4 = st.columns(4)

    total_events = sum(action_stats.values())
    with col1:
        st.metric("Всего событий", total_events)

    with col2:
        st.metric("Уникальных действий", len(action_stats))

    with col3:
        standing_count = action_stats.get('standing', 0)
        st.metric("Стоячих поз", standing_count)

    with col4:
        bending_count = action_stats.get('bending', 0)
        st.metric("Наклонов", bending_count)

    # Метрики по поездам
    st.subheader("🚆 Статистика объектов")
    col1, col2, col3, col4 = st.columns(4)
    
    total_trains = sum(train_stats.values()) if train_stats else 0
    unique_trains = len(train_stats) if train_stats else 0
    
    with col1:
        st.metric("Всего объектов", total_trains)
    
    with col2:
        st.metric("Уникальных номеров", unique_trains)
    
    with col3:
        last_train = list(train_stats.keys())[-1] if train_stats else "Нет данных"
        st.metric("Последний объект", last_train)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Прямая трансляция с камеры")
        
        video_placeholder = st.empty()
        
        frame = st.session_state.video_stream.get_frame()
        
        if frame is not None:
            frame_with_info = frame.copy()
            
            video_placeholder.image(frame_with_info, width=640, channels="RGB")
        else:
            video_placeholder.info("📹 Ожидание видеопотока...")
            
        status_placeholder = st.empty()
        if st.session_state.video_stream.is_running:
            if frame is not None:
                status_placeholder.success(f"✅ Видеопоток активен | Кадров: {st.session_state.video_stream.frame_counter}")
            else:
                status_placeholder.warning("🔄 Подключение к видео-серверу...")
        else:
            status_placeholder.warning("⏸️ Видеопоток не активен")

    with col2:
        # Две диаграммы в колонке
        tab1, tab2 = st.tabs(["📊 Действия сотрудников", "🚆 Статистика объектов"])
        
        with tab1:
            if action_stats:
                fig_pie = px.pie(
                    values=list(action_stats.values()),
                    names=list(action_stats.keys()),
                    title="Распределение действий сотрудников"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Нет данных для отображения диаграммы действий")
        
        with tab2:
            if train_stats:
                # Диаграмма для поездов
                fig_train_bar = px.bar(
                    x=list(train_stats.keys()),
                    y=list(train_stats.values()),
                    title="Количество прибытий поездов по номерам",
                    labels={'x': 'Номер поезда', 'y': 'Количество прибытий'}
                )
                fig_train_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_train_bar, use_container_width=True)
            else:
                st.info("Нет данных для отображения статистики поездов")

    st.subheader("Активность по времени")
    if hourly_activity:
        hours, counts = zip(*hourly_activity)
        df_hourly = pd.DataFrame({
            'hour': [h.strftime('%H:%M') for h in hours],
            'count': counts
        })

        fig_bar = px.bar(
            df_hourly,
            x='hour',
            y='count',
            title="Активность по времени"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Нет данных для отображения графика активности")

    # Детальная статистика в табах
    tab1, tab2 = st.tabs(["👥 Статистика действий", "🚆 Статистика объектов"])
    
    with tab1:
        st.subheader("Детальная статистика действий")
        if action_stats:
            df_stats = pd.DataFrame({
                'Действие': list(action_stats.keys()),
                'Количество': list(action_stats.values())
            })
            st.dataframe(df_stats, use_container_width=True)
        else:
            st.info("Нет данных для отображения статистики действий")
    
    with tab2:
        st.subheader("Детальная статистика объектов")
        if train_stats:
            df_train_stats = pd.DataFrame({
                'Номер поезда': list(train_stats.keys()),
                'Количество прибытий': list(train_stats.values())
            })
            st.dataframe(df_train_stats, use_container_width=True)
        else:
            st.info("Нет данных для отображения статистики объектов")

    st.sidebar.header("Настройки обновления")
    refresh_rate = st.sidebar.slider("Частота обновления видео (FPS):", 1, 30, 10)
    
    if st.session_state.video_stream.is_running:
        time.sleep(1.0 / refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()