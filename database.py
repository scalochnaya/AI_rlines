import psycopg2
import json
from datetime import datetime


class Database:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="factory_db",
            user="admin",
            password="password",
            host="localhost",
            port="5432"
        )
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id SERIAL PRIMARY KEY,
                location TEXT NOT NULL,
                description TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                event_id SERIAL PRIMARY KEY,
                camera_id INTEGER,
                person_id INTEGER,
                timestamp TIMESTAMP,
                action VARCHAR(50),
                confidence FLOAT,
                bbox JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS train_events (
                event_id SERIAL PRIMARY KEY,
                camera_id INTEGER,
                train_id INTEGER,
                timestamp TIMESTAMP,
                train_num VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            INSERT INTO cameras (location, description) 
            VALUES ('1', 'Точка обслуживания поездов РЖД')
            ON CONFLICT DO NOTHING
        """)

        self.conn.commit()
        cursor.close()

    def save_person_event(self, person_id, action, confidence, bbox):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detection_events 
            (camera_id, person_id, timestamp, action, confidence, bbox)
            VALUES (1, %s, %s, %s, %s, %s)
        """, (person_id, datetime.now(), action, confidence, json.dumps(bbox)))
        self.conn.commit()
        cursor.close()

    def save_train_event(self, train_id, train_num):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO train_events 
            (camera_id, train_id, timestamp, train_num)
            VALUES (1, %s, %s, %s)
        """, (train_id, datetime.now(), train_num))
        self.conn.commit()
        cursor.close()

    def clear_database(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE * FROM detection_events")
        cursor.execute("DELETE * FROM train_events")
        self.conn.commit()
        cursor.close()

    def get_stats(self, hours=24):
        cursor = self.conn.cursor()

        # Статистика по действиям людей
        cursor.execute("""
            SELECT action, COUNT(*) as count 
            FROM detection_events 
            WHERE timestamp >= NOW() - INTERVAL '%s hours'
            GROUP BY action
        """, (hours,))
        action_stats = dict(cursor.fetchall())

        # Статистика по поездам - ИСПРАВЛЕННАЯ ЧАСТЬ
        cursor.execute("""
            SELECT train_num, COUNT(*) as count 
            FROM train_events 
            WHERE timestamp >= NOW() - INTERVAL '%s hours'
            GROUP BY train_num
        """, (hours,))
        train_stats = dict(cursor.fetchall())

        # Почасовая активность
        cursor.execute("""
            SELECT DATE_TRUNC('hour', timestamp) as hour, COUNT(*) as count
            FROM detection_events 
            WHERE timestamp >= NOW() - INTERVAL '%s hours'
            GROUP BY hour ORDER BY hour
        """, (hours,))
        hourly_activity = cursor.fetchall()

        cursor.close()
        return action_stats, train_stats, hourly_activity