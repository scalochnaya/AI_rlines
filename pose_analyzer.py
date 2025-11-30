import numpy as np


class PoseAnalyzer:
    def __init__(self):
        print("Анализ позиции...")

    def analyze_pose(self, keypoints):
        try:
            if keypoints is None or len(keypoints) == 0:
                return "Undefined", 0.0

            if len(keypoints.shape) == 1:
                keypoints = keypoints.reshape(-1, 3)

            required_points = [5, 6, 11, 12]  # плечи и бедра

            valid_points = True
            for i in required_points:
                if i >= len(keypoints) or keypoints[i, 2] <= 0.3:
                    valid_points = False
                    break

            if not valid_points:
                return "Undefined", 0.0

            left_shoulder = keypoints[5, :2]
            right_shoulder = keypoints[6, :2]
            left_hip = keypoints[11, :2]
            right_hip = keypoints[12, :2]

            # Вектор туловища
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            torso_vector = hip_center - shoulder_center

            # Угол наклона туловища
            angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))

            # Анализ позы
            if abs(angle) < 30:
                return "Standing", 0.8
            elif angle > 45:
                return "Bending", 0.7
            elif angle < -45:
                return "Sitting", 0.7
            else:
                return "Standing", 0.6

        except Exception as e:
            print(f"Ошибка в analyze_pose: {e}")
            return "Undefined", 0.0

    def simple_action_detection(self, keypoints, prev_keypoints):
        try:
            if keypoints is None:
                return "Undefined", 0.0

            if prev_keypoints is None:
                return self.analyze_pose(keypoints)

            if len(keypoints.shape) == 1:
                keypoints = keypoints.reshape(-1, 3)
            if len(prev_keypoints.shape) == 1:
                prev_keypoints = prev_keypoints.reshape(-1, 3)

            # Проверяем что есть достаточно точек для анализа
            if len(keypoints) < 13 or len(prev_keypoints) < 13:
                return self.analyze_pose(keypoints)

            # Простая логика определения движения
            current_center = np.mean(keypoints[:13, :2], axis=0)
            prev_center = np.mean(prev_keypoints[:13, :2], axis=0)

            movement = np.linalg.norm(current_center - prev_center)

            if movement > 10:
                return "Walking", 0.8
            else:
                return self.analyze_pose(keypoints)

        except Exception as e:
            print(f"Ошибка в simple_action_detection: {e}")
            return "Undefined", 0.0