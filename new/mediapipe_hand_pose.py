import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import math

class MediaPipeHandPose:
    """
    MediaPipe手部姿态识别模块
    功能：
    1. 手部关键点检测
    2. 手势识别（握拳、张开、指向等）
    3. 手部3D姿态估计
    4. 可视化手部关键点和姿态
    """
    
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        初始化MediaPipe手部姿态识别器
        
        Args:
            static_image_mode: 是否使用静态图像模式
            max_num_hands: 最大检测手部数量
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        # 初始化MediaPipe手部解决方案
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 创建手部检测器
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 手部关键点索引（MediaPipe标准）
        self.LANDMARK_INDICES = {
            'WRIST': 0,
            'THUMB_CMC': 1, 'THUMB_MCP': 2, 'THUMB_IP': 3, 'THUMB_TIP': 4,
            'INDEX_FINGER_MCP': 5, 'INDEX_FINGER_PIP': 6, 'INDEX_FINGER_DIP': 7, 'INDEX_FINGER_TIP': 8,
            'MIDDLE_FINGER_MCP': 9, 'MIDDLE_FINGER_PIP': 10, 'MIDDLE_FINGER_DIP': 11, 'MIDDLE_FINGER_TIP': 12,
            'RING_FINGER_MCP': 13, 'RING_FINGER_PIP': 14, 'RING_FINGER_DIP': 15, 'RING_FINGER_TIP': 16,
            'PINKY_MCP': 17, 'PINKY_PIP': 18, 'PINKY_DIP': 19, 'PINKY_TIP': 20
        }
        
        # 手势识别阈值
        self.GESTURE_THRESHOLDS = {
            'fist_threshold': 0.8,      # 握拳阈值
            'open_threshold': 0.3,      # 张开阈值
            'point_threshold': 0.7,     # 指向阈值
            'peace_threshold': 0.6,     # 胜利手势阈值
            'ok_threshold': 0.5         # OK手势阈值
        }
        
        # 手部状态历史（用于平滑检测）
        self.hand_states_history = []
        self.max_history = 5
        
    def detect_hands(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的手部关键点
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            List[Dict]: 检测到的手部信息列表
        """
        # 转换BGR到RGB
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_image)
        
        hands_info = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 获取手部标签（左手/右手）
                hand_label = results.multi_handedness[idx].classification[0].label
                hand_confidence = results.multi_handedness[idx].classification[0].score
                
                # 提取关键点坐标
                landmarks = self._extract_landmarks(hand_landmarks, image.shape)
                
                # 识别手势
                gesture = self._recognize_gesture(landmarks)
                
                # 计算手部3D姿态
                pose_3d = self._calculate_hand_pose_3d(landmarks)
                
                # 计算手部中心点和边界框
                center, bbox = self._calculate_hand_center_and_bbox(landmarks, image.shape)
                
                hand_info = {
                    'hand_id': idx,
                    'label': hand_label,
                    'confidence': hand_confidence,
                    'landmarks': landmarks,
                    'gesture': gesture,
                    'pose_3d': pose_3d,
                    'center': center,
                    'bbox': bbox,
                    'landmarks_raw': hand_landmarks
                }
                
                hands_info.append(hand_info)
        
        # 更新手部状态历史
        self._update_hand_states_history(hands_info)
        
        return hands_info
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """
        提取手部关键点坐标
        
        Args:
            hand_landmarks: MediaPipe手部关键点
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            List[Tuple[int, int]]: 关键点坐标列表
        """
        landmarks = []
        height, width = image_shape[:2]
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))
        
        return landmarks
    
    def _recognize_gesture(self, landmarks: List[Tuple[int, int]]) -> str:
        """
        识别手势
        
        Args:
            landmarks: 手部关键点坐标
            
        Returns:
            str: 识别的手势名称
        """
        if len(landmarks) < 21:
            return "Unknown"
        
        # 计算各手指的弯曲程度
        finger_states = self._calculate_finger_states(landmarks)
        
        # 手势识别逻辑
        if self._is_fist(finger_states):
            return "Fist"
        elif self._is_open_hand(finger_states):
            return "Open Hand"
        elif self._is_pointing(finger_states):
            return "Pointing"
        elif self._is_peace_sign(finger_states):
            return "Peace Sign"
        elif self._is_ok_sign(finger_states):
            return "OK Sign"
        elif self._is_thumbs_up(finger_states):
            return "Thumbs Up"
        else:
            return "Other"
    
    def _calculate_finger_states(self, landmarks: List[Tuple[int, int]]) -> Dict[str, bool]:
        """
        计算各手指的弯曲状态
        
        Args:
            landmarks: 手部关键点坐标
            
        Returns:
            Dict[str, bool]: 各手指的弯曲状态
        """
        finger_states = {}
        
        # 拇指
        thumb_tip = landmarks[self.LANDMARK_INDICES['THUMB_TIP']]
        thumb_ip = landmarks[self.LANDMARK_INDICES['THUMB_IP']]
        thumb_mcp = landmarks[self.LANDMARK_INDICES['THUMB_MCP']]
        thumb_cmc = landmarks[self.LANDMARK_INDICES['THUMB_CMC']]
        
        # 其他四指
        fingers = ['INDEX_FINGER', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY']
        finger_states['thumb'] = self._is_finger_bent(thumb_tip, thumb_ip, thumb_mcp, thumb_cmc)
        
        for finger in fingers:
            tip = landmarks[self.LANDMARK_INDICES[f'{finger}_TIP']]
            dip = landmarks[self.LANDMARK_INDICES[f'{finger}_DIP']]
            pip = landmarks[self.LANDMARK_INDICES[f'{finger}_PIP']]
            mcp = landmarks[self.LANDMARK_INDICES[f'{finger}_MCP']]
            
            finger_states[finger.lower()] = self._is_finger_bent(tip, dip, pip, mcp)
        
        return finger_states
    
    def _is_finger_bent(self, tip: Tuple[int, int], dip: Tuple[int, int], 
                       pip: Tuple[int, int], mcp: Tuple[int, int]) -> bool:
        """
        判断手指是否弯曲
        
        Args:
            tip, dip, pip, mcp: 手指各关节坐标
            
        Returns:
            bool: 是否弯曲
        """
        # 计算手指各关节的角度
        angle1 = self._calculate_angle(tip, dip, pip)
        angle2 = self._calculate_angle(dip, pip, mcp)
        
        # 如果角度小于阈值，认为手指弯曲
        return angle1 < 90 and angle2 < 90
    
    def _calculate_angle(self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
        """
        计算三点之间的角度
        
        Args:
            p1, p2, p3: 三个点的坐标
            
        Returns:
            float: 角度（度）
        """
        # 计算向量
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 计算角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def _is_fist(self, finger_states: Dict[str, bool]) -> bool:
        """判断是否为握拳"""
        return (finger_states.get('thumb', False) and 
                finger_states.get('index_finger', False) and
                finger_states.get('middle_finger', False) and
                finger_states.get('ring_finger', False) and
                finger_states.get('pinky', False))
    
    def _is_open_hand(self, finger_states: Dict[str, bool]) -> bool:
        """判断是否为张开手掌"""
        return (not finger_states.get('thumb', True) and 
                not finger_states.get('index_finger', True) and
                not finger_states.get('middle_finger', True) and
                not finger_states.get('ring_finger', True) and
                not finger_states.get('pinky', True))
    
    def _is_pointing(self, finger_states: Dict[str, bool]) -> bool:
        """判断是否为指向手势"""
        return (not finger_states.get('index_finger', True) and
                finger_states.get('middle_finger', False) and
                finger_states.get('ring_finger', False) and
                finger_states.get('pinky', False))
    
    def _is_peace_sign(self, finger_states: Dict[str, bool]) -> bool:
        """判断是否为胜利手势"""
        return (not finger_states.get('index_finger', True) and
                not finger_states.get('middle_finger', True) and
                finger_states.get('ring_finger', False) and
                finger_states.get('pinky', False))
    
    def _is_ok_sign(self, finger_states: Dict[str, bool]) -> bool:
        """判断是否为OK手势"""
        return (not finger_states.get('thumb', True) and
                not finger_states.get('index_finger', True) and
                finger_states.get('middle_finger', False) and
                finger_states.get('ring_finger', False) and
                finger_states.get('pinky', False))
    
    def _is_thumbs_up(self, finger_states: Dict[str, bool]) -> bool:
        """判断是否为竖拇指"""
        return (not finger_states.get('thumb', True) and
                finger_states.get('index_finger', False) and
                finger_states.get('middle_finger', False) and
                finger_states.get('ring_finger', False) and
                finger_states.get('pinky', False))
    
    def _calculate_hand_pose_3d(self, landmarks: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        计算手部3D姿态
        
        Args:
            landmarks: 手部关键点坐标
            
        Returns:
            Dict[str, float]: 3D姿态信息
        """
        if len(landmarks) < 21:
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
        
        # 使用手腕、中指MCP和食指MCP计算手部方向
        wrist = landmarks[self.LANDMARK_INDICES['WRIST']]
        middle_mcp = landmarks[self.LANDMARK_INDICES['MIDDLE_FINGER_MCP']]
        index_mcp = landmarks[self.LANDMARK_INDICES['INDEX_FINGER_MCP']]
        
        # 计算手部中心
        hand_center = (
            (wrist[0] + middle_mcp[0] + index_mcp[0]) // 3,
            (wrist[1] + middle_mcp[1] + index_mcp[1]) // 3
        )
        
        # 计算手部方向向量
        direction_vector = np.array([middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # 计算角度
        yaw = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi
        
        # 简化的pitch和roll计算
        pitch = 0  # 需要深度信息才能准确计算
        roll = 0   # 需要深度信息才能准确计算
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'hand_center': hand_center
        }
    
    def _calculate_hand_center_and_bbox(self, landmarks: List[Tuple[int, int]], 
                                      image_shape: Tuple[int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """
        计算手部中心点和边界框
        
        Args:
            landmarks: 手部关键点坐标
            image_shape: 图像形状
            
        Returns:
            Tuple[Tuple[int, int], Tuple[int, int, int, int]]: 中心点和边界框
        """
        if not landmarks:
            return (0, 0), (0, 0, 0, 0)
        
        # 计算边界框
        x_coords = [landmark[0] for landmark in landmarks]
        y_coords = [landmark[1] for landmark in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 添加边距
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image_shape[1], x_max + margin)
        y_max = min(image_shape[0], y_max + margin)
        
        # 计算中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        return (center_x, center_y), (x_min, y_min, x_max, y_max)
    
    def _update_hand_states_history(self, hands_info: List[Dict]):
        """
        更新手部状态历史
        
        Args:
            hands_info: 当前帧的手部信息
        """
        self.hand_states_history.append(hands_info)
        
        # 保持历史长度
        if len(self.hand_states_history) > self.max_history:
            self.hand_states_history.pop(0)
    
    def get_smoothed_gesture(self, hand_id: int) -> str:
        """
        获取平滑后的手势（基于历史状态）
        
        Args:
            hand_id: 手部ID
            
        Returns:
            str: 平滑后的手势
        """
        if not self.hand_states_history:
            return "Unknown"
        
        # 统计历史帧中的手势
        gesture_counts = {}
        for frame_hands in self.hand_states_history:
            for hand in frame_hands:
                if hand['hand_id'] == hand_id:
                    gesture = hand['gesture']
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # 返回出现次数最多的手势
        if gesture_counts:
            return max(gesture_counts, key=gesture_counts.get)
        else:
            return "Unknown"
    
    def visualize_hands(self, image: np.ndarray, hands_info: List[Dict], 
                       show_landmarks: bool = True, show_gestures: bool = True) -> np.ndarray:
        """
        可视化手部检测结果
        
        Args:
            image: 输入图像
            hands_info: 手部信息列表
            show_landmarks: 是否显示关键点
            show_gestures: 是否显示手势
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        annotated_image = image.copy()
        
        for hand in hands_info:
            landmarks = hand['landmarks_raw']
            gesture = hand['gesture']
            label = hand['label']
            confidence = hand['confidence']
            center = hand['center']
            bbox = hand['bbox']
            
            # 绘制手部关键点和连接线
            if show_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # 绘制边界框
            x_min, y_min, x_max, y_max = bbox
            cv.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 绘制手部标签和手势
            if show_gestures:
                # 手部标签
                label_text = f"{label} ({confidence:.2f})"
                cv.putText(annotated_image, label_text, (x_min, y_min - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 手势标签
                gesture_text = f"Gesture: {gesture}"
                cv.putText(annotated_image, gesture_text, (x_min, y_max + 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # 中心点
                cv.circle(annotated_image, center, 5, (255, 0, 0), -1)
        
        return annotated_image
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'hands'):
            self.hands.close()

# 使用示例
if __name__ == "__main__":
    # 初始化手部姿态识别器
    hand_pose = MediaPipeHandPose()
    
    # 打开摄像头
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    print("Hand Pose Detection Demo")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测手部
            hands_info = hand_pose.detect_hands(frame)
            
            # 可视化结果
            annotated_frame = hand_pose.visualize_hands(frame, hands_info)
            
            # 显示结果
            cv.imshow('Hand Pose Detection', annotated_frame)
            
            # 打印检测结果
            if hands_info:
                for hand in hands_info:
                    print(f"Hand {hand['hand_id']}: {hand['label']} - {hand['gesture']} (confidence: {hand['confidence']:.2f})")
            
            # 检查退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        hand_pose.cleanup()
        cv.destroyAllWindows()
