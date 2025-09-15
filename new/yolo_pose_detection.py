import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional, Dict
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install with: pip install ultralytics")

class YOLOPoseDetector:
    """
    YOLOv8-pose人体姿势检测模块
    功能：
    1. 人体关键点检测
    2. 手部动作识别
    3. 姿势分析
    4. 可视化关键点和姿势
    """
    
    def __init__(self, model_path="yolov8n-pose.pt"):
        """
        初始化YOLOv8-pose检测器
        
        Args:
            model_path: YOLOv8-pose模型路径
        """
        self.model = None
        self.model_path = model_path
        
        # COCO pose关键点索引
        self.POSE_LANDMARKS = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # 手部关键点（手腕）
        self.HAND_POINTS = ['left_wrist', 'right_wrist']
        
        # 姿势历史（用于平滑检测）
        self.pose_history = []
        self.max_history = 5
        
        # 初始化模型
        self.init_model()
    
    def init_model(self):
        """初始化YOLOv8-pose模型"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available")
            return
            
        try:
            # 尝试加载模型，处理PyTorch 2.6的安全限制
            import torch
            if hasattr(torch, 'serialization'):
                # 添加安全的全局变量
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.PoseModel'])
            
            self.model = YOLO(self.model_path)
            print(f"✅ YOLOv8-pose model loaded: {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load YOLOv8-pose model: {e}")
            print("Trying alternative loading method...")
            
            try:
                # 尝试使用weights_only=False
                import torch
                torch.load = lambda *args, **kwargs: torch.load(*args, **kwargs, weights_only=False)
                self.model = YOLO(self.model_path)
                print(f"✅ YOLOv8-pose model loaded with weights_only=False: {self.model_path}")
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                self.model = None
    
    def detect_poses(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人体姿势
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            List[Dict]: 检测到的人体姿势信息列表
        """
        if self.model is None:
            return []
        
        try:
            # 运行YOLOv8-pose推理
            results = self.model(image, verbose=False)
            
            poses_info = []
            
            if len(results) > 0:
                result = results[0]
                
                # 检查是否有检测结果
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()  # [N, 17, 3] (x, y, confidence)
                    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else None
                    
                    for i, person_keypoints in enumerate(keypoints):
                        # 提取关键点坐标和置信度
                        landmarks = {}
                        for name, idx in self.POSE_LANDMARKS.items():
                            x, y, conf = person_keypoints[idx]
                            landmarks[name] = {
                                'x': int(x),
                                'y': int(y),
                                'confidence': float(conf)
                            }
                        
                        # 计算边界框
                        bbox = None
                        if boxes is not None and i < len(boxes):
                            x1, y1, x2, y2, conf, cls = boxes[i]
                            bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        # 分析手部动作
                        hand_actions = self._analyze_hand_actions(landmarks)
                        
                        # 计算人体中心点
                        center = self._calculate_person_center(landmarks)
                        
                        pose_info = {
                            'person_id': i,
                            'landmarks': landmarks,
                            'bbox': bbox,
                            'center': center,
                            'hand_actions': hand_actions,
                            'keypoints_raw': person_keypoints
                        }
                        
                        poses_info.append(pose_info)
            
            # 更新姿势历史
            self._update_pose_history(poses_info)
            
            return poses_info
            
        except Exception as e:
            print(f"Error in YOLOv8-pose detection: {e}")
            return []
    
    def _analyze_hand_actions(self, landmarks: Dict) -> Dict:
        """
        分析手部动作
        
        Args:
            landmarks: 人体关键点字典
            
        Returns:
            Dict: 手部动作信息
        """
        hand_actions = {
            'left_hand': {'action': 'Unknown', 'confidence': 0.0},
            'right_hand': {'action': 'Unknown', 'confidence': 0.0}
        }
        
        # 分析左手动作
        left_hand_action = self._analyze_single_hand(
            landmarks.get('left_shoulder'),
            landmarks.get('left_elbow'),
            landmarks.get('left_wrist')
        )
        hand_actions['left_hand'] = left_hand_action
        
        # 分析右手动作
        right_hand_action = self._analyze_single_hand(
            landmarks.get('right_shoulder'),
            landmarks.get('right_elbow'),
            landmarks.get('right_wrist')
        )
        hand_actions['right_hand'] = right_hand_action
        
        return hand_actions
    
    def _analyze_single_hand(self, shoulder, elbow, wrist) -> Dict:
        """
        分析单只手部动作
        
        Args:
            shoulder: 肩膀关键点
            elbow: 肘部关键点
            wrist: 手腕关键点
            
        Returns:
            Dict: 手部动作信息
        """
        if not all([shoulder, elbow, wrist]):
            return {'action': 'Unknown', 'confidence': 0.0}
        
        # 检查关键点置信度
        min_confidence = 0.3
        if (shoulder['confidence'] < min_confidence or 
            elbow['confidence'] < min_confidence or 
            wrist['confidence'] < min_confidence):
            return {'action': 'Low Confidence', 'confidence': 0.0}
        
        # 计算手臂角度
        shoulder_point = (shoulder['x'], shoulder['y'])
        elbow_point = (elbow['x'], elbow['y'])
        wrist_point = (wrist['x'], wrist['y'])
        
        # 计算上臂角度（肩膀到肘部）
        upper_arm_angle = self._calculate_angle(shoulder_point, elbow_point, (elbow_point[0], elbow_point[1] - 100))
        
        # 计算前臂角度（肘部到手腕）
        forearm_angle = self._calculate_angle(elbow_point, wrist_point, (wrist_point[0], wrist_point[1] - 100))
        
        # 计算手臂长度比例
        upper_arm_length = self._calculate_distance(shoulder_point, elbow_point)
        forearm_length = self._calculate_distance(elbow_point, wrist_point)
        
        # 基于角度和位置判断手部动作
        action = self._classify_hand_action(upper_arm_angle, forearm_angle, upper_arm_length, forearm_length, wrist_point)
        
        # 计算置信度（基于关键点置信度的平均值）
        confidence = (shoulder['confidence'] + elbow['confidence'] + wrist['confidence']) / 3
        
        return {
            'action': action,
            'confidence': confidence,
            'upper_arm_angle': upper_arm_angle,
            'forearm_angle': forearm_angle,
            'upper_arm_length': upper_arm_length,
            'forearm_length': forearm_length
        }
    
    def _classify_hand_action(self, upper_arm_angle, forearm_angle, upper_arm_length, forearm_length, wrist_point):
        """
        根据角度和位置分类手部动作
        
        Args:
            upper_arm_angle: 上臂角度
            forearm_angle: 前臂角度
            upper_arm_length: 上臂长度
            forearm_length: 前臂长度
            wrist_point: 手腕位置
            
        Returns:
            str: 手部动作类型
        """
        # 手臂长度比例
        arm_ratio = forearm_length / max(upper_arm_length, 1)
        
        # 基于角度判断
        if 45 <= upper_arm_angle <= 135 and 45 <= forearm_angle <= 135:
            if arm_ratio > 0.8:  # 手臂相对伸直
                return "Raised Hand"
            else:
                return "Bent Arm"
        
        elif upper_arm_angle > 135:
            if forearm_angle > 90:
                return "Extended Arm"
            else:
                return "Pointing"
        
        elif upper_arm_angle < 45:
            if forearm_angle < 45:
                return "Lowered Hand"
            else:
                return "Resting"
        
        # 基于手腕位置判断
        if wrist_point[1] < 200:  # 手腕在图像上方
            return "Raised Hand"
        elif wrist_point[1] > 400:  # 手腕在图像下方
            return "Lowered Hand"
        
        return "Normal Position"
    
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
    
    def _calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        计算两点之间的距离
        
        Args:
            p1, p2: 两个点的坐标
            
        Returns:
            float: 距离
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_person_center(self, landmarks: Dict) -> Tuple[int, int]:
        """
        计算人体中心点
        
        Args:
            landmarks: 人体关键点字典
            
        Returns:
            Tuple[int, int]: 中心点坐标
        """
        # 使用肩膀和髋部计算中心点
        left_shoulder = landmarks.get('left_shoulder')
        right_shoulder = landmarks.get('right_shoulder')
        left_hip = landmarks.get('left_hip')
        right_hip = landmarks.get('right_hip')
        
        if all([left_shoulder, right_shoulder, left_hip, right_hip]):
            center_x = (left_shoulder['x'] + right_shoulder['x'] + left_hip['x'] + right_hip['x']) // 4
            center_y = (left_shoulder['y'] + right_shoulder['y'] + left_hip['y'] + right_hip['y']) // 4
            return (center_x, center_y)
        
        # 如果关键点不完整，使用可用点
        valid_points = [landmarks[key] for key in landmarks.keys() if landmarks[key]['confidence'] > 0.3]
        if valid_points:
            center_x = sum(point['x'] for point in valid_points) // len(valid_points)
            center_y = sum(point['y'] for point in valid_points) // len(valid_points)
            return (center_x, center_y)
        
        return (0, 0)
    
    def _update_pose_history(self, poses_info: List[Dict]):
        """
        更新姿势历史
        
        Args:
            poses_info: 当前帧的姿势信息
        """
        self.pose_history.append(poses_info)
        
        # 保持历史长度
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
    
    def get_smoothed_hand_actions(self, person_id: int) -> Dict:
        """
        获取平滑后的手部动作（基于历史状态）
        
        Args:
            person_id: 人体ID
            
        Returns:
            Dict: 平滑后的手部动作
        """
        if not self.pose_history:
            return {'left_hand': {'action': 'Unknown'}, 'right_hand': {'action': 'Unknown'}}
        
        # 统计历史帧中的手部动作
        left_actions = []
        right_actions = []
        
        for frame_poses in self.pose_history:
            for pose in frame_poses:
                if pose['person_id'] == person_id:
                    left_actions.append(pose['hand_actions']['left_hand']['action'])
                    right_actions.append(pose['hand_actions']['right_hand']['action'])
                    break
        
        # 返回出现次数最多的动作
        def most_common_action(actions):
            if not actions:
                return 'Unknown'
            return max(set(actions), key=actions.count)
        
        return {
            'left_hand': {'action': most_common_action(left_actions)},
            'right_hand': {'action': most_common_action(right_actions)}
        }
    
    def visualize_poses(self, image: np.ndarray, poses_info: List[Dict], 
                       show_keypoints: bool = True, show_hand_actions: bool = True) -> np.ndarray:
        """
        可视化人体姿势检测结果
        
        Args:
            image: 输入图像
            poses_info: 姿势信息列表
            show_keypoints: 是否显示关键点
            show_hand_actions: 是否显示手部动作
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        annotated_image = image.copy()
        
        for pose in poses_info:
            landmarks = pose['landmarks']
            bbox = pose['bbox']
            center = pose['center']
            hand_actions = pose['hand_actions']
            
            # 绘制边界框
            if bbox:
                x1, y1, x2, y2 = bbox
                cv.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制关键点
            if show_keypoints:
                for name, landmark in landmarks.items():
                    if landmark['confidence'] > 0.3:
                        x, y = landmark['x'], landmark['y']
                        # 手部关键点用不同颜色
                        if name in self.HAND_POINTS:
                            cv.circle(annotated_image, (x, y), 8, (0, 0, 255), -1)  # 红色
                        else:
                            cv.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)  # 蓝色
                        
                        # 显示关键点名称
                        cv.putText(annotated_image, name, (x + 10, y - 10), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 绘制骨架连接
            self._draw_skeleton(annotated_image, landmarks)
            
            # 显示手部动作
            if show_hand_actions:
                left_action = hand_actions['left_hand']['action']
                right_action = hand_actions['right_hand']['action']
                
                # 显示左手动作
                if left_action != 'Unknown':
                    left_wrist = landmarks.get('left_wrist')
                    if left_wrist and left_wrist['confidence'] > 0.3:
                        cv.putText(annotated_image, f"L: {left_action}", 
                                  (left_wrist['x'] - 50, left_wrist['y'] - 20), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # 显示右手动作
                if right_action != 'Unknown':
                    right_wrist = landmarks.get('right_wrist')
                    if right_wrist and right_wrist['confidence'] > 0.3:
                        cv.putText(annotated_image, f"R: {right_action}", 
                                  (right_wrist['x'] + 10, right_wrist['y'] - 20), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 显示人体中心点
            cv.circle(annotated_image, center, 10, (255, 255, 0), -1)
            cv.putText(annotated_image, f"Person {pose['person_id']}", 
                      (center[0] - 30, center[1] + 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def _draw_skeleton(self, image: np.ndarray, landmarks: Dict):
        """
        绘制人体骨架连接
        
        Args:
            image: 图像
            landmarks: 关键点字典
        """
        # 骨架连接定义
        skeleton_connections = [
            # 头部
            ('left_eye', 'right_eye'),
            ('left_eye', 'nose'),
            ('right_eye', 'nose'),
            ('left_ear', 'left_eye'),
            ('right_ear', 'right_eye'),
            
            # 躯干
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # 左臂
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            
            # 右臂
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # 左腿
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            
            # 右腿
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]
        
        # 绘制连接线
        for connection in skeleton_connections:
            point1_name, point2_name = connection
            point1 = landmarks.get(point1_name)
            point2 = landmarks.get(point2_name)
            
            if (point1 and point2 and 
                point1['confidence'] > 0.3 and point2['confidence'] > 0.3):
                
                cv.line(image, 
                       (point1['x'], point1['y']), 
                       (point2['x'], point2['y']), 
                       (0, 255, 0), 2)
    
    def cleanup(self):
        """清理资源"""
        # YOLO模型会自动清理
        pass

# 使用示例
if __name__ == "__main__":
    # 初始化YOLOv8-pose检测器
    pose_detector = YOLOPoseDetector("yolov8n-pose.pt")
    
    # 打开摄像头
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    print("YOLOv8-pose Detection Demo")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测人体姿势
            poses_info = pose_detector.detect_poses(frame)
            
            # 可视化结果
            annotated_frame = pose_detector.visualize_poses(frame, poses_info)
            
            # 显示结果
            cv.imshow('YOLOv8-pose Detection', annotated_frame)
            
            # 打印检测结果
            if poses_info:
                for pose in poses_info:
                    left_action = pose['hand_actions']['left_hand']['action']
                    right_action = pose['hand_actions']['right_hand']['action']
                    print(f"Person {pose['person_id']}: Left={left_action}, Right={right_action}")
            
            # 检查退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        pose_detector.cleanup()
        cv.destroyAllWindows()
