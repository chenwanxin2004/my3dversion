import numpy as np
import cv2 as cv
import time
import argparse
from collections import defaultdict, deque
import json
from datetime import datetime

try:
    import supervision as sv
    from ultralytics import YOLO
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("⚠️ Supervision or YOLO not available. Install with: pip install supervision ultralytics")

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️ RealSense not available. Install with: pip install pyrealsense2")

try:
    from mediapipe_hand_pose import MediaPipeHandPose
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available. Install with: pip install mediapipe")

try:
    from yolo_pose_detection import YOLOPoseDetector
    YOLO_POSE_AVAILABLE = True
except ImportError:
    YOLO_POSE_AVAILABLE = False
    print("⚠️ YOLO-pose not available. Check yolo_pose_detection.py")

class Supervision3DCrossingDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        结合Supervision、YOLO和3D点云的越线检测系统
        """
        self.pipeline = None
        self.config = None
        self.camera_intrinsics = None
        self.depth_scale = None
        
        # YOLO and Supervision setup
        self.model = None
        self.tracker = None
        self.line_zone = None
        self.line_zone_annotator = None
        self.box_annotator = None
        self.label_annotator = None
        
        # 3D point cloud parameters
        self.min_distance = 0.3
        self.max_distance = 5.0
        
        # 简化的3D越线检测参数
        self.crossing_distance = 1.0  # 越线距离阈值：1米内算越线
        self.min_crossing_height = 0.3  # 最小越线高度 (meters)
        self.max_crossing_height = 2.0  # 最大越线高度 (meters)
        
        # 放宽检测区域限制，避免误判
        self.detection_zone = {
            'min_x': -2.0,  # 左边界 (meters) - 放宽
            'max_x': 2.0,   # 右边界 (meters) - 放宽
            'min_z': 0.5,   # 最近距离 (meters) - 放宽
            'max_z': 5.0    # 最远距离 (meters) - 放宽
        }
        
        # 2D visualization line (for display only)
        self.line_start = sv.Point(200, 300)  # 2D显示线起点
        self.line_end = sv.Point(600, 300)    # 2D显示线终点
        self.line_thickness = 3
        
        # Statistics
        self.crossing_events = []
        self.person_count = 0
        self.unique_persons = set()
        self.person_tracks = defaultdict(lambda: {'first_seen': None, 'crossed': False, 'positions': deque(maxlen=10)})
        
        # Hand pose detection
        self.hand_pose_detector = None
        self.yolo_pose_detector = None  # YOLOv8-pose detector
        self.hand_gestures = []
        self.hand_gesture_history = deque(maxlen=10)
        
        # Initialize components
        self.init_realsense()
        self.init_yolo_supervision(model_path)
        self.init_mediapipe_hand_pose()
        self.init_yolo_pose_detector()
        
    def init_realsense(self):
        """Initialize Intel RealSense camera"""
        if not REALSENSE_AVAILABLE:
            print("❌ RealSense not available")
            return
            
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable depth and color streams
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()
            
            self.camera_intrinsics = np.array([
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            print(f"✅ RealSense camera initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize RealSense: {e}")
            self.pipeline = None
    
    def init_yolo_supervision(self, model_path):
        """Initialize YOLO model and Supervision components"""
        if not SUPERVISION_AVAILABLE:
            print("❌ Supervision/YOLO not available")
            return
            
        try:
            # Load YOLO model
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
            
            # Initialize Supervision components
            self.tracker = sv.ByteTrack()
            # 注意：我们不再使用LineZone进行检测，只用于可视化
            self.line_zone_annotator = sv.LineZoneAnnotator(thickness=self.line_thickness, text_thickness=1, text_scale=0.5)
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
            
            print("✅ Supervision components initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize YOLO/Supervision: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def init_mediapipe_hand_pose(self):
        """Initialize MediaPipe hand pose detection"""
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe not available")
            return
            
        try:
            self.hand_pose_detector = MediaPipeHandPose(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe hand pose detection initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe hand pose: {e}")
            import traceback
            traceback.print_exc()
            self.hand_pose_detector = None
    
    def init_yolo_pose_detector(self):
        """Initialize YOLOv8-pose detector using existing module"""
        if not YOLO_POSE_AVAILABLE:
            print("❌ YOLO-pose not available")
            return
            
        try:
            # 使用现有的YOLOPoseDetector模块
            self.yolo_pose_detector = YOLOPoseDetector("yolov8n-pose.pt")
            print("✅ YOLOv8-pose detector initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize YOLOv8-pose detector: {e}")
            import traceback
            traceback.print_exc()
            self.yolo_pose_detector = None
    
    def get_frames(self):
        """Get depth and color frames from RealSense"""
        if self.pipeline is None:
            return None, None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert depth to meters
            depth_image = depth_image.astype(np.float32) * self.depth_scale
            
            return depth_image, color_image
            
        except Exception as e:
            print(f"Error getting RealSense frames: {e}")
            return None, None
    
    def detect_persons_2d(self, color_image):
        """使用YOLO检测2D图像中的人物"""
        if self.model is None:
            return [], []
        
        try:
            # Run YOLO inference
            results = self.model(color_image, verbose=False)
            
            # Get the first result (single image)
            if len(results) > 0:
                result = results[0]
            else:
                return [], []
            
            # Filter for person class (class_id = 0)
            detections = sv.Detections.from_ultralytics(result)
            
            # Check if we have any detections
            if len(detections) == 0:
                return detections, result
            
            # Filter for person class
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                person_mask = detections.class_id == 0  # Person class
                detections = detections[person_mask]
            
            # Track persons
            detections = self.tracker.update_with_detections(detections)
            
            return detections, result
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def map_2d_to_3d(self, detections, depth_image):
        """将2D检测结果映射到3D坐标，检测整个边界框区域"""
        if self.camera_intrinsics is None or len(detections) == 0:
            return []
        
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        person_3d_info = []
        
        # Check if we have valid detections
        if not hasattr(detections, 'xyxy') or not hasattr(detections, 'tracker_id'):
            return []
        
        for i, (bbox, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            x1, y1, x2, y2 = bbox.astype(int)
            
            # 确保边界框在图像范围内
            x1 = max(0, min(x1, depth_image.shape[1] - 1))
            y1 = max(0, min(y1, depth_image.shape[0] - 1))
            x2 = max(0, min(x2, depth_image.shape[1] - 1))
            y2 = max(0, min(y2, depth_image.shape[0] - 1))
            
            # 改进的深度检测：更智能的采样和过滤
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 首先获取中心点深度作为参考
            center_depth = 0
            if (0 <= center_x < depth_image.shape[1] and 
                0 <= center_y < depth_image.shape[0]):
                center_depth = depth_image[center_y, center_x]
            
            # 如果中心点深度无效，跳过这个检测
            if center_depth <= 0.1 or center_depth > 10.0:
                continue
            
            # 基于中心点深度，只采样合理范围内的点
            depth_tolerance = 1.0  # 允许1米的深度差异
            min_valid_depth = max(0.1, center_depth - depth_tolerance)
            max_valid_depth = min(10.0, center_depth + depth_tolerance)
            
            valid_depths = [center_depth]
            min_depth = center_depth
            max_depth = center_depth
            
            # 采样策略：只采样中心区域，避免边界噪声
            sample_points = [
                (center_x, center_y),  # 中心点
                (center_x - 20, center_y),  # 左
                (center_x + 20, center_y),  # 右
                (center_x, center_y - 20),  # 上
                (center_x, center_y + 20),  # 下
            ]
            
            for px, py in sample_points:
                if (0 <= px < depth_image.shape[1] and 
                    0 <= py < depth_image.shape[0]):
                    
                    depth = depth_image[py, px]
                    
                    # 更严格的深度过滤：必须在合理范围内
                    if min_valid_depth <= depth <= max_valid_depth:
                        valid_depths.append(depth)
                        min_depth = min(min_depth, depth)
                        max_depth = max(max_depth, depth)
            
            # 如果有有效深度值
            if valid_depths and center_depth > 0:
                # 使用最小深度作为检测标准（最接近相机的部分）
                detection_depth = min_depth
                
                if self.min_distance < detection_depth < self.max_distance:
                    # 计算中心点的3D坐标
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    x_3d = (center_x - cx) * center_depth / fx
                    y_3d = (center_y - cy) * center_depth / fy
                    z_3d = center_depth
                    
                    person_3d_info.append({
                        'track_id': track_id,
                        'bbox_2d': bbox,
                        'center_2d': (center_x, center_y),
                        'position_3d': (x_3d, y_3d, z_3d),
                        'depth': center_depth,  # 中心点深度
                        'min_depth': min_depth,  # 最小深度（最接近相机的部分）
                        'max_depth': max_depth,  # 最大深度
                        'confidence': detections.confidence[i] if hasattr(detections, 'confidence') else 1.0
                    })
        
        return person_3d_info
    
    def detect_hand_gestures(self, color_image):
        """检测手部姿态和手势"""
        if self.hand_pose_detector is None:
            return []
        
        try:
            # 检测手部
            hands_info = self.hand_pose_detector.detect_hands(color_image)
            
            # 更新手势历史
            current_gestures = []
            for hand in hands_info:
                gesture_info = {
                    'hand_id': hand['hand_id'],
                    'label': hand['label'],
                    'gesture': hand['gesture'],
                    'confidence': hand['confidence'],
                    'center': hand['center'],
                    'bbox': hand['bbox'],
                    'pose_3d': hand['pose_3d']
                }
                current_gestures.append(gesture_info)
            
            self.hand_gesture_history.append(current_gestures)
            
            return hands_info
            
        except Exception as e:
            print(f"Error in hand gesture detection: {e}")
            return []
    
    def get_gesture_info(self, hands_info):
        """获取手势信息（不生成控制命令）"""
        gesture_info = []
        
        for hand in hands_info:
            gesture = hand['gesture']
            hand_id = hand['hand_id']
            label = hand['label']
            confidence = hand['confidence']
            
            # 只记录手势信息，不生成命令
            gesture_info.append({
                'hand_id': hand_id,
                'label': label,
                'gesture': gesture,
                'confidence': confidence,
                'center': hand['center'],
                'bbox': hand['bbox']
            })
        
        return gesture_info
    
    def detect_poses(self, color_image):
        """使用YOLOv8-pose检测人体姿势"""
        if self.yolo_pose_detector is None:
            return []
        
        try:
            # 使用现有模块的检测方法
            poses_info = self.yolo_pose_detector.detect_poses(color_image)
            return poses_info
            
        except Exception as e:
            print(f"Error in YOLOv8-pose detection: {e}")
            return []
    
    
    def project_3d_to_2d(self, point_3d):
        """将3D点投影到2D图像坐标"""
        if self.camera_intrinsics is None:
            return None
        
        x, y, z = point_3d
        
        # 投影到2D
        u = int(self.camera_intrinsics[0, 0] * x / z + self.camera_intrinsics[0, 2])
        v = int(self.camera_intrinsics[1, 1] * y / z + self.camera_intrinsics[1, 2])
        
        return (u, v)
    
    def check_line_crossing_3d(self, person_3d_info):
        """简化的3D越线检测：距离在1米内就算越线"""
        crossing_events = []
        
        for person in person_3d_info:
            track_id = person['track_id']
            position_3d = person['position_3d']
            center_2d = person['center_2d']
            depth = person['depth']
            
            # Update person track
            if track_id not in self.person_tracks:
                self.person_tracks[track_id]['first_seen'] = datetime.now()
            
            self.person_tracks[track_id]['positions'].append(center_2d)
            
            # 简化的越线检测：人体任何部分在1米内就算越线
            if not self.person_tracks[track_id]['crossed']:
                if self.is_person_crossing_simple(person):
                    self.person_tracks[track_id]['crossed'] = True
                    self.unique_persons.add(track_id)
                    
                    crossing_event = {
                        'track_id': track_id,
                        'timestamp': datetime.now(),
                        'position_3d': position_3d,
                        'position_2d': center_2d,
                        'depth': depth,
                        'direction': self.get_crossing_direction_simple(position_3d)
                    }
                    
                    crossing_events.append(crossing_event)
                    self.crossing_events.append(crossing_event)
        
        return crossing_events
    
    def is_person_crossing_simple(self, person_info):
        """改进的越线检测：更合理的判断逻辑"""
        position_3d = person_info['position_3d']
        center_depth = person_info['depth']
        min_depth = person_info.get('min_depth', center_depth)
        max_depth = person_info.get('max_depth', center_depth)
        x, y, z = position_3d
        
        # 1. 检查高度是否在合理范围内
        if not (self.min_crossing_height <= y <= self.max_crossing_height):
            return False
        
        # 2. 检查是否在检测区域内
        zone = self.detection_zone
        if not (zone['min_x'] <= x <= zone['max_x'] and 
                zone['min_z'] <= z <= zone['max_z']):
            return False
        
        # 3. 改进的越线判断逻辑：
        # - 如果中心深度在1米内，直接越线
        # - 如果中心深度在1-2米内，且最小深度在1米内，也算越线
        # - 如果深度范围过大（可能是噪声），使用中心深度判断
        
        depth_range = max_depth - min_depth
        
        # 如果深度范围过大（>2米），可能是噪声，使用中心深度
        if depth_range > 2.0:
            return center_depth <= self.crossing_distance
        
        # 正常情况：使用最小深度判断
        return min_depth <= self.crossing_distance
    
    def get_crossing_direction_simple(self, position_3d):
        """获取越线方向（简化版）"""
        x, y, z = position_3d
        
        # 基于x坐标判断方向
        if x < -0.5:
            return "From Left"
        elif x > 0.5:
            return "From Right"
        else:
            return "From Center"
    
    def create_visualization(self, color_image, detections, person_3d_info, crossing_events, detection_summary):
        """创建可视化界面"""
        # Annotate detections
        if len(detections) > 0:
            labels = [f"Person {track_id}" for track_id in detections.tracker_id]
            annotated_image = self.box_annotator.annotate(color_image.copy(), detections)
            annotated_image = self.label_annotator.annotate(annotated_image, detections, labels)
        else:
            annotated_image = color_image.copy()
        
        # 添加手部检测可视化（MediaPipe）
        if detection_summary['hands']['data'] and self.hand_pose_detector:
            annotated_image = self.hand_pose_detector.visualize_hands(annotated_image, detection_summary['hands']['data'])
        
        # 添加YOLOv8-pose可视化
        if detection_summary['poses']['data'] and self.yolo_pose_detector:
            annotated_image = self.yolo_pose_detector.visualize_poses(annotated_image, detection_summary['poses']['data'])
        
        # 移除所有多余的线条，真正的检测基于深度值
        
        # Add 3D information overlay with depth highlighting
        for person in person_3d_info:
            center_2d = person['center_2d']
            position_3d = person['position_3d']
            track_id = person['track_id']
            depth = person['depth']
            min_depth = person.get('min_depth', depth)
            
            # 改进的越线判断逻辑
            depth_range = person.get('max_depth', depth) - min_depth
            is_crossing = False
            
            # 使用与检测逻辑相同的判断
            if depth_range > 2.0:  # 深度范围过大，使用中心深度
                is_crossing = depth <= self.crossing_distance
                detection_method = "Center"
            else:  # 正常情况，使用最小深度
                is_crossing = min_depth <= self.crossing_distance
                detection_method = "Min"
            
            # 设置颜色和状态
            if is_crossing:
                color = (0, 0, 255)  # 红色：越线
                status = "CROSSING!"
            else:
                color = (0, 255, 0)  # 绿色：正常
                status = "Normal"
            
            # 添加深度信息文本
            depth_text = f"ID:{track_id} {detection_method}:{min_depth if detection_method == 'Min' else depth:.2f}m {status}"
            cv.putText(annotated_image, depth_text, 
                      (int(center_2d[0] - 80), int(center_2d[1] - 10)), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 添加详细深度信息
            detail_text = f"Center:{depth:.2f}m Range:{min_depth:.2f}-{person.get('max_depth', depth):.2f}m"
            cv.putText(annotated_image, detail_text, 
                      (int(center_2d[0] - 80), int(center_2d[1] + 15)), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 添加手势信息可视化（从摘要中获取）
        if detection_summary['hands']['gestures']:
            for i, gesture in enumerate(detection_summary['hands']['gestures']):
                gesture_cn = self._translate_gesture_to_chinese(gesture['gesture'])
                hand_cn = "左手" if gesture['label'] == 'Left' else "右手"
                gesture_text = f"手势: {gesture_cn} ({hand_cn})"
                cv.putText(annotated_image, gesture_text, (10, 200 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 移除统计信息覆盖层，改为在独立窗口中显示
        
        return annotated_image
    
    def create_statistics_panel(self, detection_summary):
        """创建美观的统计信息面板"""
        panel_height = 500
        panel_width = 600
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # 设置渐变背景
        for y in range(panel_height):
            intensity = int(30 + (y / panel_height) * 20)
            panel[y, :] = (intensity, intensity, intensity)
        
        # 绘制标题区域
        cv.rectangle(panel, (10, 10), (590, 60), (0, 100, 200), -1)
        cv.putText(panel, "3D越线检测系统", (50, 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 绘制分隔线
        cv.line(panel, (20, 80), (580, 80), (100, 100, 100), 2)
        
        # 实时统计信息
        y_pos = 110
        cv.putText(panel, "实时统计信息", (30, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        
        # 当前帧信息（从摘要中获取）
        current_stats = [
            ("当前帧人数", detection_summary['persons']['count'], (0, 255, 0)),
            ("检测到手部", detection_summary['hands']['count'], (255, 255, 0)),
            ("识别出手势", len(detection_summary['hands']['gestures']), (255, 165, 0)),
            ("检测到姿势", detection_summary['poses']['count'], (255, 0, 255))
        ]
        
        for label, value, color in current_stats:
            cv.putText(panel, f"{label}:", (50, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv.putText(panel, str(value), (350, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30
        
        # 历史统计信息
        y_pos += 20
        cv.putText(panel, "历史统计信息", (30, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        
        historical_stats = [
            ("累计检测人数", detection_summary['statistics']['total_persons'], (0, 255, 0)),
            ("越线事件总数", detection_summary['statistics']['crossing_events'], (0, 0, 255)),
            ("活跃跟踪数", detection_summary['statistics']['active_tracks'], (255, 255, 0))
        ]
        
        for label, value, color in historical_stats:
            cv.putText(panel, f"{label}:", (50, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv.putText(panel, str(value), (350, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30
        
        # 最近的手势信息（从摘要中获取）
        if detection_summary['hands']['gestures']:
            y_pos += 20
            cv.putText(panel, "最近手势", (30, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
            
            for i, gesture in enumerate(detection_summary['hands']['gestures'][:3]):  # 只显示最近3个
                # 将手势名称翻译为中文
                gesture_cn = self._translate_gesture_to_chinese(gesture['gesture'])
                hand_cn = "左手" if gesture['label'] == 'Left' else "右手"
                gesture_text = f"  {hand_cn}: {gesture_cn}"
                cv.putText(panel, gesture_text, (50, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
        
        # 最近的越线事件
        if self.crossing_events:
            y_pos += 20
            cv.putText(panel, "最近越线事件", (30, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
            
            recent_events = self.crossing_events[-3:]  # 只显示最近3个
            for event in recent_events:
                direction = event.get('direction', 'Unknown')
                # 将方向翻译为中文
                direction_cn = self._translate_direction_to_chinese(direction)
                event_text = f"  人员 {event['track_id']} ({direction_cn})"
                cv.putText(panel, event_text, (50, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                y_pos += 25
        
        # 添加边框
        cv.rectangle(panel, (5, 5), (595, 495), (100, 100, 100), 2)
        
        return panel
    
    def _translate_gesture_to_chinese(self, gesture):
        """将手势名称翻译为中文"""
        gesture_translations = {
            'Fist': '握拳',
            'Open Hand': '张开手掌',
            'Pointing': '指向',
            'Peace Sign': '胜利手势',
            'OK Sign': 'OK手势',
            'Thumbs Up': '竖拇指',
            'Other': '其他手势',
            'Unknown': '未知手势'
        }
        return gesture_translations.get(gesture, gesture)
    
    def _translate_direction_to_chinese(self, direction):
        """将方向名称翻译为中文"""
        direction_translations = {
            'From Left': '从左侧',
            'From Right': '从右侧',
            'From Center': '从中央',
            'Unknown': '未知方向'
        }
        return direction_translations.get(direction, direction)
    
    def create_detection_summary(self, person_3d_info, hands_info, poses_info):
        """创建统一的检测信息摘要，减少参数冗余"""
        summary = {
            # 人员信息
            'persons': {
                'count': len(person_3d_info),
                'data': person_3d_info
            },
            
            # 手部检测信息（MediaPipe）
            'hands': {
                'count': len(hands_info) if hands_info else 0,
                'data': hands_info if hands_info else [],
                'gestures': []
            },
            
            # 姿势检测信息（YOLOv8-pose）
            'poses': {
                'count': len(poses_info) if poses_info else 0,
                'data': poses_info if poses_info else [],
                'hand_actions': []
            },
            
            # 统计信息
            'statistics': {
                'total_persons': len(self.unique_persons),
                'crossing_events': len(self.crossing_events),
                'active_tracks': len(self.person_tracks)
            }
        }
        
        # 提取手势信息
        if hands_info:
            for hand in hands_info:
                summary['hands']['gestures'].append({
                    'label': hand['label'],
                    'gesture': hand['gesture'],
                    'confidence': hand['confidence']
                })
        
        # 提取手部动作信息
        if poses_info:
            for pose in poses_info:
                summary['poses']['hand_actions'].append({
                    'person_id': pose['person_id'],
                    'left_hand': pose['hand_actions']['left_hand'],
                    'right_hand': pose['hand_actions']['right_hand']
                })
        
        return summary
    
    
    def colorize_depth(self, depth_image):
        """将深度图转换为彩色图像"""
        # 限制深度范围到0-5米
        depth_clipped = np.clip(depth_image, 0, 5.0)
        
        # 归一化到0-255
        depth_normalized = (depth_clipped / 5.0 * 255).astype(np.uint8)
        
        # 应用颜色映射（近处红色，远处蓝色）
        depth_colored = cv.applyColorMap(depth_normalized, cv.COLORMAP_JET)
        
        # 添加距离标注
        height, width = depth_colored.shape[:2]
        cv.putText(depth_colored, "Depth Map (Red=Close, Blue=Far)", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加距离刻度
        for i, distance in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            y_pos = int(height * (1 - distance / 5.0))
            cv.line(depth_colored, (width - 100, y_pos), (width - 80, y_pos), (255, 255, 255), 1)
            cv.putText(depth_colored, f"{distance}m", 
                      (width - 75, y_pos + 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return depth_colored
    
    def save_statistics(self, filename="crossing_statistics.json"):
        """保存统计信息到文件"""
        stats = {
            'total_unique_persons': len(self.unique_persons),
            'total_crossing_events': len(self.crossing_events),
            'crossing_events': [
                {
                    'track_id': event['track_id'],
                    'timestamp': event['timestamp'].isoformat(),
                    'position_3d': event['position_3d'],
                    'position_2d': event['position_2d'],
                    'depth': event['depth']
                }
                for event in self.crossing_events
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Statistics saved to {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.stop()
        if self.hand_pose_detector:
            self.hand_pose_detector.cleanup()
        if self.yolo_pose_detector:
            self.yolo_pose_detector.cleanup()
        cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Supervision + YOLO + 3D Point Cloud Crossing Detection')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--save-stats', action='store_true', help='Save statistics to file')
    
    args = parser.parse_args()
    
    print("3D Crossing Detection + Hand Gesture Recognition")
    print("=" * 60)
    print("Features:")
    print("- YOLO person detection")
    print("- RealSense depth information display")
    print("- Simple rule: Distance <= 1m = Crossing")
    print("- Height filtering (0.3-2.0m)")
    print("- Detection zone: 1-3m range, ±1m width")
    print("- Real-time depth map visualization")
    print("- Person counting and tracking")
    print("- MediaPipe hand gesture recognition")
    print("- YOLOv8-pose body pose detection")
    print("- Real-time gesture and pose display")
    print("Press 'q' or ESC to quit, 's' to save statistics")
    
    # Initialize detector
    detector = Supervision3DCrossingDetector(args.model)
    
    if detector.pipeline is None:
        print("❌ Failed to initialize RealSense camera")
        return
    
    if detector.model is None:
        print("❌ Failed to initialize YOLO/Supervision")
        return
    
    try:
        frame_count = 0
        while True:
            frame_count += 1
            
            # Get frames from RealSense
            depth_image, color_image = detector.get_frames()
            
            if depth_image is None or color_image is None:
                print("❌ Failed to get frames from RealSense")
                time.sleep(0.1)
                continue
            
            # Detect persons in 2D
            detections, yolo_results = detector.detect_persons_2d(color_image)
            
            # Map 2D detections to 3D coordinates
            person_3d_info = detector.map_2d_to_3d(detections, depth_image)
            
            # Check for 3D line crossing
            crossing_events = detector.check_line_crossing_3d(person_3d_info)
            
            # Detect hand gestures (MediaPipe) - 可选，与YOLOv8-pose二选一
            hands_info = detector.detect_hand_gestures(color_image)
            
            # Detect poses (YOLOv8-pose) - 包含手部动作检测
            poses_info = detector.detect_poses(color_image)
            
            # 合并检测信息，减少冗余
            detection_summary = detector.create_detection_summary(person_3d_info, hands_info, poses_info)
            
            # Create visualizations
            annotated_image = detector.create_visualization(color_image, detections, person_3d_info, crossing_events, detection_summary)
            stats_panel = detector.create_statistics_panel(detection_summary)
            
            # Display results in multiple windows
            cv.imshow('Crossing Detection', annotated_image)
            cv.imshow('Statistics', stats_panel)
            
            # 显示深度图
            depth_colored = detector.colorize_depth(depth_image)
            cv.imshow('Depth Map', depth_colored)
            
            # Print crossing events
            if crossing_events:
                for event in crossing_events:
                    direction = event.get('direction', 'Unknown')
                    direction_cn = detector._translate_direction_to_chinese(direction)
                    print(f"🚨 人员 {event['track_id']} 在 {event['timestamp'].strftime('%H:%M:%S')} 越线 ({direction_cn})")
            
            # Print gesture information
            if detection_summary['hands']['gestures']:
                for gesture in detection_summary['hands']['gestures']:
                    gesture_cn = detector._translate_gesture_to_chinese(gesture['gesture'])
                    hand_cn = "左手" if gesture['label'] == 'Left' else "右手"
                    print(f"👋 检测到手势: {gesture_cn} ({hand_cn}) (置信度: {gesture['confidence']:.2f})")
            
            # Print pose information
            if detection_summary['poses']['hand_actions']:
                for hand_action in detection_summary['poses']['hand_actions']:
                    left_action = hand_action['left_hand']['action']
                    right_action = hand_action['right_hand']['action']
                    print(f"🤸 检测到姿势: 人员 {hand_action['person_id']} - 左手: {left_action}, 右手: {right_action}")
            
            # Handle key presses
            key = cv.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):  # Save statistics
                detector.save_statistics()
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        if args.save_stats:
            detector.save_statistics()
        detector.cleanup()
        print("Crossing detection stopped.")

if __name__ == "__main__":
    main()