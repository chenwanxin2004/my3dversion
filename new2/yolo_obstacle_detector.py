#!/usr/bin/env python3
"""
YOLOv8分割障碍物检测模块
使用YOLOv8-seg进行语义分割，辅助生成障碍物掩膜
"""

import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not available, YOLOv8-seg功能不可用")

class YOLOObstacleDetector:
    """
    YOLOv8分割障碍物检测器
    使用YOLOv8-seg进行语义分割，识别和分离障碍物
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n-seg.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        初始化YOLOv8障碍物检测器
        
        Args:
            model_path: YOLOv8分割模型路径
            confidence_threshold: 检测置信度阈值
            device: 运行设备 ("cpu", "cuda", "auto")
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # 障碍物类别（使用YOLOv8模型自带的类别名称）
        # 这些是COCO数据集的80个类别，YOLOv8会自动识别
        self.obstacle_class_names = {
            # 家具类
            'chair', 'couch', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush',
            
            # 交通工具类
            'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'bicycle',
            
            # 其他物体
            'bottle', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake',
            
            # 运动用品
            'sports ball', 'tennis racket',
        }
        
        # 手部相关类别（需要排除）
        self.hand_related_class_names = {
            'person',  # 人体，包含手部
        }
        
        self.model = None
        self.is_initialized = False
        
        # 性能统计
        self.inference_times = []
        self.max_inference_time = 0.1  # 最大推理时间（秒）
        
        if YOLO_AVAILABLE:
            self._initialize_model()
        else:
            print("❌ YOLOv8不可用，请安装ultralytics: pip install ultralytics")
    
    def _initialize_model(self):
        """
        初始化YOLOv8模型
        """
        try:
            print(f"🔄 加载YOLOv8分割模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置设备
            if self.device == "auto":
                self.device = "cuda" if self.model.device.type == "cuda" else "cpu"
            
            print(f"✅ YOLOv8模型加载成功")
            print(f"   设备: {self.device}")
            print(f"   置信度阈值: {self.confidence_threshold}")
            print(f"   障碍物类别数: {len(self.obstacle_class_names)}")
            print(f"   模型自带类别: {len(self.model.names)} 个")
            
            # 显示模型支持的所有类别
            print(f"   模型支持的类别: {list(self.model.names.values())}")
            
            self.is_initialized = True
            
        except Exception as e:
            print(f"❌ YOLOv8模型初始化失败: {e}")
            self.is_initialized = False
    
    def detect_obstacles(self, 
                        image: np.ndarray, 
                        hand_landmarks_3d: Optional[List] = None) -> Dict[str, Any]:
        """
        检测图像中的障碍物
        
        Args:
            image: 输入图像 (BGR格式)
            hand_landmarks_3d: 手部关键点3D坐标列表
            
        Returns:
            Dict: 检测结果
        """
        if not self.is_initialized:
            return self._get_empty_result()
        
        start_time = time.time()
        
        try:
            # YOLOv8推理
            results = self.model(image, 
                               conf=self.confidence_threshold,
                               device=self.device,
                               verbose=False)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 保持最近100次的推理时间
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            # 处理检测结果
            detection_result = self._process_detection_results(results[0], image.shape)
            
            # 生成障碍物掩膜
            obstacle_mask = self._generate_obstacle_mask(
                detection_result, image.shape, hand_landmarks_3d
            )
            
            detection_result.update({
                'obstacle_mask': obstacle_mask,
                'inference_time': inference_time,
                'fps': 1.0 / inference_time if inference_time > 0 else 0
            })
            
            return detection_result
            
        except Exception as e:
            print(f"❌ YOLOv8推理失败: {e}")
            return self._get_empty_result()
    
    def _process_detection_results(self, result, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        处理YOLOv8检测结果
        
        Args:
            result: YOLOv8检测结果
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            Dict: 处理后的检测结果
        """
        height, width = image_shape[:2]
        
        detection_result = {
            'obstacles': [],
            'hand_regions': [],
            'obstacle_count': 0,
            'hand_region_count': 0,
            'total_detections': 0
        }
        
        if result.masks is None:
            return detection_result
        
        # 处理每个检测结果
        for i, (box, mask, conf, cls) in enumerate(zip(
            result.boxes.xyxy.cpu().numpy(),
            result.masks.data.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        )):
            class_id = int(cls)
            class_name = result.names[class_id]
            confidence = float(conf)
            
            # 调整掩膜大小到原图尺寸
            mask_resized = cv.resize(mask, (width, height))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
            
            detection_info = {
                'id': i,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': box.tolist(),
                'mask': mask_binary,
                'area': np.sum(mask_binary > 0)
            }
            
            # 分类为障碍物或手部区域
            if class_name in self.obstacle_class_names:
                detection_result['obstacles'].append(detection_info)
                detection_result['obstacle_count'] += 1
            elif class_name in self.hand_related_class_names:
                detection_result['hand_regions'].append(detection_info)
                detection_result['hand_region_count'] += 1
            
            detection_result['total_detections'] += 1
        
        return detection_result
    
    def _generate_obstacle_mask(self, 
                               detection_result: Dict[str, Any],
                               image_shape: Tuple[int, int, int],
                               hand_landmarks_3d: Optional[List] = None) -> np.ndarray:
        """
        生成障碍物掩膜（改进版本，确保与深度图对齐）
        
        Args:
            detection_result: 检测结果
            image_shape: 图像形状
            hand_landmarks_3d: 手部关键点3D坐标
            
        Returns:
            np.ndarray: 障碍物掩膜
        """
        height, width = image_shape[:2]
        obstacle_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 合并所有障碍物掩膜
        for obstacle in detection_result['obstacles']:
            mask = obstacle['mask']
            
            # 确保掩膜尺寸与图像尺寸一致
            if mask.shape != (height, width):
                mask = cv.resize(mask, (width, height))
            
            # 二值化掩膜
            mask_binary = (mask > 128).astype(np.uint8) * 255
            obstacle_mask = cv.bitwise_or(obstacle_mask, mask_binary)
        
        # 排除手部区域
        obstacle_mask = self._exclude_hand_regions(
            obstacle_mask, detection_result['hand_regions'], hand_landmarks_3d
        )
        
        # 后处理：噪声过滤和形态学操作
        obstacle_mask = self._post_process_mask(obstacle_mask)
        
        return obstacle_mask
    
    def _exclude_hand_regions(self, 
                             obstacle_mask: np.ndarray,
                             hand_regions: List[Dict],
                             hand_landmarks_3d: Optional[List] = None) -> np.ndarray:
        """
        从障碍物掩膜中排除手部区域（调试版本）
        
        Args:
            obstacle_mask: 原始障碍物掩膜
            hand_regions: 手部区域检测结果
            hand_landmarks_3d: 手部关键点3D坐标
            
        Returns:
            np.ndarray: 排除手部区域后的掩膜
        """
        result_mask = obstacle_mask.copy()
        original_pixels = np.sum(obstacle_mask > 0)
        
        # 排除YOLOv8检测到的手部区域
        for hand_region in hand_regions:
            hand_mask = hand_region['mask']
            # 膨胀手部区域以确保完全排除
            kernel = np.ones((15, 15), np.uint8)
            hand_mask_dilated = cv.dilate(hand_mask, kernel, iterations=1)
            result_mask = cv.bitwise_and(result_mask, cv.bitwise_not(hand_mask_dilated))
        
        # 排除MediaPipe检测到的手部关键点区域（减少膨胀，避免过度排除）
        if hand_landmarks_3d:
            hand_landmark_mask = self._create_hand_landmark_mask(
                hand_landmarks_3d, obstacle_mask.shape
            )
            result_mask = cv.bitwise_and(result_mask, cv.bitwise_not(hand_landmark_mask))
        
        final_pixels = np.sum(result_mask > 0)
        excluded_pixels = original_pixels - final_pixels
        
        # 调试信息
        if excluded_pixels > 0:
            print(f"🔍 手部区域排除: {excluded_pixels} 像素被排除")
        
        return result_mask
    
    def _create_hand_landmark_mask(self, 
                                  hand_landmarks_3d: List,
                                  mask_shape: Tuple[int, int]) -> np.ndarray:
        """
        基于手部关键点创建手部区域掩膜（减少膨胀，避免过度排除）
        
        Args:
            hand_landmarks_3d: 手部关键点3D坐标列表
            mask_shape: 掩膜形状
            
        Returns:
            np.ndarray: 手部区域掩膜
        """
        height, width = mask_shape
        hand_mask = np.zeros((height, width), dtype=np.uint8)
        
        for landmark in hand_landmarks_3d:
            if len(landmark) >= 2 and landmark[2] > 0:  # 有效深度
                x, y = int(landmark[0]), int(landmark[1])
                if 0 <= x < width and 0 <= y < height:
                    # 为每个关键点创建较小的膨胀区域
                    cv.circle(hand_mask, (x, y), 10, 255, -1)  # 减少半径从20到10
        
        # 减少膨胀以确保不会过度排除障碍物
        kernel = np.ones((10, 10), np.uint8)  # 减少核大小从25到10
        hand_mask = cv.dilate(hand_mask, kernel, iterations=1)
        
        return hand_mask
    
    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        掩膜后处理：噪声过滤和形态学操作
        
        Args:
            mask: 原始掩膜
            
        Returns:
            np.ndarray: 处理后的掩膜
        """
        # 移除小的噪声区域
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_small)
        
        # 填充小的空洞
        kernel_medium = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_medium)
        
        return mask
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """
        获取空的检测结果
        
        Returns:
            Dict: 空结果
        """
        return {
            'obstacles': [],
            'hand_regions': [],
            'obstacle_count': 0,
            'hand_region_count': 0,
            'total_detections': 0,
            'obstacle_mask': np.zeros((480, 640), dtype=np.uint8),
            'inference_time': 0.0,
            'fps': 0.0
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            Dict: 性能统计
        """
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'avg_fps': 0.0, 'max_inference_time': 0.0}
        
        avg_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        max_time = np.max(self.inference_times)
        
        return {
            'avg_inference_time': avg_time,
            'avg_fps': avg_fps,
            'max_inference_time': max_time,
            'total_inferences': len(self.inference_times)
        }
    
    def visualize_detection(self, 
                           image: np.ndarray, 
                           detection_result: Dict[str, Any]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detection_result: 检测结果
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_image = image.copy()
        
        # 绘制障碍物
        for obstacle in detection_result['obstacles']:
            bbox = obstacle['bbox']
            class_name = obstacle['class_name']
            confidence = obstacle['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv.putText(vis_image, label, (x1, y1 - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制手部区域
        for hand_region in detection_result['hand_regions']:
            bbox = hand_region['bbox']
            class_name = hand_region['class_name']
            confidence = hand_region['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv.putText(vis_image, label, (x1, y1 - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加统计信息
        stats_text = [
            f"Obstacles: {detection_result['obstacle_count']}",
            f"Hand Regions: {detection_result['hand_region_count']}",
            f"FPS: {detection_result['fps']:.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            cv.putText(vis_image, text, (10, 30 + i * 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image
    
    def cleanup(self):
        """
        清理资源
        """
        if self.model is not None:
            del self.model
            self.model = None
        self.is_initialized = False
        print("✅ YOLOv8障碍物检测器已清理")


def main():
    """
    测试YOLOv8障碍物检测器
    """
    print("🚀 测试YOLOv8障碍物检测器...")
    
    # 创建检测器
    detector = YOLOObstacleDetector(
        model_path="yolov8n-seg.pt",
        confidence_threshold=0.5
    )
    
    if not detector.is_initialized:
        print("❌ 检测器初始化失败")
        return
    
    # 测试图像（使用摄像头或测试图像）
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    try:
        frame_count = 0
        while frame_count < 100:  # 测试100帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测障碍物
            detection_result = detector.detect_obstacles(frame)
            
            # 可视化结果
            vis_frame = detector.visualize_detection(frame, detection_result)
            
            # 显示障碍物掩膜
            obstacle_mask = detection_result['obstacle_mask']
            mask_colored = cv.applyColorMap(obstacle_mask, cv.COLORMAP_JET)
            
            # 显示结果
            cv.imshow('YOLOv8 Obstacle Detection', vis_frame)
            cv.imshow('Obstacle Mask', mask_colored)
            
            frame_count += 1
            if frame_count % 10 == 0:
                stats = detector.get_performance_stats()
                print(f"帧 {frame_count}: {stats['avg_fps']:.1f} FPS")
            
            # 按'q'退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    
    finally:
        cap.release()
        cv.destroyAllWindows()
        detector.cleanup()
        print("✅ 测试完成")


if __name__ == "__main__":
    main()
