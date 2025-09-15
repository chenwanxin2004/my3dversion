import numpy as np
import cv2 as cv
import time
import argparse
from collections import defaultdict, deque
import json
from datetime import datetime

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️ RealSense not available. Install with: pip install pyrealsense2")

class SimpleCrossingDetector:
    def __init__(self):
        """
        简化的越线检测系统（不依赖Supervision和YOLO）
        使用OpenCV的人体检测和3D点云
        """
        self.pipeline = None
        self.config = None
        self.camera_intrinsics = None
        self.depth_scale = None
        
        # OpenCV HOG person detector
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        
        # 3D point cloud parameters
        self.min_distance = 0.3
        self.max_distance = 5.0
        
        # Crossing detection parameters
        self.line_start = (200, 300)  # 越线起点 (x, y)
        self.line_end = (600, 300)    # 越线终点 (x, y)
        self.line_thickness = 3
        
        # Statistics
        self.crossing_events = []
        self.person_count = 0
        self.unique_persons = set()
        self.person_tracks = defaultdict(lambda: {'first_seen': None, 'crossed': False, 'positions': deque(maxlen=10)})
        
        # Initialize RealSense
        self.init_realsense()
        
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
    
    def detect_persons_opencv(self, color_image):
        """使用OpenCV HOG检测人物"""
        try:
            # Detect people
            (rects, weights) = self.hog.detectMultiScale(
                color_image, 
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
                hitThreshold=0.0,
                finalThreshold=2
            )
            
            # Filter detections by confidence
            person_detections = []
            for i, (x, y, w, h) in enumerate(rects):
                if weights[i] > 0.3:  # Confidence threshold
                    person_detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': weights[i],
                        'center': (x + w//2, y + h//2)
                    })
            
            return person_detections
            
        except Exception as e:
            print(f"Error in OpenCV person detection: {e}")
            return []
    
    def map_2d_to_3d(self, person_detections, depth_image):
        """将2D检测结果映射到3D坐标"""
        if self.camera_intrinsics is None or len(person_detections) == 0:
            return []
        
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        person_3d_info = []
        
        for i, detection in enumerate(person_detections):
            center_2d = detection['center']
            center_x, center_y = center_2d
            
            # Get depth at center point
            if (0 <= center_x < depth_image.shape[1] and 
                0 <= center_y < depth_image.shape[0]):
                
                depth = depth_image[center_y, center_x]
                
                if self.min_distance < depth < self.max_distance:
                    # Convert to 3D coordinates
                    x_3d = (center_x - cx) * depth / fx
                    y_3d = (center_y - cy) * depth / fy
                    z_3d = depth
                    
                    person_3d_info.append({
                        'id': i,
                        'bbox_2d': detection['bbox'],
                        'center_2d': center_2d,
                        'position_3d': (x_3d, y_3d, z_3d),
                        'depth': depth,
                        'confidence': detection['confidence']
                    })
        
        return person_3d_info
    
    def check_line_crossing(self, person_3d_info):
        """检查人物是否越线"""
        crossing_events = []
        
        for person in person_3d_info:
            person_id = person['id']
            center_2d = person['center_2d']
            
            # Update person track
            if person_id not in self.person_tracks:
                self.person_tracks[person_id]['first_seen'] = datetime.now()
            
            self.person_tracks[person_id]['positions'].append(center_2d)
            
            # Check if person crossed the line
            if not self.person_tracks[person_id]['crossed']:
                # Simple line crossing detection using point-to-line distance
                if self.is_point_crossing_line(center_2d, self.line_start, self.line_end):
                    self.person_tracks[person_id]['crossed'] = True
                    self.unique_persons.add(person_id)
                    
                    crossing_event = {
                        'person_id': person_id,
                        'timestamp': datetime.now(),
                        'position_3d': person['position_3d'],
                        'position_2d': center_2d,
                        'depth': person['depth']
                    }
                    
                    crossing_events.append(crossing_event)
                    self.crossing_events.append(crossing_event)
        
        return crossing_events
    
    def is_point_crossing_line(self, point, line_start, line_end):
        """检查点是否越过了线"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate distance from point to line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)
        
        # Check if point is close to line (within threshold)
        threshold = 20  # pixels
        return distance < threshold
    
    def create_visualization(self, color_image, person_3d_info, crossing_events):
        """创建可视化界面"""
        annotated_image = color_image.copy()
        
        # Draw detection line
        cv.line(annotated_image, self.line_start, self.line_end, (0, 0, 255), self.line_thickness)
        cv.putText(annotated_image, "DETECTION LINE", 
                  (self.line_start[0], self.line_start[1] - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw person detections
        for person in person_3d_info:
            bbox = person['bbox_2d']
            center_2d = person['center_2d']
            position_3d = person['position_3d']
            person_id = person['id']
            
            # Draw bounding box
            x, y, w, h = bbox
            cv.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add person ID and 3D coordinates
            text = f"ID:{person_id} ({position_3d[0]:.1f},{position_3d[1]:.1f},{position_3d[2]:.1f})"
            cv.putText(annotated_image, text, 
                      (x, y - 10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw center point
            cv.circle(annotated_image, center_2d, 5, (0, 255, 0), -1)
        
        # Add statistics overlay
        stats_text = [
            f"Total Persons: {len(self.unique_persons)}",
            f"Crossing Events: {len(self.crossing_events)}",
            f"Current Frame: {len(person_3d_info)} persons"
        ]
        
        for i, text in enumerate(stats_text):
            cv.putText(annotated_image, text, (10, 30 + i * 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_image
    
    def create_statistics_panel(self):
        """创建统计信息面板"""
        panel = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Title
        cv.putText(panel, "SIMPLE CROSSING DETECTION STATISTICS", (50, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Statistics
        y_pos = 80
        stats = [
            f"Total Unique Persons: {len(self.unique_persons)}",
            f"Total Crossing Events: {len(self.crossing_events)}",
            f"Active Tracks: {len(self.person_tracks)}",
            "",
            "Recent Crossing Events:"
        ]
        
        for stat in stats:
            if stat == "":
                y_pos += 10
                continue
            cv.putText(panel, stat, (50, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
        
        # Show recent crossing events
        recent_events = self.crossing_events[-5:]  # Last 5 events
        for event in recent_events:
            timestamp = event['timestamp'].strftime("%H:%M:%S")
            text = f"  ID {event['person_id']} at {timestamp}"
            cv.putText(panel, text, (70, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
        
        return panel
    
    def save_statistics(self, filename="simple_crossing_statistics.json"):
        """保存统计信息到文件"""
        stats = {
            'total_unique_persons': len(self.unique_persons),
            'total_crossing_events': len(self.crossing_events),
            'crossing_events': [
                {
                    'person_id': event['person_id'],
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
        cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Simple Crossing Detection with OpenCV + 3D Point Cloud')
    parser.add_argument('--save-stats', action='store_true', help='Save statistics to file')
    
    args = parser.parse_args()
    
    print("Simple Crossing Detection System")
    print("=" * 50)
    print("Features:")
    print("- OpenCV HOG person detection")
    print("- 3D position mapping with RealSense")
    print("- Line crossing detection")
    print("- Person counting and tracking")
    print("Press 'q' or ESC to quit, 's' to save statistics")
    
    # Initialize detector
    detector = SimpleCrossingDetector()
    
    if detector.pipeline is None:
        print("❌ Failed to initialize RealSense camera")
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
            
            # Detect persons using OpenCV
            person_detections = detector.detect_persons_opencv(color_image)
            
            # Map 2D detections to 3D coordinates
            person_3d_info = detector.map_2d_to_3d(person_detections, depth_image)
            
            # Check for line crossing
            crossing_events = detector.check_line_crossing(person_3d_info)
            
            # Create visualizations
            annotated_image = detector.create_visualization(color_image, person_3d_info, crossing_events)
            stats_panel = detector.create_statistics_panel()
            
            # Display results
            cv.imshow('Simple Crossing Detection', annotated_image)
            cv.imshow('Statistics', stats_panel)
            
            # Print crossing events
            if crossing_events:
                for event in crossing_events:
                    print(f"🚨 Person {event['person_id']} crossed the line at {event['timestamp'].strftime('%H:%M:%S')}")
            
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
        print("Simple crossing detection stopped.")

if __name__ == "__main__":
    main()