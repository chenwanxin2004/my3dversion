import numpy as np
import cv2 as cv
import time
import argparse

class PointCloudAnalyzer:
    def __init__(self):
        """
        Analyze and understand point cloud patterns
        """
        self.pipeline = None
        self.config = None
        self.camera_intrinsics = None
        
        # Initialize RealSense
        self.init_realsense()
        
    def init_realsense(self):
        """Initialize Intel RealSense camera"""
        try:
            import pyrealsense2 as rs
            print("✅ pyrealsense2 library found")
            
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
            
        except ImportError:
            print("❌ pyrealsense2 library not found")
            self.pipeline = None
        except Exception as e:
            print(f"❌ Failed to initialize RealSense: {e}")
            self.pipeline = None
    
    def get_frames(self):
        """Get depth and color frames from RealSense"""
        if self.pipeline is None:
            return None, None
        
        try:
            import pyrealsense2 as rs
            
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
    
    def analyze_pointcloud_patterns(self, depth_image, color_image):
        """Analyze why point clouds form triangular patterns"""
        if self.camera_intrinsics is None:
            return None, None, None
        
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        z = depth_image.astype(np.float32)
        
        # Filter valid depths
        valid_mask = (z > 0.3) & (z < 3.0)
        
        # Calculate 3D coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack coordinates and apply mask
        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        colors = color_image[valid_mask]
        
        if len(points) == 0:
            return None, None, None
        
        # Analyze patterns
        analysis = self.explain_triangular_patterns(points, depth_image, valid_mask)
        
        return points, colors, analysis
    
    def explain_triangular_patterns(self, points, depth_image, valid_mask):
        """Explain why triangular patterns appear"""
        analysis = {}
        
        # 1. Camera perspective projection
        analysis['perspective'] = "相机透视投影：远处的物体看起来更小，形成锥形视野"
        
        # 2. Depth distribution
        valid_depths = depth_image[valid_mask]
        depth_std = np.std(valid_depths)
        analysis['depth_variation'] = f"深度变化：标准差 {depth_std:.3f}m，变化越大三角形越明显"
        
        # 3. Point density
        total_pixels = depth_image.size
        valid_pixels = np.sum(valid_mask)
        density = valid_pixels / total_pixels * 100
        analysis['density'] = f"点云密度：{density:.1f}% 的像素有有效深度"
        
        # 4. Distance to camera
        avg_distance = np.mean(valid_depths)
        analysis['distance'] = f"平均距离：{avg_distance:.2f}m"
        
        # 5. Field of view effect
        if len(points) > 0:
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()
            analysis['fov'] = f"视野范围：X={x_range:.2f}m, Y={y_range:.2f}m"
        
        return analysis
    
    def create_educational_visualization(self, points, colors, analysis):
        """Create educational visualization showing why triangular patterns form"""
        if len(points) == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create top-down view
        x_coords = points[:, 0]
        y_coords = points[:, 2]  # Use Z as Y in 2D
        
        # Get bounds
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Scale to image coordinates
        x_proj = ((x_coords - x_min) / (x_max - x_min) * 580 + 30).astype(int)
        y_proj = ((y_coords - y_min) / (y_max - y_min) * 400 + 40).astype(int)
        
        # Filter valid projections
        valid = (x_proj >= 0) & (x_proj < 640) & (y_proj >= 0) & (y_proj < 480)
        x_proj = x_proj[valid]
        y_proj = y_proj[valid]
        points_valid = points[valid]
        colors_valid = colors[valid]
        
        # Draw camera field of view (triangular pattern)
        camera_pos = (320, 40)  # Camera position at top center
        cv.circle(vis_img, camera_pos, 5, (0, 255, 255), -1)  # Yellow camera
        
        # Draw field of view lines
        fov_angle = 60  # degrees
        left_line = (int(320 - 200 * np.tan(np.radians(fov_angle/2))), 440)
        right_line = (int(320 + 200 * np.tan(np.radians(fov_angle/2))), 440)
        
        cv.line(vis_img, camera_pos, left_line, (0, 255, 255), 2)
        cv.line(vis_img, camera_pos, right_line, (0, 255, 255), 2)
        cv.line(vis_img, left_line, right_line, (0, 255, 255), 2)
        
        # Draw points with colors
        for i, (x, y) in enumerate(zip(x_proj, y_proj)):
            if i < len(colors_valid):
                color = colors_valid[i]
                cv.circle(vis_img, (x, y), 2, color.tolist(), -1)
        
        # Add explanatory text
        cv.putText(vis_img, "Camera Field of View", (10, 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(vis_img, "Triangular pattern due to:", (10, 450), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(vis_img, "1. Perspective projection", (10, 470), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_img
    
    def create_analysis_display(self, analysis):
        """Create analysis information display"""
        if not analysis:
            return np.zeros((300, 640, 3), dtype=np.uint8)
        
        display = np.zeros((300, 640, 3), dtype=np.uint8)
        
        y_pos = 30
        for key, value in analysis.items():
            if isinstance(value, str):
                cv.putText(display, value, (10, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
        
        return display
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.stop()
        cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Understand Point Cloud Patterns')
    
    args = parser.parse_args()
    
    print("Point Cloud Pattern Analysis")
    print("=" * 40)
    print("Understanding why point clouds form triangular patterns")
    print("Press 'q' to quit")
    
    # Initialize analyzer
    analyzer = PointCloudAnalyzer()
    
    if analyzer.pipeline is None:
        print("Failed to initialize RealSense camera")
        return
    
    try:
        while True:
            # Get frames from RealSense
            depth_image, color_image = analyzer.get_frames()
            
            if depth_image is None or color_image is None:
                print("Failed to get frames from RealSense")
                time.sleep(0.1)
                continue
            
            # Analyze point cloud patterns
            points, colors, analysis = analyzer.analyze_pointcloud_patterns(depth_image, color_image)
            
            if points is None or len(points) == 0:
                print("No valid points in point cloud")
                continue
            
            # Create visualizations
            depth_vis = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
            depth_vis = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)
            
            educational_vis = analyzer.create_educational_visualization(points, colors, analysis)
            analysis_display = analyzer.create_analysis_display(analysis)
            
            # Display results
            cv.imshow('RealSense Color', color_image)
            cv.imshow('RealSense Depth', depth_vis)
            cv.imshow('Why Triangular Pattern?', educational_vis)
            cv.imshow('Analysis', analysis_display)
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        analyzer.cleanup()
        print("Analysis stopped.")

if __name__ == "__main__":
    main()



