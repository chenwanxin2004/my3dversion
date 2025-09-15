import numpy as np
import cv2 as cv
import time
from collections import deque

class Realtime3DGenerator:
    def __init__(self, camera_intrinsics=None, max_depth=5.0, point_cloud_size=10000):
        """
        Real-time 3D point cloud generator from infrared depth camera
        
        Args:
            camera_intrinsics: [fx, fy, cx, cy] - if None, will use default values
            max_depth: Maximum depth to consider (meters)
            point_cloud_size: Maximum number of points to display
        """
        self.max_depth = max_depth
        self.point_cloud_size = point_cloud_size
        
        # Default camera intrinsics (typical for Intel RealSense, Azure Kinect, etc.)
        if camera_intrinsics is None:
            self.fx, self.fy = 525.0, 525.0
            self.cx, self.cy = 320.0, 240.0
        else:
            self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
        # Point cloud storage
        self.current_points = None
        self.point_cloud_history = deque(maxlen=10)  # Keep last 10 frames
        
    def depth_to_pointcloud(self, depth_image, color_image=None):
        """
        Convert depth image to 3D point cloud in real-time
        
        Args:
            depth_image: Depth image from infrared camera (in meters)
            color_image: Optional color image for colored point cloud
        
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (if color_image provided)
        """
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        z = depth_image.astype(np.float32)
        
        # Filter out invalid depths
        valid_mask = (z > 0) & (z < self.max_depth)
        
        # Calculate 3D coordinates
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Stack coordinates and apply mask
        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        
        # Extract colors if color image provided
        colors = None
        if color_image is not None:
            colors = color_image[valid_mask]
        
        return points, colors, valid_mask
    
    def downsample_pointcloud(self, points, colors=None, target_size=None):
        """
        Downsample point cloud for real-time visualization
        """
        if target_size is None:
            target_size = self.point_cloud_size
            
        if len(points) <= target_size:
            return points, colors
        
        # Random sampling for real-time performance
        indices = np.random.choice(len(points), target_size, replace=False)
        sampled_points = points[indices]
        sampled_colors = colors[indices] if colors is not None else None
        
        return sampled_points, sampled_colors
    
    def update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_counter.append(fps)
        self.last_time = current_time
        return np.mean(self.fps_counter)
    
    def process_frame(self, depth_image, color_image=None):
        """
        Process a single frame and generate 3D point cloud
        
        Args:
            depth_image: Depth image from camera
            color_image: Optional color image
        
        Returns:
            points: 3D point cloud
            colors: Point colors (if available)
            fps: Current FPS
        """
        # Convert depth to point cloud
        points, colors, valid_mask = self.depth_to_pointcloud(depth_image, color_image)
        
        # Downsample for real-time performance
        points, colors = self.downsample_pointcloud(points, colors)
        
        # Store current point cloud
        self.current_points = points
        self.point_cloud_history.append(points.copy())
        
        # Update FPS
        fps = self.update_fps()
        
        return points, colors, fps
    
    def get_point_cloud_statistics(self):
        """Get statistics about current point cloud"""
        if self.current_points is None:
            return None
        
        stats = {
            'num_points': len(self.current_points),
            'depth_range': (self.current_points[:, 2].min(), self.current_points[:, 2].max()),
            'x_range': (self.current_points[:, 0].min(), self.current_points[:, 0].max()),
            'y_range': (self.current_points[:, 1].min(), self.current_points[:, 1].max()),
            'mean_depth': np.mean(self.current_points[:, 2])
        }
        return stats

def simulate_camera_feed():
    """
    Simulate camera feed for demonstration
    In real application, replace with actual camera capture
    """
    # Simulate camera parameters
    width, height = 640, 480
    
    # Create synthetic depth data (simulating moving objects)
    depth_image = np.zeros((height, width), dtype=np.float32)
    
    # Add moving objects
    t = time.time()
    
    # Floor
    depth_image[300:, :] = 2.0 + 0.1 * np.sin(t)
    
    # Moving box
    box_x = int(200 + 100 * np.sin(t * 0.5))
    box_y = int(150 + 50 * np.cos(t * 0.3))
    depth_image[box_y:box_y+80, box_x:box_x+80] = 1.0 + 0.2 * np.sin(t * 2)
    
    # Rotating object
    center_x, center_y = 400, 200
    radius = 60
    obj_x = int(center_x + radius * np.cos(t))
    obj_y = int(center_y + radius * np.sin(t))
    depth_image[obj_y-20:obj_y+20, obj_x-20:obj_x+20] = 0.8
    
    # Add noise
    noise = np.random.normal(0, 0.01, depth_image.shape)
    depth_image += noise
    
    # Add invalid regions
    depth_image[50:100, 300:400] = 0
    
    # Create color image
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:, :, 0] = 100  # Red channel
    color_image[:, :, 1] = 150  # Green channel
    color_image[:, :, 2] = 200  # Blue channel
    
    return depth_image, color_image

def visualize_realtime_pointcloud(points, colors=None, fps=0, stats=None):
    """
    Simple 2D visualization of 3D point cloud for real-time display
    """
    if points is None or len(points) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create visualization image
    vis_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Project 3D points to 2D (top-down view)
    x_proj = ((points[:, 0] + 2) * 160).astype(int)  # Scale and offset
    y_proj = ((points[:, 2] * 100)).astype(int)      # Use depth as Y
    
    # Filter valid projections
    valid = (x_proj >= 0) & (x_proj < 640) & (y_proj >= 0) & (y_proj < 480)
    x_proj = x_proj[valid]
    y_proj = y_proj[valid]
    
    if len(x_proj) > 0:
        # Color by depth
        depth_colors = points[valid, 2]
        depth_colors = (depth_colors - depth_colors.min()) / (depth_colors.max() - depth_colors.min())
        
        for i, (x, y, depth) in enumerate(zip(x_proj, y_proj, depth_colors)):
            color = (int(255 * depth), int(255 * (1-depth)), 100)
            cv.circle(vis_img, (x, y), 2, color, -1)
    
    # Add text information
    cv.putText(vis_img, f"FPS: {fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(vis_img, f"Points: {len(points)}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if stats:
        cv.putText(vis_img, f"Depth: {stats['mean_depth']:.2f}m", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis_img

def main():
    print("Real-time 3D Point Cloud Generation from Infrared Depth Camera")
    print("=" * 60)
    print("Press 'q' to quit, 's' to save current point cloud")
    
    # Initialize 3D generator
    generator = Realtime3DGenerator(max_depth=5.0, point_cloud_size=5000)
    
    # For real camera, use:
    # cap = cv.VideoCapture(0)  # or specific camera index
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    save_interval = 30  # Save every 30 frames
    
    try:
        while True:
            # Simulate camera capture (replace with actual camera)
            depth_image, color_image = simulate_camera_feed()
            
            # Process frame
            points, colors, fps = generator.process_frame(depth_image, color_image)
            stats = generator.get_point_cloud_statistics()
            
            # Create visualization
            vis_img = visualize_realtime_pointcloud(points, colors, fps, stats)
            
            # Display results
            cv.imshow('Real-time 3D Point Cloud (Top-down View)', vis_img)
            cv.imshow('Depth Image', (depth_image * 50).astype(np.uint8))  # Scale for display
            cv.imshow('Color Image', color_image)
            
            # Save point cloud periodically
            frame_count += 1
            if frame_count % save_interval == 0:
                filename = f'realtime_pointcloud_{frame_count}.xyz'
                np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
                print(f"Saved point cloud: {filename} ({len(points)} points)")
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'manual_save_{int(time.time())}.xyz'
                np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
                print(f"Manually saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cv.destroyAllWindows()
        print("Real-time 3D generation stopped.")

if __name__ == "__main__":
    main()
