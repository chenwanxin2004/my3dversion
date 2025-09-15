import numpy as np
import cv2 as cv
import time
import argparse

class RealSense3DPointCloud:
    def __init__(self):
        """
        Intel RealSense 3D Point Cloud Generator
        """
        self.pipeline = None
        self.config = None
        self.camera_intrinsics = None
        
        # Initialize RealSense
        self.init_realsense()
        
        # Point cloud processing
        self.point_cloud_history = []
        self.max_history = 10
        
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
            
            print(f"✅ RealSense camera initialized successfully")
            print(f"Camera intrinsics:")
            print(f"  fx: {depth_intrinsics.fx:.2f}, fy: {depth_intrinsics.fy:.2f}")
            print(f"  cx: {depth_intrinsics.ppx:.2f}, cy: {depth_intrinsics.ppy:.2f}")
            print(f"  Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth scale: {self.depth_scale}")
            
        except ImportError:
            print("❌ pyrealsense2 library not found")
            print("Please install it with: pip install pyrealsense2")
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
    
    def depth_to_pointcloud(self, depth_image, color_image=None, max_depth=5.0):
        """Convert depth image to 3D point cloud"""
        if self.camera_intrinsics is None:
            return None, None
        
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        z = depth_image.astype(np.float32)
        
        # Filter out invalid depths
        valid_mask = (z > 0.1) & (z < max_depth)
        
        # Calculate 3D coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack coordinates and apply mask
        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        
        # Extract colors if color image provided
        colors = None
        if color_image is not None:
            colors = color_image[valid_mask]
        
        return points, colors
    
    def downsample_pointcloud(self, points, colors=None, max_points=20000):
        """Downsample point cloud for real-time performance"""
        if len(points) <= max_points:
            return points, colors
        
        # Random sampling
        indices = np.random.choice(len(points), max_points, replace=False)
        sampled_points = points[indices]
        sampled_colors = colors[indices] if colors is not None else None
        
        return sampled_points, sampled_colors
    
    def visualize_pointcloud_2d(self, points, colors=None):
        """Create 2D visualization of 3D point cloud"""
        if len(points) == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get point cloud bounds for better scaling
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # Project points to 2D (top-down view) with better scaling
        x_range = x_max - x_min if x_max > x_min else 1.0
        z_range = z_max - z_min if z_max > z_min else 1.0
        
        # Scale to fit image
        x_proj = ((points[:, 0] - x_min) / x_range * 600 + 20).astype(int)
        y_proj = ((points[:, 2] - z_min) / z_range * 400 + 40).astype(int)
        
        # Filter valid projections
        valid = (x_proj >= 0) & (x_proj < 640) & (y_proj >= 0) & (y_proj < 480)
        x_proj = x_proj[valid]
        y_proj = y_proj[valid]
        points_valid = points[valid]
        
        if len(x_proj) > 0:
            if colors is not None:
                colors_valid = colors[valid]
                for i, (x, y) in enumerate(zip(x_proj, y_proj)):
                    if i < len(colors_valid):
                        cv.circle(vis_img, (x, y), 1, colors_valid[i].tolist(), -1)
            else:
                # Color by depth (Z coordinate)
                depth_colors = points_valid[:, 2]
                if depth_colors.max() > depth_colors.min():
                    depth_colors = (depth_colors - depth_colors.min()) / (depth_colors.max() - depth_colors.min())
                else:
                    depth_colors = np.ones_like(depth_colors) * 0.5
                
                for i, (x, y, depth) in enumerate(zip(x_proj, y_proj, depth_colors)):
                    if not np.isnan(depth) and not np.isinf(depth):
                        # Use HSV color space for better depth visualization
                        hue = int(120 * (1 - depth))  # Green to red
                        color = cv.cvtColor(np.uint8([[[hue, 255, 255]]]), cv.COLOR_HSV2BGR)[0][0]
                        cv.circle(vis_img, (x, y), 1, color.tolist(), -1)
        
        # Add coordinate axes
        cv.line(vis_img, (20, 40), (620, 40), (255, 255, 255), 1)  # X axis
        cv.line(vis_img, (20, 40), (20, 440), (255, 255, 255), 1)  # Z axis
        
        # Add labels
        cv.putText(vis_img, "X", (620, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(vis_img, "Z", (25, 440), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def visualize_pointcloud_side_view(self, points, colors=None):
        """Create side view visualization of 3D point cloud"""
        if len(points) == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get point cloud bounds for better scaling
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # Project points to 2D (side view: Y vs Z)
        y_range = y_max - y_min if y_max > y_min else 1.0
        z_range = z_max - z_min if z_max > z_min else 1.0
        
        # Scale to fit image
        x_proj = ((points[:, 1] - y_min) / y_range * 600 + 20).astype(int)
        y_proj = ((points[:, 2] - z_min) / z_range * 400 + 40).astype(int)
        
        # Filter valid projections
        valid = (x_proj >= 0) & (x_proj < 640) & (y_proj >= 0) & (y_proj < 480)
        x_proj = x_proj[valid]
        y_proj = y_proj[valid]
        points_valid = points[valid]
        
        if len(x_proj) > 0:
            if colors is not None:
                colors_valid = colors[valid]
                for i, (x, y) in enumerate(zip(x_proj, y_proj)):
                    if i < len(colors_valid):
                        cv.circle(vis_img, (x, y), 1, colors_valid[i].tolist(), -1)
            else:
                # Color by depth (Z coordinate)
                depth_colors = points_valid[:, 2]
                if depth_colors.max() > depth_colors.min():
                    depth_colors = (depth_colors - depth_colors.min()) / (depth_colors.max() - depth_colors.min())
                else:
                    depth_colors = np.ones_like(depth_colors) * 0.5
                
                for i, (x, y, depth) in enumerate(zip(x_proj, y_proj, depth_colors)):
                    if not np.isnan(depth) and not np.isinf(depth):
                        # Use HSV color space for better depth visualization
                        hue = int(120 * (1 - depth))  # Green to red
                        color = cv.cvtColor(np.uint8([[[hue, 255, 255]]]), cv.COLOR_HSV2BGR)[0][0]
                        cv.circle(vis_img, (x, y), 1, color.tolist(), -1)
        
        # Add coordinate axes
        cv.line(vis_img, (20, 40), (620, 40), (255, 255, 255), 1)  # Y axis
        cv.line(vis_img, (20, 40), (20, 440), (255, 255, 255), 1)  # Z axis
        
        # Add labels
        cv.putText(vis_img, "Y", (620, 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(vis_img, "Z", (25, 440), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def save_pointcloud_ply(self, points, colors, filename):
        """Save point cloud in PLY format (with colors)"""
        if colors is None:
            colors = np.ones((len(points), 3), dtype=np.uint8) * 128
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i, point in enumerate(points):
                if i < len(colors):
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {colors[i][2]} {colors[i][1]} {colors[i][0]}\n")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.stop()
        cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Intel RealSense 3D Point Cloud Generator')
    parser.add_argument('--max-depth', type=float, default=5.0, help='Maximum depth in meters')
    parser.add_argument('--max-points', type=int, default=20000, help='Maximum points to display')
    
    args = parser.parse_args()
    
    print("Intel RealSense 3D Point Cloud Generator")
    print("=" * 50)
    print(f"Max depth: {args.max_depth}m")
    print(f"Max points: {args.max_points}")
    print("Press 'q' to quit, 's' to save point cloud, 'p' to save PLY")
    
    # Initialize RealSense
    realsense = RealSense3DPointCloud()
    
    if realsense.pipeline is None:
        print("Failed to initialize RealSense camera")
        return
    
    frame_count = 0
    save_interval = 60
    
    try:
        while True:
            # Get frames from RealSense
            depth_image, color_image = realsense.get_frames()
            
            if depth_image is None or color_image is None:
                print("Failed to get frames from RealSense")
                time.sleep(0.1)
                continue
            
            # Convert to point cloud
            points, colors = realsense.depth_to_pointcloud(depth_image, color_image, args.max_depth)
            
            if points is None or len(points) == 0:
                print("No valid points in point cloud")
                continue
            
            # Downsample for performance
            points, colors = realsense.downsample_pointcloud(points, colors, args.max_points)
            
            # Create depth visualization
            depth_vis = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
            depth_vis = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)
            
            # Visualize point cloud
            pointcloud_top = realsense.visualize_pointcloud_2d(points, colors)
            pointcloud_side = realsense.visualize_pointcloud_side_view(points, colors)
            
            # Display results
            cv.imshow('RealSense Color', color_image)
            cv.imshow('RealSense Depth', depth_vis)
            cv.imshow('3D Point Cloud (Top-down)', pointcloud_top)
            cv.imshow('3D Point Cloud (Side View)', pointcloud_side)
            
            # Add text information to both views
            cv.putText(pointcloud_top, f"Points: {len(points)}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(pointcloud_top, f"Depth: {depth_image.mean():.2f}m", (10, 70), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(pointcloud_top, f"Frame: {frame_count}", (10, 110), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv.putText(pointcloud_side, f"Points: {len(points)}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(pointcloud_side, f"Depth: {depth_image.mean():.2f}m", (10, 70), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save point cloud periodically
            frame_count += 1
            if frame_count % save_interval == 0:
                # Save as XYZ
                filename_xyz = f'realsense_pointcloud_{frame_count}.xyz'
                if colors is not None:
                    with open(filename_xyz, 'w') as f:
                        for i, point in enumerate(points):
                            if i < len(colors):
                                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
                else:
                    np.savetxt(filename_xyz, points, fmt='%.6f', delimiter=' ')
                
                # Save as PLY
                filename_ply = f'realsense_pointcloud_{frame_count}.ply'
                realsense.save_pointcloud_ply(points, colors, filename_ply)
                
                print(f"Saved point cloud: {filename_xyz} and {filename_ply} ({len(points)} points)")
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save XYZ
                filename_xyz = f'manual_save_{int(time.time())}.xyz'
                if colors is not None:
                    with open(filename_xyz, 'w') as f:
                        for i, point in enumerate(points):
                            if i < len(colors):
                                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
                else:
                    np.savetxt(filename_xyz, points, fmt='%.6f', delimiter=' ')
                print(f"Manually saved XYZ: {filename_xyz}")
            elif key == ord('p'):
                # Save PLY
                filename_ply = f'manual_save_{int(time.time())}.ply'
                realsense.save_pointcloud_ply(points, colors, filename_ply)
                print(f"Manually saved PLY: {filename_ply}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        realsense.cleanup()
        print("RealSense 3D point cloud generation stopped.")

if __name__ == "__main__":
    main()
