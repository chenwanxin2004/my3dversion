#!/usr/bin/env python3
"""
摄像头测试脚本
用于诊断和解决摄像头访问问题
"""

import cv2 as cv
import numpy as np
import time

def test_camera_access():
    """测试摄像头访问"""
    print("Camera Access Test")
    print("=" * 30)
    
    # 测试不同的摄像头索引
    camera_indices = [0, 1, 2]
    backends = [cv.CAP_ANY, cv.CAP_DSHOW, cv.CAP_MSMF]
    
    working_cameras = []
    
    for camera_index in camera_indices:
        print(f"\nTesting camera index {camera_index}...")
        
        for backend in backends:
            backend_name = {
                cv.CAP_ANY: "ANY",
                cv.CAP_DSHOW: "DSHOW", 
                cv.CAP_MSMF: "MSMF"
            }.get(backend, "UNKNOWN")
            
            print(f"  Trying backend {backend_name}...")
            
            try:
                cap = cv.VideoCapture(camera_index, backend)
                
                if cap.isOpened():
                    print(f"    ✅ Camera {camera_index} opened with {backend_name}")
                    
                    # 尝试读取一帧
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"    ✅ Successfully read frame: {frame.shape}")
                        working_cameras.append((camera_index, backend, backend_name))
                        
                        # 显示一帧图像
                        cv.imshow(f'Camera {camera_index} - {backend_name}', frame)
                        cv.waitKey(1000)  # 显示1秒
                        cv.destroyAllWindows()
                        
                        cap.release()
                        break
                    else:
                        print(f"    ❌ Failed to read frame")
                        cap.release()
                else:
                    print(f"    ❌ Failed to open camera")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                if 'cap' in locals():
                    cap.release()
    
    return working_cameras

def test_mediapipe_with_working_camera(camera_index, backend):
    """使用工作的摄像头测试MediaPipe"""
    print(f"\nTesting MediaPipe with camera {camera_index}...")
    
    try:
        from mediapipe_hand_pose import MediaPipeHandPose
        
        # 初始化MediaPipe
        hand_pose = MediaPipeHandPose()
        print("✅ MediaPipe initialized")
        
        # 打开摄像头
        cap = cv.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            print("❌ Failed to open camera for MediaPipe test")
            return False
        
        print("✅ Camera opened for MediaPipe test")
        print("Press 'q' to quit")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # 检测手部
            hands_info = hand_pose.detect_hands(frame)
            
            # 可视化
            annotated_frame = hand_pose.visualize_hands(frame, hands_info)
            
            # 添加帧计数
            cv.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(annotated_frame, f"Hands: {len(hands_info)}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示结果
            cv.imshow('MediaPipe Hand Detection Test', annotated_frame)
            
            # 打印检测结果
            if hands_info:
                for hand in hands_info:
                    print(f"Hand {hand['hand_id']}: {hand['label']} - {hand['gesture']} (confidence: {hand['confidence']:.2f})")
            
            # 检查退出
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理资源
        cap.release()
        hand_pose.cleanup()
        cv.destroyAllWindows()
        
        print(f"✅ MediaPipe test completed. Processed {frame_count} frames.")
        return True
        
    except ImportError:
        print("❌ MediaPipe not available. Please install: pip install mediapipe")
        return False
    except Exception as e:
        print(f"❌ Error in MediaPipe test: {e}")
        return False

def main():
    """主函数"""
    print("Camera and MediaPipe Test Suite")
    print("=" * 40)
    
    # 测试摄像头访问
    working_cameras = test_camera_access()
    
    if not working_cameras:
        print("\n❌ No working cameras found!")
        print("Possible solutions:")
        print("1. Check if camera is connected")
        print("2. Close other applications using the camera")
        print("3. Try running as administrator")
        print("4. Check camera drivers")
        return
    
    print(f"\n✅ Found {len(working_cameras)} working camera(s):")
    for camera_index, backend, backend_name in working_cameras:
        print(f"  Camera {camera_index} with {backend_name} backend")
    
    # 使用第一个工作的摄像头测试MediaPipe
    if working_cameras:
        camera_index, backend, backend_name = working_cameras[0]
        print(f"\nUsing camera {camera_index} with {backend_name} for MediaPipe test...")
        test_mediapipe_with_working_camera(camera_index, backend)

if __name__ == "__main__":
    main()

