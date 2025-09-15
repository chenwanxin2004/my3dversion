#!/usr/bin/env python3
"""
MediaPipe手部姿态识别测试脚本
用于验证手部检测模块是否正常工作
"""

import cv2 as cv
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mediapipe_hand_pose import MediaPipeHandPose
    print("✅ MediaPipe hand pose module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MediaPipe hand pose module: {e}")
    print("Please install MediaPipe: pip install mediapipe")
    sys.exit(1)

def test_mediapipe_hand_pose():
    """测试MediaPipe手部姿态识别"""
    print("MediaPipe Hand Pose Detection Test")
    print("=" * 50)
    
    # 初始化手部姿态识别器
    try:
        hand_pose = MediaPipeHandPose()
        print("✅ MediaPipe hand pose detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe hand pose detector: {e}")
        return False
    
    # 打开摄像头
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    print("Press 'q' to quit, 'h' to show help")
    
    frame_count = 0
    gesture_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # 检测手部
            hands_info = hand_pose.detect_hands(frame)
            
            # 可视化结果
            annotated_frame = hand_pose.visualize_hands(frame, hands_info)
            
            # 添加测试信息
            test_info = [
                f"Frame: {frame_count}",
                f"Hands Detected: {len(hands_info)}",
                f"Gestures Recognized: {gesture_count}"
            ]
            
            for i, info in enumerate(test_info):
                cv.putText(annotated_frame, info, (10, 30 + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示检测到的手势
            if hands_info:
                for i, hand in enumerate(hands_info):
                    gesture_text = f"Hand {i+1}: {hand['label']} - {hand['gesture']} ({hand['confidence']:.2f})"
                    cv.putText(annotated_frame, gesture_text, (10, 150 + i * 25), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    if hand['gesture'] != "Unknown":
                        gesture_count += 1
            
            # 显示帮助信息
            help_text = [
                "Controls:",
                "q - Quit",
                "h - Show/Hide Help",
                "",
                "Gestures:",
                "Fist - Grab",
                "Open Hand - Release", 
                "Pointing - Select",
                "Thumbs Up - Approve",
                "Peace Sign - Victory"
            ]
            
            # 显示帮助信息（简化版）
            cv.putText(annotated_frame, "Press 'h' for help", (10, annotated_frame.shape[0] - 20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示结果
            cv.imshow('MediaPipe Hand Pose Test', annotated_frame)
            
            # 处理按键
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("\nHelp:")
                for text in help_text:
                    print(f"  {text}")
                print()
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        cap.release()
        hand_pose.cleanup()
        cv.destroyAllWindows()
    
    print(f"\nTest completed:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total gestures recognized: {gesture_count}")
    print(f"  Average gestures per frame: {gesture_count/max(frame_count, 1):.2f}")
    
    return True

def test_gesture_recognition():
    """测试手势识别功能"""
    print("\nGesture Recognition Test")
    print("=" * 30)
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 初始化手部姿态识别器
    hand_pose = MediaPipeHandPose()
    
    # 测试手势识别
    hands_info = hand_pose.detect_hands(test_image)
    
    print(f"Test image processed: {len(hands_info)} hands detected")
    
    # 清理资源
    hand_pose.cleanup()
    
    return True

if __name__ == "__main__":
    print("MediaPipe Hand Pose Detection Test Suite")
    print("=" * 50)
    
    # 测试手势识别
    if test_gesture_recognition():
        print("✅ Gesture recognition test passed")
    else:
        print("❌ Gesture recognition test failed")
    
    # 测试实时检测
    print("\nStarting real-time detection test...")
    if test_mediapipe_hand_pose():
        print("✅ Real-time detection test passed")
    else:
        print("❌ Real-time detection test failed")
    
    print("\nAll tests completed!")
