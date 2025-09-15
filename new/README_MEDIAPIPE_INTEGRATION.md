# MediaPipe手部姿态识别集成说明

## 概述

本项目已成功集成了MediaPipe手部姿态识别功能，可以与现有的3D越线检测系统协同工作，实现手势控制的机器人交互。

## 新增功能

### 1. 手部检测
- 实时检测图像中的手部关键点
- 支持最多2只手同时检测
- 识别左手/右手
- 计算手部3D姿态信息

### 2. 手势识别
支持以下手势：
- **握拳 (Fist)**: 握拳手势
- **张开手掌 (Open Hand)**: 张开手掌手势  
- **指向 (Pointing)**: 指向手势
- **竖拇指 (Thumbs Up)**: 竖拇指手势
- **胜利手势 (Peace Sign)**: 胜利手势
- **OK手势 (OK Sign)**: OK手势

### 3. 手势信息记录
识别并记录手势信息：
- 手势类型识别
- 左右手分类
- 置信度评估
- 实时显示和日志记录

## 文件结构

```
new/
├── mediapipe_hand_pose.py          # MediaPipe手部姿态识别模块
├── supervision_3d_crossing_detection.py  # 主程序（已集成手部检测）
├── test_mediapipe_hand_pose.py     # 测试脚本
└── README_MEDIAPIPE_INTEGRATION.md # 本说明文档
```

## 安装依赖

```bash
# 安装MediaPipe
pip install mediapipe

# 或使用uv
uv add mediapipe
```

## 使用方法

### 1. 运行主程序（集成版本）
```bash
cd new
python supervision_3d_crossing_detection.py
```

### 2. 运行测试脚本
```bash
cd new
python test_mediapipe_hand_pose.py
```

### 3. 单独使用MediaPipe模块
```python
from mediapipe_hand_pose import MediaPipeHandPose
import cv2

# 初始化
hand_pose = MediaPipeHandPose()

# 检测手部
hands_info = hand_pose.detect_hands(image)

# 可视化
annotated_image = hand_pose.visualize_hands(image, hands_info)

# 清理资源
hand_pose.cleanup()
```

## 功能特性

### 实时检测
- 30fps实时手部检测
- 低延迟手势识别
- 平滑的手势状态跟踪

### 高精度识别
- 21个手部关键点检测
- 基于角度计算的手势识别
- 置信度阈值过滤

### 可视化界面
- 手部关键点和连接线显示
- 手势标签和置信度显示
- 边界框和中心点标记
- 实时命令状态显示

## 技术实现

### 手部关键点检测
使用MediaPipe的Hands解决方案：
- 21个关键点：手腕、拇指(4点)、食指(4点)、中指(4点)、无名指(4点)、小指(4点)
- 实时3D坐标估计
- 左右手分类

### 手势识别算法
基于手指关节角度计算：
1. 计算各手指的弯曲状态
2. 根据弯曲模式识别手势
3. 使用历史状态平滑检测结果

### 3D姿态估计
- 计算手部中心点
- 估计手部方向向量
- 计算yaw角度（需要深度信息才能计算pitch和roll）

## 配置参数

### MediaPipe参数
```python
MediaPipeHandPose(
    static_image_mode=False,        # 实时模式
    max_num_hands=2,               # 最大检测手数
    min_detection_confidence=0.7,  # 最小检测置信度
    min_tracking_confidence=0.5    # 最小跟踪置信度
)
```

### 手势识别阈值
```python
GESTURE_THRESHOLDS = {
    'fist_threshold': 0.8,      # 握拳阈值
    'open_threshold': 0.3,      # 张开阈值
    'point_threshold': 0.7,     # 指向阈值
    'peace_threshold': 0.6,     # 胜利手势阈值
    'ok_threshold': 0.5         # OK手势阈值
}
```

## 性能优化

### 计算优化
- 限制处理的手部数量
- 使用历史状态平滑检测
- 置信度阈值过滤

### 内存优化
- 及时清理MediaPipe资源
- 限制历史状态长度
- 优化图像处理流程

## 故障排除

### 常见问题

1. **MediaPipe导入失败**
   ```
   ImportError: No module named 'mediapipe'
   ```
   解决：`pip install mediapipe`

2. **摄像头无法打开**
   ```
   Could not open camera
   ```
   解决：检查摄像头是否被其他程序占用

3. **手势识别不准确**
   - 确保手部在图像中清晰可见
   - 调整光照条件
   - 降低手势识别阈值

4. **性能问题**
   - 降低检测分辨率
   - 减少最大检测手数
   - 调整置信度阈值

### 调试模式
在测试脚本中启用详细输出：
```python
# 在test_mediapipe_hand_pose.py中
print(f"Hand {hand['hand_id']}: {hand['label']} - {hand['gesture']} (confidence: {hand['confidence']:.2f})")
```

## 扩展功能

### 自定义手势
可以扩展`_recognize_gesture`方法添加新的手势：
```python
def _is_custom_gesture(self, finger_states):
    # 自定义手势逻辑
    return custom_condition

def _recognize_gesture(self, landmarks):
    # 添加自定义手势检测
    if self._is_custom_gesture(finger_states):
        return "Custom Gesture"
    # ... 其他手势
```

### 3D深度集成
结合RealSense深度信息：
```python
def calculate_hand_depth(self, landmarks, depth_image):
    # 计算手部深度信息
    hand_center = self._calculate_hand_center(landmarks)
    depth = depth_image[hand_center[1], hand_center[0]]
    return depth
```

## 总结

MediaPipe手部姿态识别模块已成功集成到3D越线检测系统中，提供了：

1. **实时手部检测**：高精度21点关键点检测
2. **手势识别**：支持6种常用手势
3. **实时显示**：直观的检测结果显示和日志记录
4. **可视化界面**：手部关键点和手势标签显示
5. **模块化设计**：可独立使用或集成到其他系统

这个集成方案为手势识别提供了完整的解决方案，特别适合手势分析、人机交互等应用场景。

