import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# 设置matplotlib后端
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端

f, cx, cy = 1000., 320., 240.
pts0 = np.loadtxt('data/image_formation0.xyz')[:,:2]
pts1 = np.loadtxt('data/image_formation1.xyz')[:,:2]
output_file = 'triangulation.xyz'

# Estimate relative pose of two view
F, _ = cv.findFundamentalMat(pts0, pts1, cv.FM_8POINT)
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
E = K.T @ F @ K
_, R, t, _ = cv.recoverPose(E, pts0, pts1)

# Reconstruct 3D points (triangulation)
P0 = K @ np.eye(3, 4, dtype=np.float32)
Rt = np.hstack((R, t))
P1 = K @ Rt
X = cv.triangulatePoints(P0, P1, pts0.T, pts1.T)
X /= X[3]
X = X.T

# Write the reconstructed 3D points
np.savetxt(output_file, X)

# 打印重建结果信息
print(f"重建了 {len(X)} 个3D点")
print(f"3D点范围:")
print(f"  X: {X[:,0].min():.3f} 到 {X[:,0].max():.3f}")
print(f"  Y: {X[:,1].min():.3f} 到 {X[:,1].max():.3f}")
print(f"  Z: {X[:,2].min():.3f} 到 {X[:,2].max():.3f}")

# Visualize the reconstructed 3D points
print("显示3D点云...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D点
ax.scatter(X[:,0], X[:,1], X[:,2], c='red', s=20, alpha=0.6)

# 设置坐标轴
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Point Cloud Reconstruction')

# 设置相等的坐标轴比例
max_range = np.array([X[:,0].max()-X[:,0].min(), X[:,1].max()-X[:,1].min(), X[:,2].max()-X[:,2].min()]).max() / 2.0
mid_x = (X[:,0].max()+X[:,0].min()) * 0.5
mid_y = (X[:,1].max()+X[:,1].min()) * 0.5
mid_z = (X[:,2].max()+X[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.grid(True)
plt.tight_layout()

print("按任意键关闭窗口...")
plt.show(block=False)
plt.pause(0.1)  # 短暂暂停让窗口显示
input("按Enter键退出...")
plt.close()