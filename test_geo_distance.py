import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一个随机的三维点集
points = np.random.rand(100, 3)

# 构建三角网格
tri = Delaunay(points)

# 计算相邻点之间的距离矩阵
dist_matrix = tri.vertex_neighbor_vertices[0][tri.vertex_neighbor_vertices[1]]

# 计算两个点之间的测地距离
start_idx = 0
end_idx = 1
_, pred = dijkstra(dist_matrix, indices=start_idx, return_predecessors=True)
path = [end_idx]
while path[-1] != start_idx:
    path.append(pred[path[-1]])
path.reverse()
geodesic_dist = dist_matrix[start_idx, end_idx] + np.sum(dist_matrix[path[:-1], path[1:]])

# 可视化三维形状和测地距离
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, alpha=0.5)
ax.plot(points[[start_idx, end_idx], 0], points[[start_idx, end_idx], 1], points[[start_idx, end_idx], 2], 'r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title(f"Geodesic distance = {geodesic_dist:.2f}")
plt.show()