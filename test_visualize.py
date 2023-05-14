import networkx as nx
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一个带有节点坐标的图
n = 10
points = np.random.rand(n, 3)
G = nx.Graph()
for i in range(n):
    G.add_node(i, pos=points[i])

# 计算最小生成树
mini_tree_G = nx.minimum_spanning_tree(G)

# 可视化完全连通图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pos = nx.get_node_attributes(mini_tree_G, 'pos')

# 绘制节点
for i, p in pos.items():
    ax.scatter(p[0], p[1], p[2], color='lightblue')
    ax.text(p[0], p[1], p[2], i+1, fontsize=16, color='black')

# 绘制边
for e in mini_tree_G.edges():
    ax.plot([pos[e[0]][0], pos[e[1]][0]], [pos[e[0]][1], pos[e[1]][1]], [pos[e[0]][2], pos[e[1]][2]], color='gray')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
