# VesselClusterNet
## 实验：

对初步分割结果进行细化，利用聚类算法根据初步分割结果对图像进行分块，每一个不规则图块聚合为一个点，然后再将这些点集合作为序列，利用ViT进行结构建模，再考虑如何反馈给图像特征

**阶段 1**

1. 加载输入数据 ✅

2. 编写聚类，实现3D输入上的图块划分✅

   1. 血管占比非常小，在标签图像中，标签只占了5%左右

      <img src="https://hexo-img.obs.cn-east-3.myhuaweicloud.com/llf/image-20230511161227776.png" alt="image-20230511161227776" style="zoom:50%;" />

   2. 实现方法：**KMeans聚类**方法，对分割结果进行聚类划分

      <img src="https://hexo-img.obs.cn-east-3.myhuaweicloud.com/llf/image-20230511160442608.png" alt="image-20230511160442608" style="zoom:50%;" />

   3. 提高簇的数量后

      <img src="https://hexo-img.obs.cn-east-3.myhuaweicloud.com/llf/image-20230511160818575.png" alt="image-20230511160818575" style="zoom:50%;" />

   4. 可添加基于流形计算的测地距离，ISOMAP

3. 编写测地线距离算法，实现每个mask点都有正确的测地线距离

想法：利用聚类方法，多簇=>少簇，一边融合簇，一边添加新簇