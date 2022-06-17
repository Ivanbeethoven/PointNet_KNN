# 基于PointNet项目 对KNN实现进行CUDA的进一步支持

> 原始README见 [`README_ORIGIN.md`](./README_ORIGIN.md)

对Pointnet中的KNN实现进行了进一步的CUDA支持

源 代码实现为 (进行了二次注释)
```python
def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print b, n, c, m  
    '''
    b = batch_size
    n = ndataset
    c = 3
    m = npoint
    '''
    print xyz1, (b,1,n,c)
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    '''
    xyz1:(batch_size,m,n,3)
    xyz2:(batch_size,m,n,3)

    '''
    dist = tf.reduce_sum((xyz1-xyz2)**2, axis=-1) #按照最后一个维度求和 即求出的是欧式距离平方 
    #dist :(batch_size,m,n)
    print dist, k
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    print idx, val
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx
```

### 源代码分析
    其中，该代码段先是待带查询的点（xyt2）与已知点（xyz1），求出所有两两对应的欧氏距离的平方（dist），这部分是调用了tf的自带函数,`tf.tile`,`tf.reducesum`,`tf.slice`，然后利用CUDA的实现的在batch内的选择排序进行计算。