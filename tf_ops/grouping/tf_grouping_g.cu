// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
    	    float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful

/**
 * @brief KNN函数实现的排序GPU支持

 * @in
 * @param b:in BatchSize
 * @param n:in 输入待定点的数量input points
 * @param m:IN query points 的数量
 * @param k:in KNN的k值
 * @param dist:in 输入的点位置欧式距离平法信息【Batch】[m][n]
 * @param outi:out 前k的距离点位的索引
 * @param out：out 前k的距离点位的距离
 * @return __global__ 
 */
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;//定位开始位置
    outi+=m*n*batch_index;
    out+=m*n*batch_index;


    int index = threadIdx.x; //线程id
    int stride = blockDim.x; //block大小

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

/**
 * @brief KNN的直接gpu实现
 * 
 * @param b 
 * @param n 
 * @param m 
 * @param k 
 * @param xyz1 [b][n][3]
 * @param xyz2 [b][m][3]
 * @param outi 
 * @param out 
 * @return __global__ 
 */
__global__ void knn_kernal_gpu(int b,int n,int m,int k,const float * xyz1,const float * xyz2,float * outi,float *out){
    //TODO:实现knn核算子
    int batch_index = blockIdx.x;
    xyz1+=3*n*batch_index;
    xyz2+=3*m*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    __shared__ int point_index[m][n];
    __shared__ float point_val[m][n]; 
    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            for(int pos =0;pos<3;pos++){
                point_val[j][s] += (xyz1[s][pos] - xyz2[j][pos])*(xyz1[s][pos] - xyz2[j][pos]);
                out[j*n+s] += (xyz1[s][pos] - xyz2[j][pos])*(xyz1[s][pos] - xyz2[j][pos]) ;//取出的是此batch中xyz1中第s个和xyz2中第j个个的距离
            }
            outi[j*n+s] = s;
            point_index[j][s]=s;
        }
    }
        //此处不需要_syncthreads()因为都是在一个block中的数据 不会互相影响
    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (point_val[j][t]<point_val[j][min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = point_val[j][min];
                point_val[j][min] = point_val[j][s];
                point_val[j][s] = tmp;
                int tmpi = point_index[j][min];
                point_index[j][min] = point_index[j][s];
                point_index[j][s] = tmpi;
            }
            //最后将结果写入输出变量
            p_dist[s] = point_val[j][s];
            outi[j*n+s] = point_index[j][min];
        }
    }


}
void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<b,256>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,256>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}

void knn_gpu(int b,int n,int m,int k,const float * xyz1,const float * xyz2,float * outi,float *out)
{
    knn_kernal_gpu<<<b,256>>>(b,n,m,k,xyz1,xyz2,outi,out);
}
