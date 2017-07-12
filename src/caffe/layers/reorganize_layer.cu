#include "caffe/layers/reorganize_layer.hpp"

namespace caffe 
{
  
template <typename Dtype>
__global__ void ReorganizeArray(const Dtype *src, Dtype *dst, const int *idx, int count)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    dst[i] = src[idx[i]];
  }

}
  
template <typename Dtype>
void ReorganizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const int* forward_idx_data = forward_idx_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  ReorganizeArray<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    bottom_data, top_data, forward_idx_data, count);
  CUDA_POST_KERNEL_CHECK;
}
  
template <typename Dtype>
void ReorganizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0]) {
    return;
  }
  const int* backward_idx_data = backward_idx_.gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  ReorganizeArray<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    top_diff, bottom_diff, backward_idx_data, count);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ReorganizeLayer);
  
} // namespace caffe