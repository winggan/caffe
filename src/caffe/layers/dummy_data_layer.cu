#include "caffe/layers/dmap_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DmapReLUForward(const int n, const Dtype* in, Dtype* out, const Dtype* gt,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype indicator = (Dtype)( (in[index] > 0) | (gt[index] > 0) );
    out[index] = in[index] * ( indicator + (1 - indicator) * negative_slope )
  }
}

template <typename Dtype>
void DmapReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2) << "Ground truth density map should be provided";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) 
    << "Ground truth density map should be the same size as the estimated one";
    
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* gt_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  DmapReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, gt_data, negative_slope);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void DmapReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* gt, Dtype* out_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype indicator = (Dtype)( (in_data[index] > 0) | (gt[index] > 0) );
    out_diff[index] = in_diff[index] * ( indicator + (1 - indicator) * negative_slope );
  }
}

template <typename Dtype>
void DmapReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* gt_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DmapReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, gt_data, bottom_diff, negative_slope);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DmapReLULayer);


}  // namespace caffe
