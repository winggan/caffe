#include "caffe/layers/dense_block_layer.hpp"

namespace caffe {

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

} // namespace caffe