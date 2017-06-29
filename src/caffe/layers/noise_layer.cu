#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN)
  {
    Dtype* noise_data = noise_.mutable_gpu_data();
    switch(type_)
    {
      case NoiseParameter::UNIFORM:
        caffe_gpu_rng_uniform<Dtype>(count, lower_bound_, upper_bound_, noise_data);
        break;
      case NoiseParameter::GAUSSIAN:
      default:
        caffe_gpu_rng_gaussian<Dtype>(count, mean_, stdvar_, noise_data);
    }
    caffe_gpu_add<Dtype>(count, bottom_data, noise_data, top_data);
  }
  else
  {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);

}