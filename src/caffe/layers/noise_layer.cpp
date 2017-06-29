#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  type_ = this->layer_param_.noise_param().noise_type();
  mean_ = 0.f;
  stdvar_ = 1.f;
  if (type_ == NoiseParameter::GAUSSIAN)
  {
    // gaussian
    mean_ = this->layer_param_.noise_param().mean();
    stdvar_ = this->layer_param_.noise_param().std_var();
    CHECK_GT(stdvar_, 0) << "Standard variance must be positive";
  }
  else if (type_ == NoiseParameter::UNIFORM)
  {
    // uniform
    upper_bound_ = this->layer_param_.noise_param().upper(); 
    lower_bound_ = this->layer_param_.noise_param().lower();
    CHECK_GT(upper_bound_, lower_bound_);
  }
  // TODO: maybe more type of noise
  else
    LOG(FATAL) << "Unknown type of noise! ";
}

template <typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  noise_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN)
  {
    Dtype* noise_data = noise_.mutable_cpu_data();
    switch(type_)
    {
      case NoiseParameter::UNIFORM:
        caffe_rng_uniform<Dtype>(count, lower_bound_, upper_bound_, noise_data);
        break;
      case NoiseParameter::GAUSSIAN:
      default:
        caffe_rng_gaussian<Dtype>(count, mean_, stdvar_, noise_data);
    }
    caffe_add<Dtype>(count, bottom_data, noise_data, top_data);
  }
  else
  {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);

} // namespace caffe
