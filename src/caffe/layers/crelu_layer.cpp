#include "caffe/layers/crelu_layer.hpp"

namespace caffe {
  
template <typename Dtype>
void CReLULayer::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const LayerParameter& param = this->layer_param_;
  CHECK_QE(param.has_scale_param(), false) << "Should not contain any scale and shift";
}

template <typename Dtype>
void CReLULayer::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const LayerParameter& param = this->layer_param_;
  CHECK_GE(bottom[0]->num_axes(), 2);
  
  vector<int> shape = bottom[0]->shape();
  shape[1] *= 2;
  top[0]->Reshape(shape);
  this->sign.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CReLULayer::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data_0 = top[0]->mutable_cpu_data();
  Dtype* top_data_1 = top_data_0 + count;
  Dtype* sign_data = this->sign.mutable_cpu_data();
  
  for (int i = 0; i < count; i ++)
  {
    bool is_negative = bottom_data[i] < 0;
    top_data_0[i] = is_negative ? 0 :  bottom_data[i];
    top_data_1[i] = is_negative ? -bottom_data[i] : 0;
    sign_data = is_negative ? Dtype(-1) : Dtype(1);
  }
}

template <typename Dtype>
void CReLULayer::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0])
    return;
  
  const int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff_0 = top[0]->cpu_diff();
  const Dtype* top_diff_1 = top_diff_0 + count;
  const Dtype* sign_data = this->sign.cpu_data();
  
  for (int i = 0; i < count; i ++)
  {
    bool is_negative = sign_data[i] < 0;
    bottom_diff[i] = is_negative ? -top_diff_1[i] : top_diff_0[i];
  }
}

}