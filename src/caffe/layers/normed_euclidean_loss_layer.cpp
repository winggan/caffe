#include "caffe/layers/normed_euclidean_loss_layer.hpp"

#include <caffe/util/math_functions.hpp>

namespace caffe {

template <typename Dtype>
NormedEuclideanLossLayer<Dtype>::NormedEuclideanLossLayer(const LayerParameter& param)
  : EuclideanLossLayer<Dtype>(param), eps_(1e-2f), norm_()
{
  if (param.norm_euclid_loss_param().has_eps())
    eps_ = param.norm_euclid_loss_param().eps();

  CHECK_GT(eps_, 0) << "eps_ should be a positive real.";
}

template <typename Dtype>
void NormedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  EuclideanLossLayer<Dtype>::Reshape(bottom, top);
  norm_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NormedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) 
{
  int count = bottom[0]->count();
  caffe_sub(
    count,
    bottom[0]->cpu_data(),
    bottom[1]->cpu_data(),
    this->diff_.mutable_cpu_data());
  caffe_add(
    count,
    bottom[0]->cpu_data(),
    bottom[1]->cpu_data(),
    this->diff_.mutable_cpu_diff()
  );
  
  caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), norm_.mutable_cpu_data());
  caffe_abs(count, norm_.cpu_data(), norm_.mutable_cpu_data());
  caffe_add_scalar(count, eps_, norm_.mutable_cpu_data());
  caffe_sqrt(count, norm_.cpu_data(), norm_.mutable_cpu_data());
  caffe_div(count, this->diff_.cpu_data(), norm_.cpu_data(), this->diff_.mutable_cpu_data());
  
  //p0 =   1/(x0*x1)^(1/2) - (x1*(x0 - x1))/(2*(x0*x1)^(3/2))
  //   =  (x0 + x1)/(2*x0*(x0*x1)^(1/2)) =  x1*(x0 + x1) / (2*(x0*x1)^(3/2))
  //p1 = - 1/(x0*x1)^(1/2) - (x0*(x0 - x1))/(2*(x0*x1)^(3/2))
  //   = -(x0 + x1)/(2*x1*(x0*x1)^(1/2)) = -x0*(x0 + x1) / (2*(x0*x1)^(3/2))

  // pre-calculate part of the diff
  caffe_mul(count, bottom[1]->cpu_data(), this->diff_.cpu_diff(),
    bottom[0]->mutable_cpu_diff());
  caffe_mul(count, bottom[0]->cpu_data(), this->diff_.cpu_diff(),
    bottom[1]->mutable_cpu_diff());

  Dtype dot = caffe_cpu_dot(count, this->diff_.cpu_data(), this->diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NormedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const int count = norm_.count();
  caffe_mul(count, norm_.cpu_data(), norm_.cpu_data(), norm_.mutable_cpu_diff());
  caffe_mul(count, norm_.cpu_diff(), norm_.cpu_data(), norm_.mutable_cpu_diff());

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      // top[0]->cpu_diff()[0] = loss_weight
      const Dtype alpha = Dtype(0.5f) * sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_div(count, bottom[i]->cpu_diff(), norm_.cpu_diff(), bottom[i]->mutable_cpu_diff());
      caffe_cpu_scale(count, alpha, bottom[i]->cpu_diff(), bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormedEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(NormedEuclideanLossLayer);
REGISTER_LAYER_CLASS(NormedEuclideanLoss);

} // namespace caffe
