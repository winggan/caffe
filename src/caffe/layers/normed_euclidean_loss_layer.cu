#include "caffe/layers/normed_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormedEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) 
{
  int count = bottom[0]->count();
  caffe_gpu_sub(
    count,
    bottom[0]->gpu_data(),
    bottom[1]->gpu_data(),
    this->diff_.mutable_gpu_data());
  caffe_gpu_add(
    count,
    bottom[0]->gpu_data(),
    bottom[1]->gpu_data(),
    this->diff_.mutable_gpu_diff()
  );
  
  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), norm_.mutable_gpu_data());
  caffe_gpu_abs(count, norm_.gpu_data(), norm_.mutable_gpu_data());
  caffe_gpu_add_scalar(count, eps_, norm_.mutable_gpu_data());
  caffe_gpu_sqrt(count, norm_.gpu_data(), norm_.mutable_gpu_data());
  caffe_gpu_div(count, this->diff_.gpu_data(), norm_.gpu_data(), this->diff_.mutable_gpu_data());
  
  //p0 =   1/(x0*x1)^(1/2) - (x1*(x0 - x1))/(2*(x0*x1)^(3/2))
  //   =  (x0 + x1)/(2*x0*(x0*x1)^(1/2)) =  x1*(x0 + x1) / (2*(x0*x1)^(3/2))
  //p1 = - 1/(x0*x1)^(1/2) - (x0*(x0 - x1))/(2*(x0*x1)^(3/2))
  //   = -(x0 + x1)/(2*x1*(x0*x1)^(1/2)) = -x0*(x0 + x1) / (2*(x0*x1)^(3/2))

  // pre-calculate part of the diff
  caffe_gpu_mul(count, bottom[1]->gpu_data(), this->diff_.gpu_diff(),
    bottom[0]->mutable_gpu_diff());
  caffe_gpu_mul(count, bottom[0]->gpu_data(), this->diff_.gpu_diff(),
    bottom[1]->mutable_gpu_diff());

  Dtype dot;
  caffe_gpu_dot(count, this->diff_.gpu_data(), this->diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NormedEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const int count = norm_.count();
  caffe_gpu_mul(count, norm_.gpu_data(), norm_.gpu_data(), norm_.mutable_gpu_diff());
  caffe_gpu_mul(count, norm_.gpu_diff(), norm_.gpu_data(), norm_.mutable_gpu_diff());

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      // top[0]->cpu_diff()[0] = loss_weight
      const Dtype alpha = Dtype(0.5f) * sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_div(count, bottom[i]->gpu_diff(), norm_.gpu_diff(), bottom[i]->mutable_gpu_diff());
      caffe_gpu_scale(count, alpha, bottom[i]->gpu_diff(), bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormedEuclideanLossLayer);

} // namespace caffe
