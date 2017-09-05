#ifndef CAFFE_NORMED_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_NORMED_EUCLIDEAN_LOSS_LAYER_HPP_

#include "caffe/layers/euclidean_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class NormedEuclideanLossLayer : public EuclideanLossLayer<Dtype> {
 public:
  explicit NormedEuclideanLossLayer(const LayerParameter& param);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NormedEuclideanLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype eps_;
  Blob<Dtype> norm_;
};


} // namespace caffe

#endif // CAFFE_NORMED_EUCLIDEAN_LOSS_LAYER_HPP_