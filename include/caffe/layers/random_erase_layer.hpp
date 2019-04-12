#ifndef CAFFE_NOISE_LAYER_HPP_
#define CAFFE_NOISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
/**
 * Add a random mask to the input blob element-wisely, noise in the mask is 
 * defined by FillerParameter
 */
template <typename Dtype>
class RandomEraseLayer : public NeuronLayer<Dtype>
{
 public:
  explicit RandomEraseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
      
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RandomErase"; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
  float area_upper_, area_lower_, aspect_upper_, aspect_lower_;
  float area1_, aspect1_;
  Blob<Dtype> noise_, all_zeros_;
  
  Blob<int> rects_; // n x 4
  Blob<float> randoms_; // n x 4
  
  shared_ptr<Layer<Dtype> > noise_layer_;
  std::vector<Blob<Dtype>*> noise_btm_, noise_top_;
  bool truncate_;
  Dtype trunc_lower_, trunc_upper_;
};

} // namespace caffe

#endif // CAFFE_NOISE_LAYER_HPP_