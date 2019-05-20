#ifndef CAFFE_REORG_LAYER_HPP_
#define CAFFE_REORG_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// This layer is a wrong implementaion of reorganize layer
// ONLY for simulating Darknet's behaviour
// See also:
//    https://github.com/pjreddie/darknet/blob/master/src/reorg_layer.c:108
template <typename Dtype>
class ReorgLayer : public Layer<Dtype> {
 public:
  explicit ReorgLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reorg"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  bool isFlatten_; // true => flattern, false => stack
  int stride_w_;
  int stride_h_;
};

}  // namespace caffe

#endif // CAFFE_REORG_LAYER_HPP_