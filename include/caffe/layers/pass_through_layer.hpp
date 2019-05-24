#ifndef CAFFE_PASS_THROUGH_LAYER_HPP_
#define CAFFE_PASS_THROUGH_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/reorganize_layer.hpp"

namespace caffe {

template <typename Dtype>
class PassThroughLayer : public ReorganizeLayer<Dtype> {

 public:
  explicit PassThroughLayer(const LayerParameter& param)
      : ReorganizeLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PassThrough"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

};

} // namespace caffe

#endif // CAFFE_PASS_THROUGH_LAYER_HPP_
