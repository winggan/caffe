#ifndef CAFFE_DENSE_BLOCK_LAYER_HPP_
#define CAFFE_DENSE_BLOCK_LAYER_HPP_

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DenseBlockLayer : public Layer<Dtype>
{
 public:
  explicit DenseBlockLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      block_param_(param.dense_block_param()){}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseBlock"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  void convertToPlainLayers(vector<LayerParameter>& layer_params);

 protected:
  
  void generataeLayerParamsForBlock();

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  int num_layers_;
  int growth_rate_;
  bool use_bottleneck_;
  int bottleneck_rate_;
  bool use_dropout_;
  DenseBlockParameter block_param_;

  vector<LayerParameter> bn_params_, scale_params_, relu_params_, conv3x3_params_, dropout_params_;
  vector<LayerParameter> bottle_bn_params_, bottle_scale_params_, bottle_relu_params_, conv1x1_params_;
  vector<LayerParameter> concat_params_; // does not need for actual computation
  LayerParameter pre_bn_param, post_scale_param, post_relu_param;
};

} // namespace caffe

#endif // CAFFE_DENSE_BLOCK_LAYER_HPP_