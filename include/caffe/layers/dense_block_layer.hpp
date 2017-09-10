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
  void setupShapeForInternalBlobs(const Blob<Dtype>* bottom);

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
  LayerParameter pre_bn_param_, post_scale_param_, post_relu_param_;

  vector<shared_ptr<Layer<Dtype> > > bn_layers_, scale_layers_, relu_layers_, conv3x3_layers_, dropout_layers_;
  vector<shared_ptr<Layer<Dtype> > > bottle_bn_layers_, bottle_scale_layers_, bottle_relu_layers_, conv1x1_layers_;
  shared_ptr<Layer<Dtype> > pre_bn_layer_, post_scale_layer_, post_relu_layer_;
  
  vector<shared_ptr<Blob<Dtype> > > expect_blobs_;

  Blob<Dtype> maps_diff_; // temp space to store the diff of a conv block so it can be added to top[0].diff
                          // bottom[0].diff is copied from top[0].diff at last
                          // same size as top[0]
  Blob<Dtype> conv3x3_inter_mem_; // size = n * (k0 + k*(L-1)) * h * w
  vector<shared_ptr<Blob<Dtype> > > bottleneck_inter_; // one for each conv3x3 block if use bottleneck
                                          // all are with the same size n * (bottleneck_rate*k) * h * w

  // the following blob are only headers without actually memory allocation
  vector<shared_ptr<Blob<Dtype> > > conv3x3_inter_; // one for each conv3x3 block to store its input (after scale)
                                                    // conv3x3Inter[i].size = n * (k0 + k*(i-1)) * h * w, shared from conv3x3_inter_mem_
                                                    // ReLU.Backward needs its input data, which will be computed again at 
                                                    // backward procedure (a Scale.Forward)
  vector<shared_ptr<Blob<Dtype> > > input_lth_;  // input for l-th conv block, size = n * (k0 + k*(l-1)) * h * w, shared from top[0]
  vector<shared_ptr<Blob<Dtype> > > output_lth_; // output for l-th conv block, size = n * k * h * w, shared from top[0]

  

  vector<bool> need_propagate_down_; // size = 1 => { true } 

};

} // namespace caffe

#endif // CAFFE_DENSE_BLOCK_LAYER_HPP_