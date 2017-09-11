#include "caffe/layers/dense_block_layer.hpp"
#include "caffe/layer_factory.hpp"
#include <string>
#include <cstdio>
namespace caffe {

static std::string _atoi(int num)
{
  char tmp[20];
  sprintf(tmp, "%d", num);
  return std::string(tmp);
}

template <typename Dtype>
static void reshapeHW(Blob<Dtype> &blob, int h, int w)
{
  std::vector<int> shape(blob.shape());
  shape.resize(4, 1);
  shape[2] = h;
  shape[3] = w;
  blob.Reshape(shape);
}

template <typename Dtype>
static void reshapeC(Blob<Dtype> &blob, int c)
{
  std::vector<int> shape(blob.shape());
  if (shape.size() >= 2)
  {
    shape[1] = c;
    blob.Reshape(shape);
  }
}

template <typename T>
inline static void append_back(vector<T> &dst, const vector<T> &src)
{
  for (size_t i = 0; i < src.size(); i++)dst.push_back(src[i]);
}

template <typename Dtype>
inline static void logLayerBlobs(const shared_ptr<Layer<Dtype> >& layer, const LayerParameter param)
{
  LOG(INFO) << param.name() << "(" << param.type() << "): blobs_.size() = " << layer->blobs().size();
}

// return the output blob name
static std::string add_layer(const std::string &input_blob_name, LayerParameter &param_tpl, const std::string &layer_name, bool in_place)
{
  param_tpl.clear_bottom();
  param_tpl.add_bottom(input_blob_name);
  std::string actual_name = param_tpl.name() + "_" + layer_name;
  param_tpl.set_name(actual_name);
  param_tpl.clear_top();
  if (in_place)
  {
    param_tpl.add_top(input_blob_name);
    return input_blob_name;
  }
  else
  {
    param_tpl.add_top(actual_name);
    return actual_name;
  }
}

static std::string add_concat_layer(const std::string &input_blob_name1, const std::string &intput_blob_name2, 
  LayerParameter &param_tpl, const std::string &layer_name)
{
  param_tpl.clear_bottom();
  param_tpl.add_bottom(input_blob_name1);
  param_tpl.add_bottom(intput_blob_name2);
  std::string actual_name = param_tpl.name() + "_" + layer_name;
  param_tpl.set_name(actual_name);
  param_tpl.clear_top();
  param_tpl.add_top(actual_name);
  return actual_name;
}

template <typename Dtype>
inline static void createLayers(const vector<LayerParameter> &params, vector<shared_ptr<Layer<Dtype> > >& layers)
{
  layers.clear();
  for (size_t i = 0; i < params.size(); i++)
    layers.push_back(LayerRegistry<Dtype>::CreateLayer(params[i]));
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  num_layers_ = block_param_.num_layers();
  growth_rate_ = block_param_.growth_rate();
  use_bottleneck_ = block_param_.use_bottleneck();
  bottleneck_rate_ = block_param_.bottleneck_rate();
  use_dropout_ = block_param_.use_dropout();

  CHECK_GT(num_layers_, 1) << "There should be at least 2 layers in the block";
  CHECK_GT(growth_rate_, 0) << "Growth rate cannot be 0";
  CHECK_GT(bottleneck_rate_, 0) << "Bottleneck rate should be at least 1";

  generataeLayerParamsForBlock();

  need_propagate_down_.resize(1, true);

  // create internal layer instances
  createLayers(bn_params_, bn_layers_);
  createLayers(scale_params_, scale_layers_);
  createLayers(relu_params_, relu_layers_);
  createLayers(conv3x3_params_, conv3x3_layers_);
  
  if (use_dropout_)
    createLayers(dropout_params_, dropout_layers_);
  
  if (use_bottleneck_)
  {
    createLayers(bottle_bn_params_, bottle_bn_layers_);
    createLayers(bottle_scale_params_, bottle_scale_layers_);
    createLayers(bottle_relu_params_, bottle_relu_layers_);
    createLayers(conv1x1_params_, conv1x1_layers_);
  }

  pre_bn_layer_ = LayerRegistry<Dtype>::CreateLayer(pre_bn_param_);
  post_scale_layer_ = LayerRegistry<Dtype>::CreateLayer(post_scale_param_);
  post_relu_layer_ = LayerRegistry<Dtype>::CreateLayer(post_relu_param_);

  if (use_bottleneck_)
    bottleneck_inter_.resize(num_layers_);
  conv3x3_inter_.resize(num_layers_);
  input_lth_.resize(num_layers_);
  output_lth_.resize(num_layers_);
  
  for (int l = 0; l < num_layers_; l++)
  {
    input_lth_[l].reset(new Blob<Dtype>);
    conv3x3_inter_[l].reset(new Blob<Dtype>);
    output_lth_[l].reset(new Blob<Dtype>);
    if (use_bottleneck_)
      bottleneck_inter_[l].reset(new Blob<Dtype>);
  }

  // Reshape is more like to adjust H and W in CNN case, maybe N.
  // In fact BatchNorm needs the bottom shape in LayerSetUp (see batch_norm_layer.cpp)
  setupShapeForInternalBlobs(bottom[0]);

  // invoke LayerSetUp for every internal layer
  vector<shared_ptr<Blob<Dtype> > >& expect_blobs(expect_blobs_);
  pre_bn_layer_->LayerSetUp(bottom, top);
  {
    append_back(expect_blobs, pre_bn_layer_->blobs());
    logLayerBlobs(pre_bn_layer_, pre_bn_param_);
  }
  
  for (int l = 0; l < num_layers_; l++)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());
      
      bottle_scale_layers_[l]->LayerSetUp(the_input_lth, the_conv3x3_inter_l);
      bottle_relu_layers_[l]->LayerSetUp(the_conv3x3_inter_l, the_conv3x3_inter_l);
      {
        append_back(expect_blobs, bottle_scale_layers_[l]->blobs());
        logLayerBlobs(bottle_scale_layers_[l], bottle_scale_params_[l]);
        append_back(expect_blobs, bottle_relu_layers_[l]->blobs());
        logLayerBlobs(bottle_relu_layers_[l], bottle_relu_params_[l]);
      }
      
      conv1x1_layers_[l]->LayerSetUp(the_conv3x3_inter_l, the_bottleneck_inter_l);
      bottle_bn_layers_[l]->LayerSetUp(the_bottleneck_inter_l, the_bottleneck_inter_l);
      scale_layers_[l]->LayerSetUp(the_bottleneck_inter_l, the_bottleneck_inter_l);
      relu_layers_[l]->LayerSetUp(the_bottleneck_inter_l, the_bottleneck_inter_l);
      {
        append_back(expect_blobs, conv1x1_layers_[l]->blobs());
        logLayerBlobs(conv1x1_layers_[l], conv1x1_params_[l]);
        append_back(expect_blobs, bottle_bn_layers_[l]->blobs());
        logLayerBlobs(bottle_bn_layers_[l], bottle_bn_params_[l]);
        append_back(expect_blobs, scale_layers_[l]->blobs());
        logLayerBlobs(scale_layers_[l], scale_params_[l]);
        append_back(expect_blobs, relu_layers_[l]->blobs());
        logLayerBlobs(relu_layers_[l], relu_params_[l]);
      }

      conv3x3_layers_[l]->LayerSetUp(the_bottleneck_inter_l, the_output_lth);
      {
        append_back(expect_blobs, conv3x3_layers_[l]->blobs());
        logLayerBlobs(conv3x3_layers_[l], conv3x3_params_[l]);
      }
    }
    else
    {
      scale_layers_[l]->LayerSetUp(the_input_lth, the_conv3x3_inter_l);
      relu_layers_[l]->LayerSetUp(the_conv3x3_inter_l, the_conv3x3_inter_l);
      {
        append_back(expect_blobs, scale_layers_[l]->blobs());
        logLayerBlobs(scale_layers_[l], scale_params_[l]);
        append_back(expect_blobs, relu_layers_[l]->blobs());
        logLayerBlobs(relu_layers_[l], relu_params_[l]);
      }

      conv3x3_layers_[l]->LayerSetUp(the_conv3x3_inter_l, the_output_lth);
      {
        append_back(expect_blobs, conv3x3_layers_[l]->blobs());
        logLayerBlobs(conv3x3_layers_[l], conv3x3_params_[l]);
      }
    }
    
    if (use_dropout_)
    {
      dropout_layers_[l]->LayerSetUp(the_output_lth, the_output_lth);
      append_back(expect_blobs, dropout_layers_[l]->blobs());
      logLayerBlobs(dropout_layers_[l], dropout_params_[l]);
    }

    bn_layers_[l]->LayerSetUp(the_output_lth, the_output_lth);
    {
      append_back(expect_blobs, bn_layers_[l]->blobs());
      logLayerBlobs(bn_layers_[l], bn_params_[l]);
    }

  }

  Blob<Dtype> tmp_top;
  tmp_top.ReshapeLike(maps_diff_); // we should not modify "top" here
  post_scale_layer_->LayerSetUp(vector<Blob<Dtype>*>(1, &tmp_top), vector<Blob<Dtype>*>(1, &tmp_top));
  post_relu_layer_->LayerSetUp(vector<Blob<Dtype>*>(1, &tmp_top), vector<Blob<Dtype>*>(1, &tmp_top));
  {
    append_back(expect_blobs, post_scale_layer_->blobs());
    logLayerBlobs(post_scale_layer_, post_scale_param_);
    append_back(expect_blobs, post_relu_layer_->blobs());
    logLayerBlobs(post_relu_layer_, post_relu_param_);
  }

  LOG(INFO) << "expect_blobs.size = " << expect_blobs.size();
  if (this->blobs().size() == 0)
  {
    // random initialize, assign pointer to internal layer.blobs() 
    // to this->blobs()
    append_back(this->blobs(), expect_blobs);
    expect_blobs.clear();
  }
  else
  {
    // copy the parameters into internel layer.blobs() and 
    // replace pointer in this->blobs() with the ones of internal layers.
    CHECK_EQ(expect_blobs.size(), this->blobs().size())
      << "number of paramster blobs does not match the expectation";
    // Size Check and Data Copy will be done in Reshape
  }

}

template <typename Dtype>
inline static void set_data_cpu(Blob<Dtype>& blob, Dtype* ptr)
{
  blob.set_cpu_data(ptr);
}

template <typename Dtype>
inline static void set_data_gpu(Blob<Dtype>& blob, Dtype* ptr)
{
  blob.set_gpu_data(ptr);
}

template <typename Dtype>
inline static void set_diff_cpu(Blob<Dtype>& blob, Dtype* ptr)
{
  blob.diff()->set_cpu_data(ptr);
}

template <typename Dtype>
inline static void set_diff_gpu(Blob<Dtype>& blob, Dtype* ptr)
{
  blob.diff()->set_gpu_data(ptr);
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::setupMemoryForInternalBlobs(Blob<Dtype>* bottom, Blob<Dtype>* top)
{
  vector<int> input_shape(bottom->shape());
  const int init_channles = input_shape[1];
  const int n = input_shape[0];
  const int h = input_shape[2];
  const int w = input_shape[3];

  Dtype* maps_diff_diff;
  Dtype* conv3x3_inter_mem_data;
  Dtype* conv3x3_inter_mem_diff;
  Dtype* output_mem_data;
  Dtype* output_mem_diff;

  if (Caffe::mode() == Caffe::GPU)
  {
    maps_diff_diff = maps_diff_.mutable_gpu_diff();
    conv3x3_inter_mem_data = conv3x3_inter_mem_.mutable_gpu_data();
    conv3x3_inter_mem_diff = conv3x3_inter_mem_.mutable_gpu_diff();
    output_mem_data = output_mem_.mutable_gpu_data();
    output_mem_diff = output_mem_.mutable_gpu_diff();
  }
  else
  {
    maps_diff_diff = maps_diff_.mutable_cpu_diff();
    conv3x3_inter_mem_data = conv3x3_inter_mem_.mutable_cpu_data();
    conv3x3_inter_mem_diff = conv3x3_inter_mem_.mutable_cpu_diff();
    output_mem_data = output_mem_.mutable_cpu_data();
    output_mem_diff = output_mem_.mutable_cpu_diff();
  }

  if (Caffe::mode() == Caffe::GPU)
    for (size_t i = 0; i < bottleneck_inter_.size(); i++)
    {
      bottleneck_inter_[i]->gpu_data();
      bottleneck_inter_[i]->gpu_diff();
    }
  else
    for (size_t i = 0; i < bottleneck_inter_.size(); i++)
    {
      bottleneck_inter_[i]->cpu_data();
      bottleneck_inter_[i]->cpu_diff();
    }

  typedef void (*SetFunc)(Blob<Dtype>& blob, Dtype* ptr);
  SetFunc set_data, set_diff;
  if (Caffe::mode() == Caffe::GPU)
  {
    set_data = set_data_gpu;
    set_diff = set_diff_gpu;
  }
  else
  {
    set_data = set_data_cpu;
    set_diff = set_diff_cpu;
  }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::setupShapeForInternalBlobs(const Blob<Dtype>* bottom)
{
  // Reshape is more like to adjust H and W in CNN case. 
  // In fact BatchNorm needs the bottom shape in LayerSetUp (see batch_norm_layer.cpp)
  // Here we check bottom should at least have N and C
  // H and W will be set to 1 if not specified

  CHECK_GE(bottom->shape().size(), 2)
    << "[DenseBlock] Cannot set up layer without knowing bottom shape";
  CHECK_GT(bottom->count(), 0)
    << "[DenseBlock] Invalid bottom shape";

  vector<int> btm_shape(bottom->shape());
  for (size_t i = btm_shape.size(); i < 4; i++)
    btm_shape.push_back(1);

  {
    std::string shapeStr("[");
    for (size_t i = 0; i < btm_shape.size(); i++)
      shapeStr += _atoi(btm_shape[i]) + ", ";
    shapeStr += "]";
    LOG(INFO) << "Reshape based on bottom shape: " << shapeStr;
  }
  vector<int> shape(btm_shape);
  shape[1] = btm_shape[1] + num_layers_ * growth_rate_;
  maps_diff_.Reshape(shape);

  conv3x3_inter_mem_.Reshape(shape);

  shape[1] = growth_rate_ * bottleneck_rate_;
  for (size_t i = 0; i < bottleneck_inter_.size(); i++)
    bottleneck_inter_[i]->Reshape(shape);

  for (size_t i = 0; i < conv3x3_inter_.size(); i++)
  {
    shape[1] = btm_shape[1] + i * growth_rate_;
    conv3x3_inter_[i]->Reshape(shape);
    input_lth_[i]->Reshape(shape);
  }
  
  shape[1] = growth_rate_;
  output_mem_.Reshape(shape);
  for (size_t i = 0; i < output_lth_.size(); i++)
    output_lth_[i]->Reshape(shape);

}

template <typename Dtype>
void DenseBlockLayer<Dtype>::convertToPlainLayers(vector<LayerParameter>& layer_params)
{
  layer_params.clear();
  layer_params.push_back(pre_bn_param_);
  for (int l = 0; l < num_layers_; l++)
  {
    
    if (use_bottleneck_)
    {
      layer_params.push_back(LayerParameter(bottle_scale_params_[l]));
      layer_params.push_back(LayerParameter(bottle_relu_params_[l]));
      layer_params.push_back(LayerParameter(conv1x1_params_[l]));
      layer_params.push_back(LayerParameter(bottle_bn_params_[l]));
    }

    layer_params.push_back(LayerParameter(scale_params_[l]));
    layer_params.push_back(LayerParameter(relu_params_[l]));
    layer_params.push_back(LayerParameter(conv3x3_params_[l]));
    
    if (use_dropout_)
    {
      layer_params.push_back(LayerParameter(dropout_params_[l]));
    }

    layer_params.push_back(LayerParameter(bn_params_[l]));
    layer_params.push_back(LayerParameter(concat_params_[l]));
  }
  layer_params.push_back(post_scale_param_);
  layer_params.push_back(post_relu_param_);

  for (size_t i = 0; i < layer_params.size(); i++)
    layer_params[i].clear_phase();

}

template <typename Dtype>
void DenseBlockLayer<Dtype>::generataeLayerParamsForBlock()
{
  const LayerParameter &param_ = this->layer_param_;
  
  // set up layer parameter template 
  LayerParameter bn_param_tpl(param_);
  bn_param_tpl.set_type("BatchNorm");
  bn_param_tpl.clear_dense_block_param();
  bn_param_tpl.clear_param();
  bn_param_tpl.clear_blobs();
  
  LayerParameter scale_param_tpl(param_);
  scale_param_tpl.set_type("Scale");
  scale_param_tpl.clear_dense_block_param();
  scale_param_tpl.clear_param();
  scale_param_tpl.clear_blobs();
  scale_param_tpl.mutable_scale_param()->set_bias_term(true);

  LayerParameter relu_param_tpl(param_);
  relu_param_tpl.set_type("ReLU");
  relu_param_tpl.clear_dense_block_param();
  relu_param_tpl.clear_param();
  relu_param_tpl.clear_blobs();
  if (param_.dense_block_param().has_relu_param())
    relu_param_tpl.mutable_relu_param()->CopyFrom(param_.dense_block_param().relu_param());

  LayerParameter dropout_param_tpl(param_);
  if (use_dropout_)
  {
    dropout_param_tpl.set_type("Dropout");
    dropout_param_tpl.clear_dense_block_param();
    dropout_param_tpl.clear_param();
    dropout_param_tpl.clear_blobs();
    if (param_.dense_block_param().has_dropout_param())
      dropout_param_tpl.mutable_dropout_param()->CopyFrom(param_.dense_block_param().dropout_param());
  }

  LayerParameter conv3x3_param_tpl(param_);
  conv3x3_param_tpl.set_type("Convolution");
  conv3x3_param_tpl.clear_dense_block_param();
  conv3x3_param_tpl.clear_param();
  conv3x3_param_tpl.clear_blobs();
  {
    ConvolutionParameter* conv_param = conv3x3_param_tpl.mutable_convolution_param();
    conv_param->add_kernel_size(3);
    conv_param->add_pad(1);
    conv_param->add_stride(1);
    conv_param->set_bias_term(param_.dense_block_param().conv3x3_bias_term());
    conv_param->set_num_output(growth_rate_);
    if (param_.dense_block_param().has_conv3x3_weights_filler())
      conv_param->mutable_weight_filler()->CopyFrom(param_.dense_block_param().conv3x3_weights_filler());
    if (conv_param->bias_term() && param_.dense_block_param().has_conv3x3_bias_filler())
      conv_param->mutable_bias_filler()->CopyFrom(param_.dense_block_param().conv3x3_bias_filler());
  }
  
  LayerParameter conv1x1_param_tpl(param_);
  if (use_bottleneck_)
  {
    conv1x1_param_tpl.set_type("Convolution");
    conv1x1_param_tpl.clear_dense_block_param();
    conv1x1_param_tpl.clear_param();
    conv1x1_param_tpl.clear_blobs();
    ConvolutionParameter* conv_param = conv1x1_param_tpl.mutable_convolution_param();
    conv_param->add_kernel_size(1);
    conv_param->add_pad(0);
    conv_param->add_stride(1);
    conv_param->set_bias_term(param_.dense_block_param().conv1x1_bias_term());
    conv_param->set_num_output(bottleneck_rate_ * growth_rate_);
    if (param_.dense_block_param().has_conv1x1_weights_filler())
      conv_param->mutable_weight_filler()->CopyFrom(param_.dense_block_param().conv1x1_weights_filler());
    if (conv_param->bias_term() && param_.dense_block_param().has_conv1x1_bias_filler())
      conv_param->mutable_bias_filler()->CopyFrom(param_.dense_block_param().conv1x1_bias_filler());
  }

  LayerParameter concat_param_tpl(param_);
  concat_param_tpl.set_type("Concat");
  concat_param_tpl.clear_dense_block_param();
  concat_param_tpl.clear_param();
  concat_param_tpl.clear_blobs();
  concat_param_tpl.mutable_concat_param()->set_axis(1);

  std::string prefix = param_.name();
  std::string previous_feature_name = param_.bottom(0);

  pre_bn_param_.CopyFrom(bn_param_tpl);
  previous_feature_name = add_layer(previous_feature_name, pre_bn_param_, "pre_bn", false);

  for (int l = 0; l < num_layers_; l++)
  {
    std::string the_num = _atoi(l);
    std::string blob = previous_feature_name;
    
    if (use_bottleneck_)
    {
      bottle_scale_params_.push_back(LayerParameter(scale_param_tpl));
      blob = add_layer(blob, bottle_scale_params_.back(), "bottle_scale_" + the_num, false);
      bottle_relu_params_.push_back(LayerParameter(relu_param_tpl));
      blob = add_layer(blob, bottle_relu_params_.back(), "bottle_relu_" + the_num, true);
      conv1x1_params_.push_back(LayerParameter(conv1x1_param_tpl));
      blob = add_layer(blob, conv1x1_params_.back(), "conv1x1_" + the_num, false);
      bottle_bn_params_.push_back(LayerParameter(bn_param_tpl));
      blob = add_layer(blob, bottle_bn_params_.back(), "bottle_bn_" + the_num, true);
    }
    
    scale_params_.push_back(LayerParameter(scale_param_tpl));
    // when not use bottleneck, in-place=>false to avoid modify BN data (stored in shared blob to be output)
    blob = add_layer(blob, scale_params_.back(), "scale_" + the_num, use_bottleneck_);
    relu_params_.push_back(LayerParameter(relu_param_tpl));
    blob = add_layer(blob, relu_params_.back(), "relu_" + the_num, true);
    conv3x3_params_.push_back(LayerParameter(conv3x3_param_tpl));
    blob = add_layer(blob, conv3x3_params_.back(), "conv3x3_" + the_num, false);
    if (use_dropout_)
    {
      dropout_params_.push_back(LayerParameter(dropout_param_tpl));
      blob = add_layer(blob, dropout_params_.back(), "dropout_" + the_num, true);
    }
    bn_params_.push_back(LayerParameter(bn_param_tpl));
    blob = add_layer(blob, bn_params_.back(), "bn_" + the_num, true);
    
    concat_params_.push_back(LayerParameter(concat_param_tpl));
    previous_feature_name = add_concat_layer(previous_feature_name, blob, concat_params_.back(), "concat_" + the_num);
  }
  
  concat_params_.back().set_top(0, param_.top(0));
  post_scale_param_.CopyFrom(scale_param_tpl);
  add_layer(concat_params_.back().top(0), post_scale_param_, "post_scale", true);
  post_relu_param_.CopyFrom(relu_param_tpl);
  add_layer(post_scale_param_.top(0), post_relu_param_, "post_relu", true);

}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom.size(), 1) << "Dense block should have exactly 1 input";
  CHECK_EQ(top.size(), 1)    << "Dense block should have exactly 1 output";

  vector<int> outputShape(bottom[0]->shape());
  CHECK_EQ(outputShape.size(), 4) << "Dense block are designed for 4D (n, c, h, w) tensor";

  setupShapeForInternalBlobs(bottom[0]);
  top[0]->ReshapeLike(maps_diff_);

  // invoke Reshape of the internal layers
  pre_bn_layer_->Reshape(bottom, top);

  for (int l = 0; l < num_layers_; l++)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());

      bottle_scale_layers_[l]->Reshape(the_input_lth, the_conv3x3_inter_l);
      bottle_relu_layers_[l]->Reshape(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv1x1_layers_[l]->Reshape(the_conv3x3_inter_l, the_bottleneck_inter_l);
      bottle_bn_layers_[l]->Reshape(the_bottleneck_inter_l, the_bottleneck_inter_l);
      scale_layers_[l]->Reshape(the_bottleneck_inter_l, the_bottleneck_inter_l);
      relu_layers_[l]->Reshape(the_bottleneck_inter_l, the_bottleneck_inter_l);

      conv3x3_layers_[l]->Reshape(the_bottleneck_inter_l, the_output_lth);

    }
    else
    {
      scale_layers_[l]->Reshape(the_input_lth, the_conv3x3_inter_l);
      relu_layers_[l]->Reshape(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv3x3_layers_[l]->Reshape(the_conv3x3_inter_l, the_output_lth);

    }

    if (use_dropout_)
    {
      dropout_layers_[l]->Reshape(the_output_lth, the_output_lth);
    }

    bn_layers_[l]->Reshape(the_output_lth, the_output_lth);

  }

  post_scale_layer_->Reshape(top, top);
  post_relu_layer_->Reshape(top, top);

  setupMemoryForInternalBlobs(bottom[0], top[0]);

  if (expect_blobs_.size() > 0)
  { // if there are parameters to be copied into working parameter blobs
    // now this->blob() has the parameters to be copied
    // expect_blobs_ stores the pointers to the working parameter blobs

    LOG(INFO) << "[DenseBlock] Copying trained parameters from LayerParamter";
    // check shape and copy
    for (size_t i = 0; i < expect_blobs_.size(); i++)
    {
      CHECK(this->blobs()[i]->shape() == expect_blobs_[i]->shape())
        << "parameter blobs does not match the expectation at " << i;
      caffe_copy(this->blobs()[i]->count(), this->blobs()[i]->cpu_data(), expect_blobs_[i]->mutable_cpu_data());
    }
    this->blobs().clear();
    append_back(this->blobs(), expect_blobs_);
    expect_blobs_.clear();
  }

}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(DenseBlockLayer);
REGISTER_LAYER_CLASS(DenseBlock);

} // namespace 

