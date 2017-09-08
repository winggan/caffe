#include "caffe/layers/dense_block_layer.hpp"
#include <string>
#include <cstdio>
namespace caffe {

static std::string _atoi(int num)
{
  char tmp[20];
  sprintf(tmp, "%d", num);
  return std::string(tmp);
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
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::convertToPlainLayers(vector<LayerParameter>& layer_params)
{
  layer_params.clear();
  for (int l = 0; l < num_layers_; l++)
  {
    
    if (use_bottleneck_)
    {
      layer_params.push_back(LayerParameter(bottle_bn_params_[l]));
      layer_params.push_back(LayerParameter(bottle_scale_params_[l]));
      layer_params.push_back(LayerParameter(bottle_relu_params_[l]));
      layer_params.push_back(LayerParameter(conv1x1_params_[l]));
    }

    layer_params.push_back(LayerParameter(bn_params_[l]));
    layer_params.push_back(LayerParameter(scale_params_[l]));
    layer_params.push_back(LayerParameter(relu_params_[l]));
    layer_params.push_back(LayerParameter(conv3x3_params_[l]));
    
    if (use_dropout_)
    {
      layer_params.push_back(LayerParameter(dropout_params_[l]));
    }

    layer_params.push_back(LayerParameter(concat_params_[l]));
  }

}

template <typename Dtype>
void DenseBlockLayer<Dtype>::generataeLayerParamsForBlock()
{
  const LayerParameter &param_ = this->layer_param_;
  
  // set up layer parameter template 
  LayerParameter bn_param_tpl(param_);
  bn_param_tpl.set_type("BatchNorm");
  bn_param_tpl.clear_dense_block_param();
  
  LayerParameter scale_param_tpl(param_);
  scale_param_tpl.set_type("Scale");
  scale_param_tpl.clear_dense_block_param();
  scale_param_tpl.mutable_scale_param()->set_bias_term(true);

  LayerParameter relu_param_tpl(param_);
  relu_param_tpl.set_type("ReLU");
  relu_param_tpl.clear_dense_block_param();
  relu_param_tpl.mutable_relu_param()->CopyFrom(param_.dense_block_param().relu_param());

  LayerParameter dropout_param_tpl(param_);
  if (use_dropout_)
  {
    dropout_param_tpl.set_type("Dropout");
    dropout_param_tpl.clear_dense_block_param();
    dropout_param_tpl.mutable_dropout_param()->CopyFrom(param_.dense_block_param().dropout_param());
  }

  LayerParameter conv3x3_param_tpl(param_);
  conv3x3_param_tpl.set_type("Convolution");
  conv3x3_param_tpl.clear_dense_block_param();
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
  concat_param_tpl.mutable_concat_param()->set_axis(1);

  std::string prefix = param_.name();
  std::string previous_feature_name = param_.bottom(0);
  for (int l = 0; l < num_layers_; l++)
  {
    std::string the_num = _atoi(l);
    std::string blob = previous_feature_name;
    if (use_bottleneck_)
    {
      bottle_bn_params_.push_back(LayerParameter(bn_param_tpl));
      blob = add_layer(blob, bottle_bn_params_.back(), "bottle_bn_" + the_num, false);
      bottle_scale_params_.push_back(LayerParameter(scale_param_tpl));
      blob = add_layer(blob, bottle_scale_params_.back(), "bottle_scale_" + the_num, true);
      bottle_relu_params_.push_back(LayerParameter(relu_param_tpl));
      blob = add_layer(blob, bottle_relu_params_.back(), "bottle_relu_" + the_num, true);
      conv1x1_params_.push_back(LayerParameter(conv1x1_param_tpl));
      blob = add_layer(blob, conv1x1_params_.back(), "conv1x1_" + the_num, false);
    }
    bn_params_.push_back(LayerParameter(bn_param_tpl));
    // when use bottleneck, input of this bn is the conv1x1
    blob = add_layer(blob, bn_params_.back(), "bn_" + the_num, use_bottleneck_); 
    scale_params_.push_back(LayerParameter(scale_param_tpl));
    blob = add_layer(blob, scale_params_.back(), "scale_" + the_num, true);
    relu_params_.push_back(LayerParameter(relu_param_tpl));
    blob = add_layer(blob, relu_params_.back(), "relu_" + the_num, true);
    conv3x3_params_.push_back(LayerParameter(conv3x3_param_tpl));
    blob = add_layer(blob, conv3x3_params_.back(), "conv3x3_" + the_num, false);
    if (use_dropout_)
    {
      dropout_params_.push_back(LayerParameter(dropout_param_tpl));
      blob = add_layer(blob, dropout_params_.back(), "dropout_" + the_num, true);
    }
    
    concat_params_.push_back(LayerParameter(concat_param_tpl));
    previous_feature_name = add_concat_layer(previous_feature_name, blob, concat_params_.back(), "concat_" + the_num);
  }
  
  concat_params_.back().set_top(0, param_.top(0));
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  NOT_IMPLEMENTED;
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

