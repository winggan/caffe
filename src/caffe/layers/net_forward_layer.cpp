#include "caffe/layers/net_forward_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void NetForwardLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const LayerParameter& layer_param = this->layer_param_;
  const NetForwardParameter& forward_param = layer_param.net_forward_param();
  
  CHECK_GT(forward_param.net_proto().length(), 0) 
    << "Net prototxt should be specified";
  CHECK_GT(forward_param.net_model().length(), 0) 
    << "Net model should be specified";
    
  CHECK_EQ(top.size(), forward_param.top_blob_names_size())
    << "Should specify the number of outputs the same as the numbers tops";
    
  this->net_.reset(new Net<Dtype>(forward_param.net_proto(), caffe::TEST));
  this->net_->CopyTrainedLayersFrom(forward_param.net_model());
 
  Net<Dtype> &net = *(this->net_.get());
  num_input_blobs_ = net.num_inputs();
  CHECK_EQ(num_input_blobs_, bottom.size())
    << "Number of bottom blobs is expected to match inputs of the network";
 
  for (int i = 0; i < layer_param.top_size(); i ++)
  {
    const string &output_name = forward_param.top_blob_names(i);
    size_t idx;
    for (idx = 0; idx < net.blob_names().size(); idx++)
      if (net.blob_names()[idx] == output_name)
        break;
    CHECK_LT(idx, net.blob_names().size())
      << "Blob " << output_name << " does not exists";
    this->output_blob_idx_.push_back((int)idx);
  }

}

template <typename Dtype>
void NetForwardLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  const LayerParameter& layer_param = this->layer_param_;
  const NetForwardParameter& forward_param = layer_param.net_forward_param();

  Net<Dtype> &net = *(this->net_.get());

  if (forward_param.can_reshape())
  { // reshape net's input blobs according to bottom blobs
    for (size_t i = 0; i < bottom.size(); i++)
      net.input_blobs()[i]->ReshapeLike(*bottom[i]);
    net.Reshape(); // apply the reshape operation
  }
  else
  { // check input shape match bottom shape
    for (size_t i = 0; i < bottom.size(); i++)
      CHECK(net.input_blobs()[i]->shape() == bottom[i]->shape())
        << i << "-th input blobs: shape dis-match";
  }

  for (size_t i = 0; i < output_blob_idx_.size(); i++)
    top[i]->ReshapeLike(*(net.blobs()[output_blob_idx_[i]].get()));
}

template <typename Dtype>
void NetForwardLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  Net<Dtype> &net = *(this->net_.get());

  for (size_t i = 0; i < bottom.size(); i++)
    caffe_copy(bottom[i]->count(),
      bottom[i]->cpu_data(), net.input_blobs()[i]->mutable_cpu_data());
  
  net.Forward(NULL);

  for (size_t i = 0; i < output_blob_idx_.size(); i++)
    caffe_copy(net.blobs()[output_blob_idx_[i]]->count(),
      net.blobs()[output_blob_idx_[i]]->cpu_data(), top[i]->mutable_cpu_data());
}

template <typename Dtype>
void NetForwardLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  // do nothing
}

#ifdef CPU_ONLY
STUB_GPU(NetForwardLayer);
#else

template <typename Dtype>
void NetForwardLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  Net<Dtype> &net = *(this->net_.get());

  for (size_t i = 0; i < bottom.size(); i++)
    caffe_copy(bottom[i]->count(),
      bottom[i]->gpu_data(), net.input_blobs()[i]->mutable_gpu_data());

  net.Forward(NULL);

  for (size_t i = 0; i < output_blob_idx_.size(); i++)
    caffe_copy(net.blobs()[output_blob_idx_[i]]->count(),
      net.blobs()[output_blob_idx_[i]]->gpu_data(), top[i]->mutable_gpu_data());
}

template <typename Dtype>
void NetForwardLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  // do nothing
}
#endif

INSTANTIATE_CLASS(NetForwardLayer);
REGISTER_LAYER_CLASS(NetForward);


} // namespace caffe