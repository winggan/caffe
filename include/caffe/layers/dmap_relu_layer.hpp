#include "caffe/layers/neuron_layer.hpp"

namespace caffe { 

template <typename Dtype>
class DmapReLULayer : public NeuronLayer<Dtype>
{
 public:
  explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  
  virtual inline const char* type() const { return "DmapReLU"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
};

} // namespace caffe