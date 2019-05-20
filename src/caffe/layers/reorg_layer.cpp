#include "caffe/layers/reorg_layer.hpp"

// only correct when forwarding or backwarding of flatten (Depth to Space)
template <typename Dtype>
static void reorg_cpu(Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}


namespace caffe {

template <typename Dtype>
void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  const ReorganizeParameter& reorg_param = this->layer_param_.reorg_param();
  isFlatten_ = reorg_param.is_flatten();
  if (reorg_param.has_stride() && (reorg_param.has_stride_w() || reorg_param.has_stride_h()))
    LOG(FATAL) << "Can not specify stride and stride_w / stride_h at the same time.";
  
  if (reorg_param.has_stride())
    stride_h_ = stride_w_ = reorg_param.stride();
  else if (reorg_param.has_stride_w() && reorg_param.has_stride_h())
  {
    stride_w_ = reorg_param.stride_w();
    stride_h_ = reorg_param.stride_h();
  }
  else
    LOG(FATAL) << "stride or (stride_w, stride_h) should be specified. ";

  CHECK_EQ(stride_h_, stride_w_) << "Darknet's implementation only support same stride";
}

template <typename Dtype>
void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  CHECK_EQ(bottom[0]->shape().size(), 4) << this->type() << " Layer only support 4-D input (NCHW)";
  const Blob<Dtype> *const input = bottom[0];
  if (isFlatten_)
    CHECK_EQ(0, input->shape()[1] % (stride_w_ * stride_h_)) // c % (s_w * s_h) == 0
      << "Channels should be divided by (stride_w * stride_h) when flattening.";
  else
  { // h % s_h == 0 && w % s_w == 0
    CHECK_EQ(0, input->shape()[2] % stride_h_)
      << "Height should be divided by stride_h when stacking";
    CHECK_EQ(0, input->shape()[3] % stride_w_)
      << "Width  should be divided by stride_w when stacking";
  }
  
  if (isFlatten_)
  {
    std::vector<int> outputShape(input->shape());
    outputShape[1] /= stride_h_ * stride_w_;
    outputShape[2] *= stride_h_;
    outputShape[3] *= stride_w_;
    top[0]->Reshape(outputShape);
   }
  else
  { // stacking
    std::vector<int> outputShape(input->shape());
    outputShape[1] *= stride_h_ * stride_w_;
    outputShape[2] /= stride_h_;
    outputShape[3] /= stride_w_;
    top[0]->Reshape(outputShape);
  }
}

template <typename Dtype>
void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  std::vector<int> shape(bottom[0]->shape());
  const int n = shape[0];
  const int c = shape[1];
  const int h = shape[2];
  const int w = shape[3];

  if (this->isFlatten_)
    reorg_cpu(const_cast<Dtype*>(bottom_data), w, h, c, n, this->stride_h_, 1, top_data);
  else
    reorg_cpu(const_cast<Dtype*>(bottom_data), w, h, c, n, this->stride_h_, 0, top_data);
}

template <typename Dtype>
void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  std::vector<int> shape(bottom[0]->shape());
  const int n = shape[0];
  const int c = shape[1];
  const int h = shape[2];
  const int w = shape[3];

  if (this->isFlatten_)
    reorg_cpu(const_cast<Dtype*>(top_diff), w, h, c, n, this->stride_h_, 0, bottom_diff);
  else
    reorg_cpu(const_cast<Dtype*>(top_diff), w, h, c, n, this->stride_h_, 1, bottom_diff);
}

#ifdef CPU_ONLY
// we do not create gpu kernel
// only for simulate Darknet's behaviour
STUB_GPU(ReorgLayer);
#endif

INSTANTIATE_CLASS(ReorgLayer);
REGISTER_LAYER_CLASS(Reorg);

} // namespace caffe