#include "caffe/layers/reorganize_layer.hpp"

static void gen_flatten_idx(int channels, int height, int width, int stride_h, int stride_w, 
  int *forward_idx, int *backward_idx)
{
  const int output_channels = channels / (stride_h * stride_w);
  const int output_height = height * stride_h;
  const int output_width  = width  * stride_w;
  
  const int c_step = height * width;
  const int output_c_step = output_height * output_width;
  
  for (int c = 0, offset_c = 0; offset_c < channels; c++, offset_c += stride_h * stride_w)
    for (int grid_y = 0, offset_y = 0; grid_y < height; grid_y ++, offset_y += stride_h)
      for (int grid_x = 0, offset_x = 0; grid_x < width; grid_x ++, offset_x += stride_w)
      {
        for (int inner_y = 0; inner_y < stride_h; inner_y ++)
          for (int inner_x = 0; inner_x < stride_w; inner_x ++)
          {
            int input_idx  = (offset_c + inner_y * stride_w + inner_x) * c_step + grid_y * width + grid_x;
            int output_idx = c * output_c_step + (offset_y + inner_y) * output_width + (offset_x + inner_x);
            forward_idx[output_idx] = input_idx;
            backward_idx[input_idx] = output_idx;
          }
        
      }
}

namespace caffe {
  
template <typename Dtype>
void ReorganizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
}

template <typename Dtype>
void ReorganizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  
  const int input_c = input->shape()[1];
  const int input_h = input->shape()[2];
  const int input_w = input->shape()[3];
  
  backward_idx_.Reshape(input->shape());
  if (isFlatten_)
  {
    std::vector<int> outputShape(input->shape());
    outputShape[1] /= stride_h_ * stride_w_;
    outputShape[2] *= stride_h_;
    outputShape[3] *= stride_w_;
    top[0]->Reshape(outputShape);
    forward_idx_.Reshape(outputShape);
    int *backward_idx_data = backward_idx_.mutable_cpu_data();
    int *forward_idx_data = forward_idx_.mutable_cpu_data();
    gen_flatten_idx(input_c, input_h, input_w, 
      stride_h_, stride_w_, forward_idx_data, backward_idx_data);
    //const int output_c_step = outputShape[2] * outputShape[3];
    //const int input_c_step  = input_h * input_w;
    //const int output_w = outputShape[3];
    //for (int offset_c = 0, output_c = 0; offset_c < input_c; offset_c += stride_h_ * stride_w_, output_c ++)
    //  for (int grid_y = 0, offset_y = 0; grid_y < input_h; grid_y ++, offset_y += stride_h_)
    //    for (int grid_x = 0, offset_x = 0; grid_x < input_w; grid_x ++, offset_x += stride_w_)
    //    {
    //      for (int inner_y = 0; inner_y < stride_h_; inner_y ++)
    //        for (int inner_x = 0; inner_x < stride_w_; inner_x ++)
    //        {
    //          int output_idx = output_c * output_c_step + (offset_y + inner_y) * output_w + (offset_x + inner_x);
    //          int input_idx  = (offset_c + inner_y * stride_w_ + inner_x) * input_c_step + grid_y * input_w + grid_x;
    //          forward_idx_data[output_idx] = input_idx;
    //          backward_idx_data[input_idx] = output_idx;
    //        }
    //    }
  }
  else
  { // stacking
    std::vector<int> outputShape(input->shape());
    outputShape[1] *= stride_h_ * stride_w_;
    outputShape[2] /= stride_h_;
    outputShape[3] /= stride_w_;
    top[0]->Reshape(outputShape);
    forward_idx_.Reshape(outputShape);
    int *forward_idx_data = forward_idx_.mutable_cpu_data();
    int *backward_idx_data = backward_idx_.mutable_cpu_data();
    gen_flatten_idx(outputShape[1], outputShape[2], outputShape[3], 
      stride_h_, stride_w_, backward_idx_data, forward_idx_data);
    //const int output_c_step = outputShape[2] * outputShape[3];
    //const int input_c_step  = input_h * input_w;
    //const int output_w = outputShape[3];
    //for (int offset_c = 0, c = 0; c < input_c; offset_c += stride_h_ * stride_w_, c ++)
    //  for (int grid_y = 0, offset_y = 0; offset_y < input_h; grid_y ++, offset_y += stride_h_)
    //    for (int grid_x = 0, offset_x = 0; offset_x < input_w; grid_x ++, offset_x += stride_w_)
    //    {
    //      for (int inner_y = 0; inner_y < stride_h_; inner_y ++)
    //        for (int inner_x = 0; inner_x < stride_w_; inner_x ++)
    //        {
    //          int input_idx = c * input_c_step + (grid_y + inner_y) * input_w + (grid_x + inner_x);
    //          int output_idx = (offset_c + inner_y * stride_w_ + inner_x) * output_c_step + grid_y * output_w + grid_x;
    //          forward_idx_data[output_idx] = input_idx;
    //          backward_idx_data[input_idx] = output_idx;
    //        }
    //    }
  }
}

template <typename Dtype>
void ReorganizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const int* forward_idx_data = forward_idx_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->count(); i ++)
    top_data[i] = bottom_data[forward_idx_data[i]];
  
}

template <typename Dtype>
void ReorganizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0]) {
    return;
  }
  const int* backward_idx_data = backward_idx_.cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < top[0]->count(); i ++)
    bottom_diff[i] = top_diff[backward_idx_data[i]];
}

#ifdef CPU_ONLY
STUB_GPU(ReorganizeLayer);
#endif

INSTANTIATE_CLASS(ReorganizeLayer);
REGISTER_LAYER_CLASS(Reorganize);

} // namespace caffe