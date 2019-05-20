#include "caffe/layers/reorganize_layer.hpp"

// parameter is the shape of "depth" blob
static void gen_flatten_idx(int channels, int height, int width, int stride_h, int stride_w, 
  int *input_to_output, int *output_to_input)
{
  // a.k.a depth to space
  const int output_channels = channels / (stride_h * stride_w);
  const int output_height = height * stride_h;
  const int output_width  = width  * stride_w;
  
  const int c_step = height * width;
  const int output_c_step = output_height * output_width;
  
  for (int c = 0; c < output_channels; c++)
    for (int grid_y = 0; grid_y < height; grid_y ++)
      for (int grid_x = 0; grid_x < width; grid_x ++)
      {
        // offset_x = grid_x * stride_w
        // offset_y = grid_y * stride_h
        for (int inner_y = 0; inner_y < stride_h; inner_y ++)
          for (int inner_x = 0; inner_x < stride_w; inner_x ++)
          {
            // "depth" (input)
            int c_idx_d = c + (inner_y * stride_w + inner_x) * output_channels;
            int h_idx_d = grid_y;
            int w_idx_d = grid_x;
            // "space" (output)
            int c_idx_s = c;
            int h_idx_s = grid_y * stride_h + inner_y;
            int w_idx_s = grid_x * stride_w + inner_x;

            int input_idx  = c_idx_d * c_step + h_idx_d * width + w_idx_d;
            int output_idx = c_idx_s * output_c_step + h_idx_s * output_width + w_idx_s;
            // for flatten (a.k.a depth to space)
            //    fwd: top[output_idx] = bottom[input_idx]
            //    bwd: bottom[input_idx] = top[output_idx]
            output_to_input[output_idx] = input_idx;
            input_to_output[input_idx] = output_idx;
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
  if (bottom[0]->shape() == this->current_shape_) return;

  this->current_shape_ = bottom[0]->shape();

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
  
  const int sample_count = input_c * input_h * input_w;
  std::vector<int> idx_shape(1, sample_count);

  top_to_bottom_.Reshape(idx_shape);
  bottom_to_top_.Reshape(idx_shape);
  int *top_to_bottom_data = top_to_bottom_.mutable_cpu_data();
  int *bottom_to_top_data = bottom_to_top_.mutable_cpu_data();
  if (isFlatten_)
  { // flatten, a.k.a depth to space
    std::vector<int> outputShape(input->shape());
    outputShape[1] /= stride_h_ * stride_w_;
    outputShape[2] *= stride_h_;
    outputShape[3] *= stride_w_;
    top[0]->Reshape(outputShape);
    gen_flatten_idx(input_c, input_h, input_w, 
      stride_h_, stride_w_, bottom_to_top_data, top_to_bottom_data);
  }
  else
  { // stack, a.k.a space to depth
    std::vector<int> outputShape(input->shape());
    outputShape[1] *= stride_h_ * stride_w_;
    outputShape[2] /= stride_h_;
    outputShape[3] /= stride_w_;
    top[0]->Reshape(outputShape);
    gen_flatten_idx(outputShape[1], outputShape[2], outputShape[3], 
      stride_h_, stride_w_, top_to_bottom_data, bottom_to_top_data);

  }
}

template <typename Dtype>
void ReorganizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  const int* top_to_bottom_data = top_to_bottom_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int sample_stride = this->top_to_bottom_.count();
  const int batch_size = bottom[0]->shape()[0];
  int b;
  const Dtype* src;
  Dtype* dst;
  for (b = 0, src = bottom_data, dst = top_data;
       b < batch_size;
       b++, src += sample_stride, dst += sample_stride)
    for (int i = 0; i < sample_stride; i ++)
      dst[i] = src[top_to_bottom_data[i]];
  
}

template <typename Dtype>
void ReorganizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  if (!propagate_down[0]) {
    return;
  }
  const int* bottom_to_top_data = bottom_to_top_.cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  const int sample_stride = this->top_to_bottom_.count();
  const int batch_size = bottom[0]->shape()[0];
  int b;
  const Dtype* src;
  Dtype* dst;
  for (b = 0, src = top_diff, dst = bottom_diff;
       b < batch_size;
       b++, src += sample_stride, dst += sample_stride)
    for (int i = 0; i < top[0]->count(); i ++)
      dst[i] = src[bottom_to_top_data[i]];
}

#ifdef CPU_ONLY
STUB_GPU(ReorganizeLayer);
#endif

INSTANTIATE_CLASS(ReorganizeLayer);
REGISTER_LAYER_CLASS(Reorganize);

} // namespace caffe