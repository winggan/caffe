#include "caffe/layers/random_erase_layer.hpp"
#include <omp.h>
#include <cmath>

// aspect is generated from T(x) = a * exp(A * x), where A = ln(b / a), x ~ U[0,1]

namespace caffe {

// ~0.15 ms / 1024 rects, on i5-6600@3.7GHz
inline static void generateRect(const float* randoms, 
  float area0, float area1, 
  float aspect0, float aspect1,  // aspect0 = aspect_lower, aspect1 = log(aspect_upper / aspect_lower)
  int W, int H, 
  int &x, int &y, int &w, int &h)
{
  float s = area0 + randoms[0] * area1;
  float r = aspect0 * expf(aspect1 * randoms[1]);
  float _h = sqrtf(s / r);
  float _w = r * _h;
  w = floor(_w * W) + 1;
  h = floor(_h * H) + 1;
  x = floor((W - w + 1) * randoms[2]);
  y = floor((H - h + 1) * randoms[3]);
}


template <typename Dtype>
void RandomEraseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const RandomEraseParameter &param =  this-> layer_param_.random_erase_param();
  area_upper_   = param.area_ratio_upper();
  area_lower_   = param.area_ratio_lower();
  
  CHECK_GE(area_lower_, 0.f);
  CHECK_GE(area_upper_, area_lower_);
  CHECK_GE(1.f, area_upper_);
  
  if (!param.has_aspect_ratio_lower() && !param.has_aspect_ratio_upper())
    LOG(FATAL) << "you should specify at least one of aspect_ratio_lower and aspect_ratio_upper";    
  else if (param.has_aspect_ratio_lower() && param.has_aspect_ratio_upper())
  {
    CHECK_LE(param.has_aspect_ratio_lower(), param.has_aspect_ratio_upper())
      << "aspect lower bound should less than or equal to upper bound";
    aspect_upper_ = param.aspect_ratio_upper();
    aspect_lower_ = param.aspect_ratio_lower();
  }
  else if (param.has_aspect_ratio_lower())
  {
    CHECK_LE(param.aspect_ratio_lower(), 1.f) 
      << "aspect lower bound should be less than 1 if it is the only specified";
    aspect_lower_ = param.aspect_ratio_lower();
    aspect_upper_ = 1.f / aspect_lower_;
  }
  else if (param.has_aspect_ratio_upper())
  {
    CHECK_GE(param.aspect_ratio_upper(), 1.f)
      << "aspect upper bound should be larger than 1 if it is the only specified";
    aspect_upper_ = param.aspect_ratio_upper();
    aspect_lower_= 1.f / aspect_upper_;
  }
  CHECK_GT(aspect_lower_, 0);
  CHECK_GT(aspect_upper_, 0);
  
  truncate_     = param.truncate();
  trunc_upper_  = param.trunc_upper();
  trunc_lower_  = param.trunc_lower();
  if (truncate_)
    CHECK_GT(trunc_upper_, trunc_lower_);
  
  CHECK_EQ(bottom[0]->shape().size(),4);
  
  {
    LayerParameter noise_layer_param(this->layer_param_);
    noise_layer_param.set_type("Noise");
    noise_layer_param.clear_random_erase_param();
    noise_layer_param.clear_param();
    noise_layer_param.clear_blobs();
    noise_layer_ = LayerRegistry<Dtype>::CreateLayer(noise_layer_param);
  }
  
  noise_btm_.push_back(&all_zeros_);
  noise_top_.push_back(&noise_);
  noise_layer_->LayerSetUp(noise_btm_, noise_top_);

  {
    area1_ = area_upper_ - area_lower_;
    aspect1_ = log(aspect_upper_ / aspect_lower_);
  }
}

template <typename Dtype>
void RandomEraseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  //if (noise_.count() < bottom[0]->count())
  {
    noise_.ReshapeLike(*bottom[0]);
    all_zeros_.ReshapeLike(*bottom[0]);
    //noise_layer_->Reshape(noise_btm_, noise_top_);
  }
  //if (rects_.shape()[0] < bottom[0]->shape()[0])
  {
    std::vector<int> shape(1, bottom[0]->shape()[0]);
    shape.push_back(4);
    rects_.Reshape(shape);
    randoms_.Reshape(shape);
  }
}

template <typename Dtype>
void RandomEraseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (bottom[0] != top[0])
    caffe_copy(count, bottom_data, top_data);
  if (this->phase_ == TRAIN) // only apply the mask in train phase
  {
    int batch_size = bottom[0]->shape(0);
    int C = bottom[0]->shape(1);
    int H = bottom[0]->shape(2);
    int W = bottom[0]->shape(3);
    int c_stride = H * W;
    int sample_stride = c_stride * C;
    
    caffe_set(count, Dtype(0), all_zeros_.mutable_cpu_data()); 
    noise_layer_->Forward(noise_btm_, noise_top_);
    caffe_rng_uniform<float>(randoms_.count(), 0.f, 1.f, randoms_.mutable_cpu_data());
    const float* randoms = randoms_.cpu_data();
    int *rect_data = rects_.mutable_cpu_data();
    const Dtype *noise_data = noise_.cpu_data();
    for (int i=0; i < batch_size; i++) 
    {
      int *rect_i = rect_data + (i << 2);
      generateRect(randoms + (i << 2), 
        area_lower_, area1_, aspect_lower_, aspect1_,
        W, H, 
        rect_i[0], rect_i[1], rect_i[2], rect_i[3]);
       
      for (int c = 0; c < C; c ++)
      {
        int channel_start = i * sample_stride + c * c_stride
            + rect_i[1] * W + rect_i[0];
            
        for (int offset = channel_start, h = 0; h < rect_i[3]; h ++, offset += W)
          caffe_copy(rect_i[2], noise_data + offset, top_data + offset);
          
        if (truncate_)
          for (int offset = channel_start, h = 0; h < rect_i[3]; h ++, offset += W)
            for (int k = 0; k < rect_i[2]; k ++)
            {
              Dtype val = top_data[offset + k];
              val = val > trunc_upper_ ? trunc_upper_ : val;
              val = val < trunc_lower_ ? trunc_lower_ : val;
              top_data[offset + k] = val;
            }
      }
      
    }
  }
  return ;
}

template <typename Dtype>
void RandomEraseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
    
    int batch_size = bottom[0]->shape(0);
    int C = bottom[0]->shape(1);
    int H = bottom[0]->shape(2);
    int W = bottom[0]->shape(3);
    int c_stride = H * W;
    int sample_stride = c_stride * C;
    
    const int *rect_data = rects_.mutable_cpu_data();
      
    for (int i=0; i < batch_size; i++) 
    {
      const int *rect_i = rect_data + (i << 2);

      for (int c = 0; c < C; c ++)
      {
        int channel_start = i * sample_stride + c * c_stride
            + rect_i[1] * W + rect_i[0];
            
        for (int offset = channel_start, h = 0; h < rect_i[3]; h ++, offset += W)
          caffe_set(rect_i[2], Dtype(0), bottom_diff + offset);
          
      }
      
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RandomEraseLayer);
#endif

INSTANTIATE_CLASS(RandomEraseLayer);
REGISTER_LAYER_CLASS(RandomErase);

} // namespace caffe
