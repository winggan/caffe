#include "caffe/layers/random_erase_layer.hpp"

namespace caffe {
  
__device__ static void generateRect_gpu(const float* randoms, 
float area0, float area1, 
float aspect0, float aspect1,  // aspect0 = aspect_lower, aspect1 = log(aspect_upper / aspect_lower)
int W, int H, 
int &x, int &y, int &w, int &h)
{
  float s = area0 + randoms[0] * area1;
  float r = aspect0 * exp(aspect1 * randoms[1]);
  float _h = sqrt(s / r);
  float _w = r * _h;
  w = floor(_w * W) + 1;
  h = floor(_h * H) + 1;
  x = floor((W - w + 1) * randoms[2]);
  y = floor((H - h + 1) * randoms[3]);
}

__global__ static void generateRect_gpu_all(const int n, const float *all_randoms,
  float area0, float area1, 
  float aspect0, float aspect1,  // aspect0 = aspect_lower, aspect1 = log(aspect_upper / aspect_lower)
  int W, int H,
  int *all_rects)
{
   CUDA_KERNEL_LOOP(index, n) {
    int offset = index << 2;
    generateRect_gpu(all_randoms + offset, area0, area1, aspect0, aspect1, W, H,
      all_rects[index], all_rects[index + 1], all_rects[index + 2], all_rects[index + 3]);
   }
}

template <typename Dtype>
__global__ void random_erase_forward(const int count, const Dtype *bottom_data, const Dtype* noise_data,
  const int *all_rect, int N, int C, int H, int W, Dtype *top_data)
{
  CUDA_KERNEL_LOOP(index, count) {
    int idx = index;
    int w = idx % W;
    idx /= W;
    int h = idx % H;
    idx /= (H * C);
    idx <<= 2;
    int rect_x = all_rect[idx], rect_y = all_rect[idx + 1], 
        rect_w = all_rect[idx + 2], rect_h = all_rect[idx + 3];
    Dtype noise_weight = (Dtype)( 
          (w >= rect_x) & (w < rect_x + rect_w) & (h >= rect_y) & (h < rect_y + rect_h) 
    );
    top_data[index] = noise_data[index] * noise_weight + bottom_data[index] * (Dtype(1) - noise_weight);
  }
}

template <typename Dtype>
__global__ void random_erase_forward_with_trunc(const int count, const Dtype *bottom_data, const Dtype* noise_data,
  const int *all_rect, int N, int C, int H, int W, Dtype trunc_lower, Dtype trunc_upper, Dtype *top_data)
{
  CUDA_KERNEL_LOOP(index, count) {
    int idx = index;
    int w = idx % W;
    idx /= W;
    int h = idx % H;
    idx /= (H * C);
    idx <<= 2;
    int rect_x = all_rect[idx], rect_y = all_rect[idx + 1], 
        rect_w = all_rect[idx + 2], rect_h = all_rect[idx + 3];
    Dtype noise_weight = (Dtype)( 
          (w >= rect_x) & (w < rect_x + rect_w) & (h >= rect_y) & (h < rect_y + rect_h) 
    );
    Dtype val = noise_data[index] * noise_weight + bottom_data[index] * (Dtype(1) - noise_weight);
    val = val < trunc_lower ? trunc_lower : val;
    val = val > trunc_upper ? trunc_upper : val;
    top_data[index] = val;
  }
}

template <typename Dtype>
__global__ void random_erase_backward(const int count, const Dtype *top_diff, const int *all_rect, 
  int N, int C, int H, int W, Dtype *bottom_diff)
{
  CUDA_KERNEL_LOOP(index, count) {
    int idx = index;
    int w = idx % W;
    idx /= W;
    int h = idx % H;
    idx /= (H * C);
    idx <<= 2;
    int rect_x = all_rect[idx], rect_y = all_rect[idx + 1], 
        rect_w = all_rect[idx + 2], rect_h = all_rect[idx + 3];
    Dtype noise_weight = (Dtype)( 
          (w >= rect_x) & (w < rect_x + rect_w) & (h >= rect_y) & (h < rect_y + rect_h) 
    );
    bottom_diff[index] = top_diff[index] * (Dtype(1) - noise_weight);
  }
}
  
template <typename Dtype>
__global__ void trunc(const int n, Dtype *src,  Dtype lower, 
   Dtype upper) 
{
  CUDA_KERNEL_LOOP(index, n) {
    if (src[index]<lower){
      src[index]=lower;
    }
    if (src[index]>upper) {
      src[index]=upper;
    }
  }
}
template <typename Dtype>
void RandomEraseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  //if (bottom[0] != top[0])
  //  caffe_copy(count, bottom_data, top_data);
  if (this->phase_ == TRAIN)
  {
    int batch_size = bottom[0]->shape(0);
    int C = bottom[0]->shape(1);
    int H = bottom[0]->shape(2);
    int W = bottom[0]->shape(3);   
    
    noise_layer_->Forward(noise_btm_, noise_top_);
    caffe_gpu_rng_uniform<float>(randoms_.count(), 0.f, 1.f, randoms_.mutable_gpu_data());
    generateRect_gpu_all<<<CAFFE_GET_BLOCKS(batch_size), CAFFE_CUDA_NUM_THREADS>>>(
      batch_size, randoms_.gpu_data(), 
      area_lower_, area1_, aspect_lower_, aspect1_, W, H, 
      rects_.mutable_gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;
    
    if (truncate_)
      random_erase_forward_with_trunc<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, noise_.gpu_data(), rects_.gpu_data(), 
        batch_size, C, H, W, trunc_lower_, trunc_upper_, top_data
      );
    else
      random_erase_forward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, noise_.gpu_data(), rects_.gpu_data(), 
        batch_size, C, H, W, top_data
      );
    CUDA_POST_KERNEL_CHECK;

  }
}

template <typename Dtype>
void RandomEraseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
  if (propagate_down[0])
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    
    int batch_size = bottom[0]->shape(0);
    int C = bottom[0]->shape(1);
    int H = bottom[0]->shape(2);
    int W = bottom[0]->shape(3);   
    
    random_erase_backward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, rects_.gpu_data(), batch_size, C, H, W,
      bottom_diff
    );
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(RandomEraseLayer);

}
