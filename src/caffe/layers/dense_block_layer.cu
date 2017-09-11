#include "caffe/layers/dense_block_layer.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

template <typename Dtype>
inline static void caffe_gpu_copy_async(const int N, const Dtype* X, Dtype* Y, const cudaStream_t& stream) {
  if (X != Y && Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpyAsync(Y, X, sizeof(Dtype) * N, cudaMemcpyDeviceToDevice, stream));
#else
      NO_GPU;
#endif
  }
}

template <typename Dtype>
static void assemble_maps_gpu(const int n, const int h, const int w, const int c0, const int c_add,
                              Dtype* dst, const Dtype* new_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps BEFORE assemble
  // c_add = #feature-maps to be added
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c0 * c_stride;
  const int dst_stride = c1 * c_stride;
  const int new_stride = c_add * c_stride;
  
  const Dtype* new_map_ptr = new_map + (n - 1) * new_stride;
  const Dtype *src_ptr = dst + (n - 1) * src_stride;
  Dtype *dst_ptr = dst + (n - 1) * dst_stride;
  Dtype *dst_ptr_for_new = dst_ptr + src_stride;
  
  const int src_count = c0 * c_stride;
  const int new_count = c_add * c_stride;
  
  for (int i = n - 1; i >= 0; i --, 
    new_map_ptr -= new_stride, 
    src_ptr     -= src_stride,
    dst_ptr     -= dst_stride,
    dst_ptr_for_new -= dst_stride)
  {
    caffe_gpu_copy_async(src_count, src_ptr, dst_ptr, stream);
    caffe_gpu_copy_async(new_count, new_map_ptr, dst_ptr_for_new, stream);  
  }
  
}

template <typename Dtype>
static void disassemble_maps_gpu(const int n, const int h, const int w, const int c0, const int c_add,
                                 Dtype* src, Dtype* out_map, const cudaStream_t& stream)
{
  // c0 = #feature-maps AFTER disassemble
  // c_add = #feature-maps in out_map
  const int c1 = c0 + c_add;
  const int c_stride = h * w;
  const int src_stride = c1 * c_stride;
  const int dst_stride = c0 * c_stride;
  const int out_stride = c_add * c_stride;
  
  Dtype* out_map_ptr = out_map;
  Dtype *dst_ptr = src;
  const Dtype *src_ptr = src;
  const Dtype *src_ptr_for_out = src_ptr + dst_stride;
  
  const int dst_count = c0 * c_stride;
  const int out_count = c_add * c_stride;
  
  for (int i = 0; i < n; i ++,
    out_map_ptr += out_stride,
    dst_ptr     += dst_stride,
    src_ptr     += src_stride,
    src_ptr_for_out += src_stride)
  {
    caffe_gpu_copy_async(out_count, src_ptr_for_out, out_map_ptr, stream);
    caffe_gpu_copy_async(dst_count, src_ptr, dst_ptr, stream);
  }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

} // namespace caffe