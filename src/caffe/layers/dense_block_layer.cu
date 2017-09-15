#include "caffe/layers/dense_block_layer.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cublas_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <> 
void caffe_cublas_mul<float>(const int N, const float* a, const float* b, float* y)
{
  float one(1.);
  float zero(0.);
  CUBLAS_CHECK(cublasSsbmv(Caffe::cublas_handle(),
    CUBLAS_FILL_MODE_LOWER, N, 0,
    &one, a, 1, b, 1,
    &zero, y, 1));
}
template <>
void caffe_cublas_mul<double>(const int N, const double* a, const double* b, double* y)
{
  double one(1.);
  double zero(0.);
  CUBLAS_CHECK(cublasDsbmv(Caffe::cublas_handle(),
    CUBLAS_FILL_MODE_LOWER, N, 0,
    &one, a, 1, b, 1,
    &zero, y, 1));
}

#ifdef USE_CUDNN
template <typename Dtype>
dense_block::StaticVariable<Dtype> dense_block::StaticVariable<Dtype>::instance_;

template <typename Dtype>
dense_block::StaticVariable<Dtype>::StaticVariable() : fast_scale_fwd_op_desc_(NULL)
{
  CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&fast_scale_fwd_op_desc_));
  CUDNN_CHECK(cudnnSetOpTensorDescriptor(fast_scale_fwd_op_desc_, CUDNN_OP_TENSOR_MUL, cudnn::dataType<Dtype>::type, CUDNN_PROPAGATE_NAN));
}

template <typename Dtype>
dense_block::StaticVariable<Dtype>::~StaticVariable()
{
  if (fast_scale_fwd_op_desc_)
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(fast_scale_fwd_op_desc_));
}

template <typename Dtype>
static void reduce_nhw(cudnnHandle_t handle, 
                       Dtype alpha_, cudnnTensorDescriptor_t nchw_desc_, const Dtype* nchw_ptr_,
                       Dtype beta_, cudnnTensorDescriptor_t c_desc_, Dtype* c_ptr_)
{
  Dtype a(alpha_);
  Dtype b(beta_);
  // we assume that these 2 tensor descriptors are configured to right data type
  CUDNN_CHECK(cudnnConvolutionBackwardBias(handle,
    &a, nchw_desc_, nchw_ptr_, &b, c_desc_, c_ptr_));
}

template <typename Dtype>
void dense_block::ScaleLayerFastForward(cudnnHandle_t handle,
  cudnnTensorDescriptor_t bottom_desc, Blob<Dtype>* bottom,
  cudnnTensorDescriptor_t top_desc, Blob<Dtype> *top,
  cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<Dtype> *scale_layer)
{
  CHECK_NE(bottom, top) << "ScaleLayerFastForward dose not support in-place computation";

  Dtype one(1.);
  Dtype zero(0.);

  CUDNN_CHECK(cudnnOpTensor(handle, StaticVariable<Dtype>::get().fast_scale_fwd_op_desc(),
    &one, bottom_desc, bottom->gpu_data(),
    &one, scale_bias_desc, scale_layer->blobs()[0]->gpu_data(),
    &zero, top_desc, top->mutable_gpu_data()));

  CUDNN_CHECK(cudnnAddTensor(handle,
    &one, scale_bias_desc, scale_layer->blobs()[1]->gpu_data(),
    &one, top_desc, top->mutable_gpu_data()));

}

template <typename Dtype>
void dense_block::ScaleLayerFastBackward(cudnnHandle_t handle,
  cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<Dtype> *scale_layer,
  cudnnTensorDescriptor_t top_desc, Blob<Dtype> *top,
  cudnnTensorDescriptor_t bottom_desc, Blob<Dtype>* bottom)
{
  CHECK_NE(bottom, top) << "ScaleLayerFastForward dose not support in-place computation";
  
  Dtype one(1.);
  Dtype zero(0.);

  // gradient w.r.t bias
  reduce_nhw(handle, one, top_desc,   top->gpu_diff(),     one, scale_bias_desc, scale_layer->blobs()[1]->mutable_gpu_diff());

  // gradient w.r.t scale
  caffe_cublas_mul(bottom->count(), bottom->gpu_data(), top->gpu_diff(), bottom->mutable_gpu_diff());
  reduce_nhw(handle, one, bottom_desc, bottom->gpu_diff(), one, scale_bias_desc, scale_layer->blobs()[0]->mutable_gpu_diff());

  // gradient w.r.t bottom
  CUDNN_CHECK(cudnnOpTensor(handle, StaticVariable<Dtype>::get().fast_scale_fwd_op_desc(),
    &one, top_desc, top->gpu_diff(),
    &one, scale_bias_desc, scale_layer->blobs()[0]->gpu_data(),
    &zero, bottom_desc, bottom->mutable_gpu_diff()));
}

namespace dense_block {
  template void ScaleLayerFastForward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t bottom_desc, Blob<float>* bottom,
    cudnnTensorDescriptor_t top_desc, Blob<float> *top,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<float> *scale_layer);
  template void ScaleLayerFastForward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t bottom_desc, Blob<double>* bottom,
    cudnnTensorDescriptor_t top_desc, Blob<double> *top,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<double> *scale_layer);
  template void ScaleLayerFastBackward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<float> *scale_layer,
    cudnnTensorDescriptor_t top_desc, Blob<float> *top,
    cudnnTensorDescriptor_t bottom_desc, Blob<float>* bottom);
  template void ScaleLayerFastBackward(cudnnHandle_t handle,
    cudnnTensorDescriptor_t scale_bias_desc, ScaleLayer<double> *scale_layer,
    cudnnTensorDescriptor_t top_desc, Blob<double> *top,
    cudnnTensorDescriptor_t bottom_desc, Blob<double>* bottom);
} // namespace dense_block
#endif // USE_CUNN

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
    if (dst_ptr > src_ptr && dst_ptr - src_ptr < src_count)
      // dst_ptr is pointing within the src region [src_ptr, src_ptr + src_count]
      // directly memcpy will cause data lossing, so we copy channel by channel from back to front
    {
      const int batch = dst_ptr - src_ptr;
      int remains = src_count;
      Dtype* p_dst = dst_ptr + src_count - batch;
      const Dtype* p_src = src_ptr + src_count - batch;
      for (; remains >= batch; remains -= batch, p_dst -= batch, p_src -= batch)
        caffe_gpu_copy_async(batch, p_src, p_dst, stream);
      if (remains)
        caffe_gpu_copy_async(remains, src_ptr, dst_ptr, stream);
    }
    else
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
static void assemble_maps_gpu_adding_part(const int n, const int h, const int w, const int c0, const int c_add,
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

  for (int i = n - 1; i >= 0; i--,
    new_map_ptr -= new_stride,
    src_ptr -= src_stride,
    dst_ptr -= dst_stride,
    dst_ptr_for_new -= dst_stride)
  {
    //if (dst_ptr > src_ptr && dst_ptr - src_ptr < src_count)
    //  // dst_ptr is pointing within the src region [src_ptr, src_ptr + src_count]
    //  // directly memcpy will cause data lossing, so we copy channel by channel from back to front
    //{
    //  const int batch = dst_ptr - src_ptr;
    //  int remains = src_count;
    //  Dtype* p_dst = dst_ptr + src_count - batch;
    //  const Dtype* p_src = src_ptr + src_count - batch;
    //  for (; remains >= batch; remains -= batch, p_dst -= batch, p_src -= batch)
    //    caffe_gpu_copy_async(batch, p_src, p_dst, stream);
    //  if (remains)
    //    caffe_gpu_copy_async(remains, src_ptr, dst_ptr, stream);
    //}
    //else
    //  caffe_gpu_copy_async(src_count, src_ptr, dst_ptr, stream);
    caffe_gpu_copy_async(new_count, new_map_ptr, dst_ptr_for_new, stream);
  }

}

template <typename Dtype>
static void disassemble_maps_gpu_adding_part(const int n, const int h, const int w, const int c0, const int c_add,
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

  for (int i = 0; i < n; i++,
    out_map_ptr += out_stride,
    dst_ptr += dst_stride,
    src_ptr += src_stride,
    src_ptr_for_out += src_stride)
  {
    caffe_gpu_copy_async(out_count, src_ptr_for_out, out_map_ptr, stream);
    //caffe_gpu_copy_async(dst_count, src_ptr, dst_ptr, stream);
  }
}

template <typename Dtype>
static void assemble_maps_gpu_origin_part(const int n, const int h, const int w, const int c0, const int c_add,
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

  for (int i = n - 1; i >= 0; i--,
    new_map_ptr -= new_stride,
    src_ptr -= src_stride,
    dst_ptr -= dst_stride,
    dst_ptr_for_new -= dst_stride)
  {
    if (dst_ptr > src_ptr && dst_ptr - src_ptr < src_count)
      // dst_ptr is pointing within the src region [src_ptr, src_ptr + src_count]
      // directly memcpy will cause data lossing, so we copy channel by channel from back to front
    {
      const int batch = dst_ptr - src_ptr;
      int remains = src_count;
      Dtype* p_dst = dst_ptr + src_count - batch;
      const Dtype* p_src = src_ptr + src_count - batch;
      for (; remains >= batch; remains -= batch, p_dst -= batch, p_src -= batch)
        caffe_gpu_copy_async(batch, p_src, p_dst, stream);
      if (remains)
        caffe_gpu_copy_async(remains, src_ptr, dst_ptr, stream);
    }
    else
      caffe_gpu_copy_async(src_count, src_ptr, dst_ptr, stream);
    //caffe_gpu_copy_async(new_count, new_map_ptr, dst_ptr_for_new, stream);
  }

}

template <typename Dtype>
static void disassemble_maps_gpu_origin_part(const int n, const int h, const int w, const int c0, const int c_add,
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

  for (int i = 0; i < n; i++,
    out_map_ptr += out_stride,
    dst_ptr += dst_stride,
    src_ptr += src_stride,
    src_ptr_for_out += src_stride)
  {
    //caffe_gpu_copy_async(out_count, src_ptr_for_out, out_map_ptr, stream);
    caffe_gpu_copy_async(dst_count, src_ptr, dst_ptr, stream);
  }
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  const vector<int> shape(bottom[0]->shape());
  const int n = shape[0];
  const int k0= shape[1];
  const int h = shape[2];
  const int w = shape[3];

  CHECK_EQ(k0 + num_layers_ * growth_rate_, top[0]->shape()[1])
    << "Invalid top shape according to k0 + num_layers_ * growth_rate_";

  pre_bn_layer_->Forward(bottom, vector<Blob<Dtype>*>(1, input_lth_[0].get()));

  for (int l = 0; l < num_layers_; l++)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());

    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());
#ifdef USE_CUDNN
      dense_block::ScaleLayerFastForward(cudnn_handle_, 
        input_desc_[l], the_input_lth[0],
        input_desc_[l], the_conv3x3_inter_l[0],
        input_scale_bias_desc_[l], (ScaleLayer<Dtype>*)(bottle_scale_layers_[l].get())
      );
#else
      bottle_scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
#endif
      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

      bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv1x1_layers_[l]->Forward(the_conv3x3_inter_l, the_bottleneck_inter_l);
      bottle_bn_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
#ifdef USE_CUDNN
      caffe_copy(bottleneck_scale_tmp_[l]->count(), 
                 the_bottleneck_inter_l[0]->gpu_data(), 
                 bottleneck_scale_tmp_[l]->mutable_gpu_data());
      dense_block::ScaleLayerFastForward(cudnn_handle_,
        bottleneck_inter_desc_, bottleneck_scale_tmp_[l].get(),
        bottleneck_inter_desc_, the_bottleneck_inter_l[0],
        bottleneck_scale_bias_desc_, (ScaleLayer<Dtype>*)(scale_layers_[l].get())
      );
#else
      scale_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);
#endif
      relu_layers_[l]->Forward(the_bottleneck_inter_l, the_bottleneck_inter_l);

      conv3x3_layers_[l]->Forward(the_bottleneck_inter_l, the_output_lth);

    }
    else
    {
#ifdef USE_CUDNN
      dense_block::ScaleLayerFastForward(cudnn_handle_,
        input_desc_[l], the_input_lth[0],
        input_desc_[l], the_conv3x3_inter_l[0],
        input_scale_bias_desc_[l], (ScaleLayer<Dtype>*)(scale_layers_[l].get())
      );
#else
      scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
#endif
      // (in gpu) async "assemble" (original part) can start here to prepare for next conv block
      assemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
        maps_diff_.mutable_gpu_data(), (const Dtype*)NULL /*output_lth_[l]->gpu_data()*/, dataCopyStream_);

      relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);

      conv3x3_layers_[l]->Forward(the_conv3x3_inter_l, the_output_lth);

    }

    if (use_dropout_)
    {
      dropout_layers_[l]->Forward(the_output_lth, the_output_lth);
    }

    bn_layers_[l]->Forward(the_output_lth, the_output_lth);

    // (in gpu) start async "assemble" (adding part) for this conv block
    //assemble_maps(n, h, w, k0 + l * growth_rate_, growth_rate_,
    //  maps_diff_.mutable_cpu_data(), output_lth_[l]->cpu_data());
    assemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), output_lth_[l]->gpu_data(), dataCopyStream_);

    // (in gpu) synchronize "assemble" here so we can start next conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
  }

  // maps_diff_.data() store the output data (before post_scale_layer_), 
  //which will be used in backward of the input scale of each conv block
#ifdef USE_CUDNN
  dense_block::ScaleLayerFastForward(cudnn_handle_,
    final_output_desc_, &maps_diff_,
    final_output_desc_, top[0],
    scale_bias_desc_, (ScaleLayer<Dtype>*)(post_scale_layer_.get())
  );
#else
  post_scale_layer_->Forward(vector<Blob<Dtype>*>(1, &maps_diff_), top);
#endif
  post_relu_layer_->Forward(top, top);
}

template <typename Dtype>
void DenseBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  const vector<int> shape(top[0]->shape());
  const int n = shape[0];
  const int k0= shape[1] - num_layers_ * growth_rate_;
  const int h = shape[2];
  const int w = shape[3];
  
  CHECK_EQ(k0, bottom[0]->shape()[1])
    << "Invalid top shape according to k0 + num_layers_ * growth_rate_";
    
  post_relu_layer_->Backward(top, need_propagate_down_, top);
  post_scale_layer_->Backward(top, need_propagate_down_, vector<Blob<Dtype>*>(1, &maps_diff_));
  
  for (int l = num_layers_ - 1; l >= 0 ; l --)
  {
    vector<Blob<Dtype>*> the_input_lth(1, input_lth_[l].get());
    vector<Blob<Dtype>*> the_conv3x3_inter_l(1, conv3x3_inter_[l].get());
    vector<Blob<Dtype>*> the_output_lth(1, output_lth_[l].get());
    
    // diff and data, each use a individual stream
    // (in gpu) start async "disassemble" (adding part) for this conv block
    disassemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_, 
                     maps_diff_.mutable_gpu_data(), output_lth_[l]->mutable_gpu_data(), dataCopyStream_);
    disassemble_maps_gpu_adding_part(n, h, w, k0 + l * growth_rate_, growth_rate_, 
                     maps_diff_.mutable_gpu_diff(), output_lth_[l]->mutable_gpu_diff(), diffCopyStream_);
    
    // (in gpu) synchronize "disassemble" (adding part) here so we can start the 
    // Backward for the conv block
    CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
    CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));

    // (in gpu) start async "disassemble" (original part) to prepare for Backward of 
    // earlier conv in the conv block
    disassemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_data(), (Dtype*)NULL /*output_lth_[l]->mutable_gpu_data()*/, dataCopyStream_);
    disassemble_maps_gpu_origin_part(n, h, w, k0 + l * growth_rate_, growth_rate_,
      maps_diff_.mutable_gpu_diff(), (Dtype*)NULL /*output_lth_[l]->mutable_gpu_diff()*/, diffCopyStream_);
    
    bn_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_output_lth);
    
    if (use_dropout_)
    {
      dropout_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_output_lth);
    }
    
    if (use_bottleneck_)
    {
      vector<Blob<Dtype>*> the_bottleneck_inter_l(1, bottleneck_inter_[l].get());
      
      conv3x3_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_bottleneck_inter_l);
      
      relu_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      scale_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      bottle_bn_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_bottleneck_inter_l);
      
      // (in gpu) synchronize "disassemble" (original part) so we can continue the preparation 
      // for Backward of conv3x3 in the conv block
      CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
      CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));
      
      // re-calculate the bottom_data of conv1x1 from bottle_scale_layers_[l] and bottle_relu_layers_[l]
      bottle_scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      bottle_relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      
      conv1x1_layers_[l]->Backward(the_bottleneck_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      
      bottle_relu_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      bottle_scale_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_input_lth);
    }
    else
    {
      // (in gpu) synchronize "disassemble" (original part) so we can continue the preparation 
      // for Backward of conv3x3 in the conv block
      CUDA_CHECK(cudaStreamSynchronize(dataCopyStream_));
      CUDA_CHECK(cudaStreamSynchronize(diffCopyStream_));
      
      // re-calculate the bottom_data of conv3x3 from scale_layers_[l] and relu_layers_[l]
      scale_layers_[l]->Forward(the_input_lth, the_conv3x3_inter_l);
      relu_layers_[l]->Forward(the_conv3x3_inter_l, the_conv3x3_inter_l);
      
      conv3x3_layers_[l]->Backward(the_output_lth, need_propagate_down_, the_conv3x3_inter_l);
      
      relu_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_conv3x3_inter_l);
      scale_layers_[l]->Backward(the_conv3x3_inter_l, need_propagate_down_, the_input_lth);
    }
    
    { // add the diff together before continue
      const int count = input_lth_[l]->count();
      Dtype* target_ptr;
      const Dtype* adding_in_ptr;
      if (l > 0)
      {
        target_ptr = maps_diff_.mutable_gpu_diff(); 
        adding_in_ptr = tmp_diff_.gpu_diff(); // diff of input_lth_[l]
      }
      else
      {
        // for the first conv block, store the sum of diff in tmp_diff_.diff (input_lth_[0].diff)
        // because pre_bn_layer_ treat input_lth_[0] as the top blob.
        target_ptr = tmp_diff_.mutable_gpu_diff(); // diff of input_lth_[l]
        adding_in_ptr = maps_diff_.gpu_diff();
      }
      
      // in gpu caffe_gpu_axpy is used
      //caffe_axpy(count, Dtype(1.), adding_in_ptr, target_ptr);
      caffe_gpu_axpy(count, Dtype(1.), adding_in_ptr, target_ptr);
    }
  }
  
  pre_bn_layer_->Backward(vector<Blob<Dtype>*>(1, input_lth_[0].get()), need_propagate_down_, bottom);
    
}

INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);

} // namespace caffe