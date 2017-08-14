#ifdef USE_OPENCV

#include "caffe/layers/align_data_layer.hpp"
#include <nppi.h>


#define NPP_CHECK(condition) \
  do { \
    NppStatus status = condition; \
    CHECK_EQ(status, NPP_NO_ERROR) << " " \
      << "npp error code = " << status; \
  } while (0)
    
namespace caffe {
  
#if ALING_DATA_USE_REMAP

#define NUM_TH_2D 16
inline static int GET_BLOCKS_2D(const int N) {
  return (N + NUM_TH_2D - 1) / NUM_TH_2D;
}

__global__ void calculate_map_kernel(const int height, const int width, 
   const float M0, const float M1, const float M2, const float M3, const float M4, const float M5,
   float *xMap, float *yMap)
{
  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y)
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x)
    {
      xMap[x + y * width] = (float)x * M0 + (float)y * M1 + M2;
      yMap[x + y * width] = (float)x * M3 + (float)y * M4 + M5;
    }
}

void static my_warp_affine_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u * pDst[3], int nDstStep, NppiRect oDstROI,
                               const double aCoeffs[2][3], int eInterpolation, float *xMap, float *yMap)
{
  float M[6];
  { // inverse the transform matrix
    M[0] = (float)aCoeffs[0][0];
    M[1] = (float)aCoeffs[0][1];
    M[2] = (float)aCoeffs[0][2];
    M[3] = (float)aCoeffs[1][0];
    M[4] = (float)aCoeffs[1][1];
    M[5] = (float)aCoeffs[1][2];
    {
      float D = M[0] * M[4] - M[1] * M[3];
      D = D != 0 ? 1.f / D : 0;
      float A11 = M[4] * D, A22 = M[0] * D;
      M[0] = A11; M[1] *= -D;
      M[3] *= -D; M[4] = A22;
      float b1 = -M[0] * M[2] - M[1] * M[5];
      float b2 = -M[3] * M[2] - M[4] * M[5];
      M[2] = b1; M[5] = b2;
    }
  }
  dim3 num_thread(NUM_TH_2D, NUM_TH_2D);
  dim3 num_block(GET_BLOCKS_2D(oDstROI.width), GET_BLOCKS_2D(oDstROI.height));
  calculate_map_kernel<<<num_block, num_thread>>>(oDstROI.height, oDstROI.width, M[0], M[1], M[2], M[3], M[4], M[5], xMap, yMap);
  CUDA_POST_KERNEL_CHECK;
  const NppiSize oDstSizeROI = { oDstROI.width , oDstROI.height };
  NPP_CHECK(nppiRemap_8u_P3R(
    pSrc, oSrcSize, nSrcStep, oSrcROI,
    xMap, oDstROI.width * sizeof(float), yMap, oDstROI.width * sizeof(float),
    pDst, nDstStep, oDstSizeROI, eInterpolation)
  );
}

void static my_warp_affine_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                               const double aCoeffs[2][3], int eInterpolation, float *xMap, float *yMap)
{
  float M[6];
  { // inverse the transform matrix
    M[0] = (float)aCoeffs[0][0];
    M[1] = (float)aCoeffs[0][1];
    M[2] = (float)aCoeffs[0][2];
    M[3] = (float)aCoeffs[1][0];
    M[4] = (float)aCoeffs[1][1];
    M[5] = (float)aCoeffs[1][2];
    {
      float D = M[0] * M[4] - M[1] * M[3];
      D = D != 0 ? 1.f / D : 0;
      float A11 = M[4] * D, A22 = M[0] * D;
      M[0] = A11; M[1] *= -D;
      M[3] *= -D; M[4] = A22;
      float b1 = -M[0] * M[2] - M[1] * M[5];
      float b2 = -M[3] * M[2] - M[4] * M[5];
      M[2] = b1; M[5] = b2;
    }
  }
  dim3 num_thread(NUM_TH_2D, NUM_TH_2D);
  dim3 num_block(GET_BLOCKS_2D(oDstROI.width), GET_BLOCKS_2D(oDstROI.height));
  calculate_map_kernel<<<num_block, num_thread >>>(oDstROI.height, oDstROI.width, M[0], M[1], M[2], M[3], M[4], M[5], xMap, yMap);
  CUDA_POST_KERNEL_CHECK;
  const NppiSize oDstSizeROI = { oDstROI.width , oDstROI.height };
  NPP_CHECK(nppiRemap_8u_C1R(
    pSrc, oSrcSize, nSrcStep, oSrcROI,
    xMap, oDstROI.width * sizeof(float), yMap, oDstROI.width * sizeof(float),
    pDst, nDstStep, oDstSizeROI, eInterpolation)
  );
}

#endif // ALING_DATA_USE_REMAP
  
template <typename Dtype>
__global__ void align_transform_kernel(const int n, const unsigned char *src, const Dtype* data_mean, 
  const Dtype scale, Dtype* dst) 
{
  CUDA_KERNEL_LOOP(index, n) {
    dst[index] = ( (Dtype)src[index] - data_mean[index] ) * scale;
  }
}

template <typename Dtype>
__global__ void align_float2dtype_kernel(const int n, const float *src, Dtype* dst) 
{
  CUDA_KERNEL_LOOP(index, n) {
    dst[index] = (Dtype)src[index];
  }
}
  
template <typename Dtype>
void AlignDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  AlignBatch* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  
  int batch_size = (int)batch->data_.size();
  batch->data_[0]->data().get()->async_gpu_push(picPushStream_);
  
  const NppiRect dstROI = { 0, 0, align_augmenter_->width(), align_augmenter_->height() };
  unsigned char **pDst = pWarpDst_.data();
  const int dstStep = align_augmenter_->width();
  const int dstCount = expect_channels_ * align_augmenter_->width() * align_augmenter_->height();
  const Dtype scale = this->layer_param_.transform_param().scale();
  Dtype *top0_data = top[0]->mutable_gpu_data();
  
#if ALING_DATA_USE_REMAP
  float *xMap = this->xyMap_.mutable_gpu_data();
  float *yMap = xMap + dstROI.height * dstROI.width;
#endif // ALING_DATA_USE_REMAP

  if (expect_channels_ == 3)
  {
    
    for (int i = 0; i < batch_size - 1; i ++)
    {
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      batch->data_[i+1]->data().get()->async_gpu_push(picPushStream_);
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i]->gpu_data();
      const unsigned char *pSrc[3] = { pSrc0, pSrc0 + w * h, pSrc0 + 2 * w * h };
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount));
#if ALING_DATA_USE_REMAP
      my_warp_affine_P3R(
        pSrc, srcSize, w, srcROI,
        pDst, dstStep, dstROI,
        transArray, NPPI_INTER_LINEAR, xMap, yMap
      );
#else // ALING_DATA_USE_REMAP
      NPP_CHECK( nppiWarpAffine_8u_P3R(
        pSrc, srcSize, w, srcROI,
        pDst, dstStep, dstROI,
        transArray, NPPI_INTER_LINEAR)
      );
#endif // ALING_DATA_USE_REMAP
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
      CUDA_POST_KERNEL_CHECK;
    }
    {
      int i = batch_size - 1;
      
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i]->gpu_data();
      const unsigned char *pSrc[3] = { pSrc0, pSrc0 + w * h, pSrc0 + 2 * w * h };
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount));
#if ALING_DATA_USE_REMAP
      my_warp_affine_P3R(
        pSrc, srcSize, w, srcROI,
        pDst, dstStep, dstROI,
        transArray, NPPI_INTER_LINEAR, xMap, yMap
      );
#else // ALING_DATA_USE_REMAP
      NPP_CHECK( nppiWarpAffine_8u_P3R(
        pSrc, srcSize, w, srcROI,
        pDst, dstStep, dstROI,
        transArray, NPPI_INTER_LINEAR)
      );
#endif // ALING_DATA_USE_REMAP
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
      CUDA_POST_KERNEL_CHECK;
    }
  }
  else
  {
    for (int i = 0; i < batch_size - 1; i ++)
    {
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      batch->data_[i+1]->data().get()->async_gpu_push(picPushStream_);
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i]->gpu_data();
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount)); 
      for (int c = 0; c < expect_channels_; c++)
#if ALING_DATA_USE_REMAP
        my_warp_affine_C1R(
          pSrc0 + c * w * h, srcSize, w, srcROI,
          pDst[c], dstStep, dstROI,
          transArray, NPPI_INTER_LINEAR, xMap, yMap
        );
#else // ALING_DATA_USE_REMAP
        NPP_CHECK( nppiWarpAffine_8u_C1R(
          pSrc0 + c * w * h, srcSize, w, srcROI,
          pDst[c], dstStep, dstROI,
          transArray, NPPI_INTER_LINEAR
        ) );
#endif // ALING_DATA_USE_REMAP 
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
      CUDA_POST_KERNEL_CHECK;
    }
    {
      int i = batch_size - 1;
      
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i]->gpu_data();
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount));
      for (int c = 0; c < expect_channels_; c++)
#if ALING_DATA_USE_REMAP
        my_warp_affine_C1R(
          pSrc0 + c * w * h, srcSize, w, srcROI,
          pDst[c], dstStep, dstROI,
          transArray, NPPI_INTER_LINEAR, xMap, yMap
        );
#else // ALING_DATA_USE_REMAP
        NPP_CHECK( nppiWarpAffine_8u_C1R(
          pSrc0 + c * w * h, srcSize, w, srcROI,
          pDst[c], dstStep, dstROI,
          transArray, NPPI_INTER_LINEAR
        ) );
#endif // ALING_DATA_USE_REMAP
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
      CUDA_POST_KERNEL_CHECK;
    }
  }
   
  
  align_float2dtype_kernel<<<CAFFE_GET_BLOCKS(batch->pts_.count()), CAFFE_CUDA_NUM_THREADS>>>(
    batch->pts_.count(), batch->pts_.gpu_data(), top[1]->mutable_gpu_data()
  );
  CUDA_POST_KERNEL_CHECK;
  align_float2dtype_kernel<<<CAFFE_GET_BLOCKS(batch->label_.count()), CAFFE_CUDA_NUM_THREADS>>>(
    batch->label_.count(), batch->label_.gpu_data(), top[2]->mutable_gpu_data()
  );
  CUDA_POST_KERNEL_CHECK;
  if (top.size() > 3)
  {
    align_float2dtype_kernel<<<CAFFE_GET_BLOCKS(batch->trans_blob_.count()), CAFFE_CUDA_NUM_THREADS>>>(
      batch->trans_blob_.count(), batch->trans_blob_.gpu_data(), top[3]->mutable_gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;
  }
  if (top.size() > 4)
  {
    align_float2dtype_kernel<<<CAFFE_GET_BLOCKS(batch->extra_data_.count()), CAFFE_CUDA_NUM_THREADS>>>(
      batch->extra_data_.count(), batch->extra_data_.gpu_data(), top[4]->mutable_gpu_data()
    );
    CUDA_POST_KERNEL_CHECK;
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(AlignDataLayer);

} // namespace caffe

#endif // USE_OPENCV
