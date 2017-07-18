#ifdef USE_OPENCV

#include "caffe/layers/align_data_layer.hpp"
#include <nppi.h>



namespace caffe {
  
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
  batch->data_[0].data().get()->async_gpu_push(picPushStream_);
  
  const NppiRect dstROI = { 0, 0, align_augmenter_->width(), align_augmenter_->height() };
  unsigned char **pDst = pWarpDst_.data();
  const int dstStep = align_augmenter_->width();
  const int dstCount = expect_channels_ * align_augmenter_->width() * align_augmenter_->height();
  const Dtype scale = this->layer_param_.transform_param().scale();
  Dtype *top0_data = top[0]->mutable_gpu_data();
  
  if (expect_channels_ == 3)
  {
    
    for (int i = 0; i < batch_size - 1; i ++)
    {
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      batch->data_[i+1].data().get()->async_gpu_push(picPushStream_);
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i].gpu_data();
      const unsigned char *pSrc[3] = { pSrc0, pSrc0 + w * h, pSrc0 + 2 * w * h };
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount));
      nppiWarpAffine_8u_P3R(
        pSrc, srcSize, w, srcROI,
        pDst, dstStep, dstROI,
        transArray, NPPI_INTER_LINEAR);
      
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
    }
    {
      int i = batch_size - 1;
      
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i].gpu_data();
      const unsigned char *pSrc[3] = { pSrc0, pSrc0 + w * h, pSrc0 + 2 * w * h };
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount));
      nppiWarpAffine_8u_P3R(
        pSrc, srcSize, w, srcROI,
        pDst, dstStep, dstROI,
        transArray, NPPI_INTER_LINEAR);
      
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
    }
  }
  else
  {
    for (int i = 0; i < batch_size - 1; i ++)
    {
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      batch->data_[i+1].data().get()->async_gpu_push(picPushStream_);
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i].gpu_data();
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount)); 
      for (int c = 0; c < expect_channels_; c ++)
        nppiWarpAffine_8u_C1R(
          pSrc0 + c * w * h, srcSize, w, srcROI,
          pDst[c], dstStep, dstROI,
          transArray, NPPI_INTER_LINEAR
        );
      
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
    }
    {
      int i = batch_size - 1;
      
      // check current transfer is done and start transfer for the next
      CUDA_CHECK(cudaStreamSynchronize(picPushStream_));
      
      // perform warpaffine here
      const int w = batch->w_[i], h = batch->h_[i];
      NppiSize srcSize = { w, h };
      NppiRect srcROI = { 0, 0, w, h };
      const unsigned char *pSrc0 = (const unsigned char *)batch->data_[i].gpu_data();
      float *transData = (float *)batch->trans_[i].data;
      double transArray[2][3] = {
        {transData[0], transData[1], transData[2]},
        {transData[3], transData[4], transData[5]}
      };
      CUDA_CHECK(cudaMemset(pDst[0], 0, dstCount));
      for (int c = 0; c < expect_channels_; c ++)
        nppiWarpAffine_8u_C1R(
          pSrc0 + c * w * h, srcSize, w, srcROI,
          pDst[c], dstStep, dstROI,
          transArray, NPPI_INTER_LINEAR
        );
      
      // perform transformation here (scale and sub mean)
      align_transform_kernel<<<CAFFE_GET_BLOCKS(dstCount), CAFFE_CUDA_NUM_THREADS>>>(
        dstCount, pDst[0], data_mean_.gpu_data(), scale, top0_data + i * dstCount
      );
    }
  }
   
  
  align_float2dtype_kernel<<<CAFFE_GET_BLOCKS(batch->pts_.count()), CAFFE_CUDA_NUM_THREADS>>>(
    batch->pts_.count(), batch->pts_.gpu_data(), top[1]->mutable_gpu_data()
  );
  align_float2dtype_kernel<<<CAFFE_GET_BLOCKS(batch->pts_.count()), CAFFE_CUDA_NUM_THREADS>>>(
    batch->label_.count(), batch->label_.gpu_data(), top[2]->mutable_gpu_data()
  );
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(AlignDataLayer);

} // namespace caffe

#endif // USE_OPENCV
