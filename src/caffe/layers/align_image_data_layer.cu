#ifdef USE_OPENCV

#include "caffe/layers/align_image_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  
template <typename Dtype>
void AlignImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // arrangement: [batch_size] labels, [batch_size * (num_points * 2)] pts
    std::vector<int> top_shape = batch->label_.shape();
    top_shape[1] --;
    top[1]->Reshape(top_shape);
    // Copy the pts.
    caffe_copy(batch->label_.count() - top_shape[0], 
        batch->label_.gpu_data() + top_shape[0],
        top[1]->mutable_gpu_data());
        
    top_shape.pop_back();
    top[2]->Reshape(top_shape);
    caffe_copy(top_shape[0], 
      batch->label_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(AlignImageDataLayer);

} // namespace caffe

#endif // USE_OPENCV

