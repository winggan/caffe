#ifndef CAFFE_ALIGN_DATA_LAYER_HPP_
#define CAFFE_ALIGN_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/align_augmenter.hpp"

//#if (CUDART_VERSION <= 7050) // if cuda is version 7.5 or older
#define ALING_DATA_USE_REMAP 1
//#else
//#define ALING_DATA_USE_REMAP 0
//#endif

namespace caffe {

class AlignBatch {
 public:
  std::vector<int> w_, h_;
  std::vector< shared_ptr<Blob<float> > > data_;
  std::vector<cv::Mat> trans_;
  Blob<float> pts_;
  Blob<float> label_; 
};
  
template <typename Dtype>
class AlignDataLayer : public BaseDataLayer<Dtype>, public InternalThread
{
 public:
  explicit AlignDataLayer(const LayerParameter& param);
  virtual ~AlignDataLayer();
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AlignData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
 protected:
  void InternalThreadEntry();
  void load_batch(AlignBatch& batch);
  
  std::string loaderKey_;
  
  AlignAugmentationParameter augmentation_param_;
  shared_ptr<AlignAugmenter<Dtype> > align_augmenter_;
  
  vector<shared_ptr<AlignBatch> > prefetch_;
  BlockingQueue<AlignBatch*> prefetch_free_, prefetch_full_;
  int expect_channels_;
  Blob<float> warpBuffer_;
  Blob<Dtype> data_mean_;
  std::vector<unsigned char *> pWarpDst_;
  
#if ALING_DATA_USE_REMAP
  Blob<float> xyMap_; // 2 x height x width
#endif // ALING_DATA_USE_REMAP
  
  cudaStream_t picPushStream_;
}; // AlignDataLayer
  
} // namespace caffe

#endif // CAFFE_ALIGN_DATA_LAYER_HPP_

