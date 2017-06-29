#ifndef ALIGN_IMAGE_DATA_LAYER_HPP_
#define ALIGN_IMAGE_DATA_LAYER_HPP_

#ifdef USE_OPENCV

#include <vector>
#include <string>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/align_augmenter.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define CAFFE_ALIGN_AUGMENT_WORKER_

namespace caffe {
  
 /**
  * @brief Provides data to the Net from image files together with pts files
  * designed for face alignment training and maybe other point regression task
  */
template <typename Dtype>
class AlignImageDataLayer : public BasePrefetchingDataLayer<Dtype> 
{
 public:
  explicit AlignImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), 
	  augmentation_param_(this->layer_param_.align_image_data_param().augment_param()) {}
      
  virtual ~AlignImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "AlignImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  // 3rd blob is reserved for label, can be ignored
  virtual inline int ExactNumTopBlobs() const { return 3; }    
  
  // rewrite forward_cpu because we need to split prefetch_[i].label_
  // into pts & label
  virtual void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
      const std::vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom, 
      const std::vector<Blob<Dtype>*>& top); 

 protected:
  typedef std::pair< std::pair<std::string, std::string>, int > Line;
  
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  inline Line makeLine(std::string imgFile, std::string ptsFile, int label)
  {
    return std::make_pair( std::make_pair(imgFile, ptsFile), label);
  }
  
  // store in [n x 2] as CV_64F
  cv::Mat readPtsFile(const char* fileName) const;
  
  AlignAugmentationParameter augmentation_param_;
  shared_ptr<AlignAugmenter<Dtype> > align_augmenter_;
  
  std::vector<Line> lines_;
  //vector<std::pair<std::string, int> > lines_;
  int lines_id_;

#ifdef CAFFE_ALIGN_AUGMENT_WORKER_
  
  
  
  template <typename DDtype>
  class AlignAugmentWorker : public InternalThread
  {
   public:
    explicit AlignAugmentWorker(const AlignAugmentationParameter &augParam,
        const TransformationParameter &transParam, const AlignImageDataLayer<DDtype>& owner,
		Phase phase, int id, int workerNum);
	  virtual ~AlignAugmentWorker();
    
	  enum WorkerState {WORKING, IDLE};
    class sync_lock;
    
	  void toWorkState();
	  WorkerState getState();
    void lock(shared_ptr<sync_lock> &lock);
    
    const std::vector<Line>& getTaskList();
	  Batch<DDtype>* getBatchPtr();
    const std::string& getRootDir();
    bool getColor();
    cv::Mat readPtsFile(const char *filename) const;
    
   protected:
    virtual void InternalThreadEntry();
	  
	  void toIdleState();
    
	  const AlignImageDataLayer<DDtype>& owner_;
	  const AlignAugmentationParameter& augment_param_;
	  const TransformationParameter& transform_param_;
	  shared_ptr<AlignAugmenter<DDtype> > align_augmenter_;
	  shared_ptr<DataTransformer<DDtype> > data_transformer_;
	  WorkerState state_;
	  int worker_id_;
	  int worker_num_;
    
    std::string root_dir_;
    Blob<DDtype> transformed_data_;
    
	  class sync;
	  shared_ptr<sync> sync_;
  };
  
  
  
  /* task list provided to the workers, length should be the same as batch size.
   * It is initialized with lines_[0] x batch_size. 
   */
  friend const std::vector<Line>& AlignAugmentWorker<Dtype>::getTaskList();
  std::vector<Line> prefetch_task_;
  
  friend Batch<Dtype>* AlignAugmentWorker<Dtype>::getBatchPtr();
  Batch<Dtype>* prefetch_batch_;
  
  friend const std::string& AlignAugmentWorker<Dtype>::getRootDir();
  
  friend bool AlignAugmentWorker<Dtype>::getColor();
  
  friend cv::Mat AlignAugmentWorker<Dtype>::readPtsFile(const char *filename) const;
  
  std::vector<shared_ptr<AlignAugmentWorker<Dtype> > > workers_;
  int num_workers_;
  int setNumWorkers();  
  std::vector<shared_ptr<typename AlignAugmentWorker<Dtype>::sync_lock> > workers_locker_;

#endif // CAFFE_ALIGN_AUGMENT_WORKER_
};

} // namespace caffe

#endif // USE_OPENCV

#endif // ALIGN_IMAGE_DATA_LAYER_HPP_
