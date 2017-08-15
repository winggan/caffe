#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <boost/thread.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/align_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

bool __extractPointFromLine(char * buf, cv::Point2d &p)
{
  double x, y;
  int match = std::sscanf(buf, "%lf %lf", &x, &y);
  if (match < 2 || cvIsNaN(x) || cvIsNaN(y) || cvIsInf(x) || cvIsInf(y))
    return false;
  p.x = x;
  p.y = y;
  return true;
}

namespace caffe {
  
template <typename Dtype>
AlignImageDataLayer<Dtype>::~AlignImageDataLayer()
{ this->StopInternalThread(); }

template <typename Dtype>
void AlignImageDataLayer<Dtype>::Forward_cpu(const std::vector<Blob<Dtype>*>& bottom,
      const std::vector<Blob<Dtype>*>& top)
{
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) 
  {
    // arrangement: [batch_size] labels, [batch_size * (num_points * 2)] pts

    std::vector<int> top_shape = batch->label_.shape();
    top_shape[1] --;
    top[1]->Reshape(top_shape);
    caffe_copy(batch->label_.count() - top_shape[0], 
      batch->label_.cpu_data() + top_shape[0],
        top[1]->mutable_cpu_data());
  
    top_shape.pop_back();
    top[2]->Reshape(top_shape);
    caffe_copy(top_shape[0], 
      batch->label_.cpu_data(),
        top[2]->mutable_cpu_data());
  }
  
  this->prefetch_free_.push(batch);
}

template <typename Dtype>
void AlignImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  const bool isColor = this->layer_param_.align_image_data_param().is_color();
  string  rootDir = this->layer_param_.align_image_data_param().root_folder();
  const string& source = this->layer_param_.align_image_data_param().source();
  
  align_augmenter_.reset(
    new AlignAugmenter<Dtype>(augmentation_param_,
      this->layer_param_.transform_param(), this->phase_)
  );
  align_augmenter_->InitRand();
  
#ifdef CAFFE_ALIGN_AUGMENT_WORKER_
  
  num_workers_ = setNumWorkers();
  for (int i = 0; i < num_workers_; i ++)
  {
    workers_.push_back( shared_ptr<AlignAugmentWorker<Dtype> >() );
    workers_.back().reset( new AlignAugmentWorker<Dtype>(augmentation_param_,
        this->layer_param_.transform_param(), *this, this->phase_, 
        i, num_workers_) );
    workers_locker_.push_back( shared_ptr<typename AlignAugmentWorker<Dtype>::sync_lock> () );
  }
  
#endif // CAFFE_ALIGN_AUGMENT_WORKER_
  
  if(rootDir[rootDir.length() - 1] == '\\')
    rootDir[rootDir.length() - 1] = '/';
  else if(rootDir[rootDir.length() - 1] != '/')
    rootDir = rootDir + "/";
  
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  
  std::string line;
  int label;
  size_t pos;
  while (std::getline(infile, line))
  {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    line = line.substr(0, pos);
    pos = line.find_last_not_of(' ');
    pos = line.substr(0, pos + 1).find_last_of(' ');
    lines_.push_back( makeLine(line.substr(0, pos), line.substr(pos + 1), label) );
  }
  
  CHECK(!lines_.empty()) << "File is empty";
  
  if (this->layer_param_.align_image_data_param().shuffle()) 
  {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images";
  
  lines_id_ = 0;
  if (this->layer_param_.align_image_data_param().rand_skip()) 
  {
    lines_id_ = caffe_rng_rand() % 
        this->layer_param_.align_image_data_param().rand_skip();
    CHECK_GT(lines_.size(), lines_id_) << "Not enough lines to skip";
    LOG(INFO) << "Skipping first " << lines_id_ << " lines of data";
  }
  
  cv::Mat sampleImg = ReadImageToCVMat(
      rootDir + lines_[lines_id_].first.first, 0, 0, isColor);
  CHECK(sampleImg.data) << "Could not load " 
    << rootDir + lines_[lines_id_].first.first;

  cv::Mat samplePts = readPtsFile((rootDir + lines_[lines_id_].first.second).c_str());
  CHECK(samplePts.data) << "Could not load pts " 
    << rootDir + lines_[lines_id_].first.second;
  CHECK_EQ(samplePts.rows, augmentation_param_.num_points()) 
    << "Invalid pts: number of points do not match";
  cv::Mat cv_img, cv_pts;
  align_augmenter_->Augment(sampleImg, (const cv::Mat&)samplePts, cv_img, cv_pts);
  
  std::vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  
  const int batch_size = this->layer_param_.align_image_data_param().batch_size(); 
  CHECK_GT(batch_size, 0) << "Positive batch size required"; 
  top_shape[0] = batch_size; 
  for (int i = 0; i < this->prefetch_.size(); ++i) 
  { 
    this->prefetch_[i]->data_.Reshape(top_shape); 
  }
  top[0]->Reshape(top_shape);
  
  LOG(INFO) << "output data size: " << top[0]->num() << "," 
    << top[0]->channels() << "," << top[0]->height() << "," 
    << top[0]->width(); 

  vector<int> prefetch_shape(1, batch_size);
  prefetch_shape.push_back(1 + samplePts.rows * 2);
  vector<int> pts_shape(1, batch_size);
  pts_shape.push_back(samplePts.rows * 2);
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(pts_shape);
  top[2]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) 
  { 
    this->prefetch_[i]->label_.Reshape(prefetch_shape); 
  } 
  
#ifdef CAFFE_ALIGN_AUGMENT_WORKER_

  for (int i = 0; i < batch_size; i ++)
    prefetch_task_.push_back(lines_[0]);
  
#endif // CAFFE_ALIGN_AUGMENT_WORKER_
  
  LOG(INFO) << "output pts size: "
    << top[1]->num() << ","  << top[1]->channels();
  // arrangement: [batch_size] labels, [batch_size * (num_points * 2)] pts
  LOG(INFO) << "prefetch label size: "
    << this->prefetch_[0]->label_.num() << ","  
   << this->prefetch_[0]->label_.channels();
}

template <typename Dtype>
void AlignImageDataLayer<Dtype>::ShuffleImages()
{
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void AlignImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
  CPUTimer batch_timer;
  batch_timer.Start();
#ifndef CAFFE_ALIGN_AUGMENT_WORKER_
  double readTime = 0;
  double transTime = 0;
#endif // CAFFE_ALIGN_AUGMENT_WORKER_
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  
  const bool isColor = this->layer_param_.align_image_data_param().is_color();
  string  rootDir = this->layer_param_.align_image_data_param().root_folder();

  if(rootDir[rootDir.length() - 1] == '\\')
    rootDir[rootDir.length() - 1] = '/';
  else if(rootDir[rootDir.length() - 1] != '/')
    rootDir = rootDir + "/";
   
  cv::Mat rawImage = ReadImageToCVMat(
      rootDir + lines_[lines_id_].first.first, 0, 0, isColor);
  CHECK(rawImage.data) << "Could not load "
    << rootDir + lines_[lines_id_].first.first;
    
  cv::Mat rawPts = readPtsFile((rootDir + lines_[lines_id_].first.second).c_str());
  CHECK(rawPts.data) << "Could not load pts " 
    << rootDir + lines_[lines_id_].first.second;
  CHECK_EQ(rawPts.rows, augmentation_param_.num_points())
    << "Invalid pts: number of points do not match";
  cv::Mat cv_img, cv_pts;
  align_augmenter_->Augment(rawImage, (const cv::Mat&)rawPts, cv_img, cv_pts);
  
  std::vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  
  const int batch_size = this->layer_param_.align_image_data_param().batch_size(); 
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_pts = batch->label_.mutable_cpu_data() + batch_size;
  
#ifdef CAFFE_ALIGN_AUGMENT_WORKER_
  
  // enter critical section
  for (int i = 0; i < num_workers_; i ++)
    workers_[i]->lock(workers_locker_[i]);
  
  // assign tasks
  const int lines_size = lines_.size();
  prefetch_batch_ = batch;
  for (int item_id = 0; item_id < batch_size; item_id ++)
  {
    CHECK_GT(lines_size, lines_id_);
    prefetch_task_[item_id] = lines_[lines_id_];
    //LOG(WARNING) << "Reading image: " << lines_[lines_id_].first.first;
    lines_id_ ++;
    if (lines_id_ >= lines_size)
    {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.align_image_data_param().shuffle())
        ShuffleImages();
    }
  }
  // leave critical section, 
  // get workers to work
  for (int i = 0; i < num_workers_; i ++)
  {
    workers_locker_[i].reset();
    workers_[i]->toWorkState();
  }
  
  // wait workers to finish their jobs, 
  // then enter critical section
  for (int i = 0; i < num_workers_; i ++)
  {
    while (AlignAugmentWorker<Dtype>::WORKING == workers_[i]->getState())
      boost::this_thread::yield();
    workers_[i]->lock(workers_locker_[i]);
  }
  
  //for (int item_id = 0; item_id < batch_size; item_id ++)
  //{
  //  LOG(WARNING) << "item[" << item_id << "].pts[0] = " 
  //    << prefetch_pts[item_id * rawPts.rows * 2] << ", " 
  //    << prefetch_pts[item_id * rawPts.rows * 2 + 1];
  //}
  //LOG(WARNING) << "Confirm all on prefetch thread.";
  
#else // CAFFE_ALIGN_AUGMENT_WORKER_
  
  const int lines_size = lines_.size();
  const int numPts = augmentation_param_.num_points();
  for (int item_id = 0; item_id < batch_size; item_id ++)
  {
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat rawImage, rawPts, cv_img, cv_pts;
    
    //LOG(WARNING) << "Reading image: " << lines_[lines_id_].first.first;
    
    rawImage = ReadImageToCVMat(
        rootDir + lines_[lines_id_].first.first, 0, 0, isColor);
    CHECK(rawImage.data) << "Could not load "
      << rootDir + lines_[lines_id_].first.first;
    
    rawPts = readPtsFile((rootDir + lines_[lines_id_].first.second).c_str());
    CHECK(rawPts.data) << "Could not load pts " 
      << rootDir + lines_[lines_id_].first.second;
    CHECK_EQ(rawPts.rows, numPts)
      << "Invalid pts: number of points do not match";
    readTime += timer.MicroSeconds();

    timer.Start();
    // produce an instance of augmentation
    align_augmenter_->Augment(rawImage, (const cv::Mat&)rawPts, cv_img, cv_pts);
    // copy image data
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    // copy pts data
    Dtype* transformed_pts = prefetch_pts + item_id * numPts * 2;
    for(int i = 0; i < numPts; i ++)
    {
      const int idx = (i << 1);
      const double* rowPtr = (const double *)cv_pts.ptr(i);
      transformed_pts[idx] = static_cast<Dtype>(rowPtr[0]);
      transformed_pts[idx+1] = static_cast<Dtype>(rowPtr[1]);
    }
    // copy label
    prefetch_label[item_id] = static_cast<Dtype>(lines_[lines_id_].second);
    lines_id_ ++;
    transTime += timer.MicroSeconds();
    if (lines_id_ >= lines_size)
    {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.align_image_data_param().shuffle())
        ShuffleImages();
    }
  }
  
#endif // CAFFE_ALIGN_AUGMENT_WORKER_
  
  batch_timer.Stop();
  //LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
#ifdef CAFFE_ALIGN_AUGMENT_WORKER_
  for (int i = 0; i < num_workers_; i ++)
    workers_locker_[i].reset();
#else // CAFFE_ALIGN_AUGMENT_WORKER_
  DLOG(INFO) << "     Read time: " << readTime / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << transTime / 1000 << " ms.";
#endif // CAFFE_ALIGN_AUGMENT_WORKER_
}

template <typename Dtype>
cv::Mat AlignImageDataLayer<Dtype>::readPtsFile(const char* fileName) const
{
  std::ifstream ptsFile(fileName);
  if (ptsFile.fail())
    return cv::Mat();
  
  const int bufSize = 1024;
  char buffer[bufSize];
  
  //version line 
  ptsFile.getline(buffer, bufSize);
  
  //n_points
  ptsFile.getline(buffer, bufSize);
  std::string line(buffer);
  int start = line.find_last_of(": ");
  if (start == std::string::npos)
    return cv::Mat();
  start++;
  long long nPoints;
  int match = std::sscanf(line.c_str() + start, "%lld", &nPoints);
  if (match < 1 || nPoints <= 0)
    return cv::Mat();
  
  // first {
  ptsFile.getline(buffer, bufSize);
  
  int n = (int)nPoints;
  cv::Mat ret(n, 2, CV_64F);
  for (int i = 0; i < n; i++)
  {
    ptsFile.getline(buffer, bufSize);
    cv::Point2d p;
    if (!__extractPointFromLine(buffer, p))
    {
      return cv::Mat();
    }
    //ret.push_back(p);
  ret.at<double>(i, 0) = p.x;
  ret.at<double>(i, 1) = p.y;
  }
  return ret;
}

#ifdef CAFFE_ALIGN_AUGMENT_WORKER_

template <typename Dtype>
int AlignImageDataLayer<Dtype>::setNumWorkers()
{
  char* ptrNumWorkers = NULL;
  ptrNumWorkers = getenv("CAFFE_AUGMENT_NUM_WORKERS");
  int maxNum = boost::thread::hardware_concurrency();
  if(maxNum == 0) maxNum = 4;
  if(ptrNumWorkers != NULL)
  {
    int num;
    if(sscanf(ptrNumWorkers, "%d", &num) == 1 && num <= maxNum)
    {
      LOG(INFO) << "Use " << num << " workers for augmentation."; 
      return num;
    }
  }
  LOG(INFO) << "Use " << maxNum << " workers for augmentation.";
  return maxNum;
}

template <typename Dtype> template<typename DDtype>
class AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::sync_lock
{
 public:
  boost::mutex::scoped_lock lock_;
  
  sync_lock(boost::mutex &mutex) : lock_(mutex){}
 private:
  sync_lock(){}
};

template <typename Dtype> template<typename DDtype>
class AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::sync
{
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template <typename Dtype> template<typename DDtype>
AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::AlignAugmentWorker(
  const AlignAugmentationParameter &augParam,
  const TransformationParameter &transParam,
  const AlignImageDataLayer<DDtype> &owner, Phase phase, int id, int workerNum)
    : owner_(owner), augment_param_(augParam), transform_param_(transParam), 
    state_(IDLE), worker_id_(id), worker_num_(workerNum), sync_(new sync())
{
  align_augmenter_.reset(new AlignAugmenter<DDtype>(augParam, transParam, phase));
  align_augmenter_->InitRand();
  
  data_transformer_.reset(new DataTransformer<DDtype>(transParam, phase));
  data_transformer_->InitRand();
  
  root_dir_ = getRootDir();
  if(root_dir_[root_dir_.length() - 1] == '\\')
    root_dir_[root_dir_.length() - 1] = '/';
  else if(root_dir_[root_dir_.length() - 1] != '/')
    root_dir_ = root_dir_ + "/";
  
  this->StartInternalThread();
}

template <typename Dtype> template<typename DDtype>
AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::~AlignAugmentWorker()
{
  this->StopInternalThread();
}

template <typename Dtype> template<typename DDtype>
void AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::toWorkState()
{
  boost::mutex::scoped_lock lock(sync_->mutex_);
  state_ = WORKING;
  lock.unlock();
  sync_->condition_.notify_one();
}

template <typename Dtype> template<typename DDtype>
typename AlignImageDataLayer<Dtype>::template AlignAugmentWorker<DDtype>::WorkerState
AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::getState()
{
  return state_;
}

template <typename Dtype> template<typename DDtype>
void AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::toIdleState()
{
  boost::mutex::scoped_lock lock(sync_->mutex_);
  state_ = IDLE;
  lock.unlock();
}

template <typename Dtype> template<typename DDtype>
void AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::lock(
    shared_ptr<typename AlignImageDataLayer<Dtype>::template AlignAugmentWorker<DDtype>::sync_lock> &lock)
{
  lock.reset(new sync_lock(sync_->mutex_) );
}

template <typename Dtype> template<typename DDtype>
const std::vector<typename AlignImageDataLayer<Dtype>::Line>& 
AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::getTaskList()
{
  return owner_.prefetch_task_;
}

template <typename Dtype> template<typename DDtype>
Batch<DDtype>* AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::getBatchPtr()
{
  return owner_.prefetch_batch_;
}

template <typename Dtype> template<typename DDtype>
const std::string& AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::getRootDir()
{
  return owner_.layer_param_.align_image_data_param().root_folder();
}

template <typename Dtype> template<typename DDtype>
bool AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::getColor()
{
  return owner_.layer_param_.align_image_data_param().is_color();
}

template <typename Dtype> template<typename DDtype>
cv::Mat AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::readPtsFile(const char *filename) const
{
  return owner_.readPtsFile(filename);
}

template <typename Dtype> template<typename DDtype>
void AlignImageDataLayer<Dtype>::AlignAugmentWorker<DDtype>::InternalThreadEntry()
{
  try
  {
    while(!must_stop())
    {
      if(IDLE == getState())
        boost::this_thread::yield();
      else
      {
        boost::mutex::scoped_lock lock(sync_->mutex_);
        //Do augmentation & transformation
        const vector<Line> &taskList = getTaskList();
        Batch<DDtype>* batch = getBatchPtr();
        
        const bool isColor = getColor();
        const int batch_size = static_cast<int>(taskList.size());
        const int numPts = augment_param_.num_points();
        const int start = worker_id_;
        const int step = worker_num_;
        DDtype* prefetch_data = batch->data_.mutable_cpu_data();
        DDtype* prefetch_label = batch->label_.mutable_cpu_data();
        DDtype* prefetch_pts = batch->label_.mutable_cpu_data() + batch_size;
        
        std::vector<int> top_shape = batch->data_.shape();
        top_shape[0] = 1;
        transformed_data_.Reshape(top_shape);
        
        CPUTimer timer;
        double imgTime = 0, ptsTime = 0, warpTime = 0, restTime = 0; 
        for (int item_id = start; item_id < batch_size; item_id += step)
        {
          cv::Mat rawImage, rawPts, cv_img, cv_pts;
          timer.Start();
          rawImage = ReadImageToCVMat(
              root_dir_ + taskList[item_id].first.first, 0, 0, isColor);
          CHECK(rawImage.data) << "Could not load "
            << root_dir_ + taskList[item_id].first.first;
          imgTime += timer.MicroSeconds();
          timer.Start();
          rawPts = readPtsFile((root_dir_ + taskList[item_id].first.second).c_str());
          CHECK(rawPts.data) << "Could not load pts " 
            << root_dir_ + taskList[item_id].first.second;
          CHECK_EQ(rawPts.rows, numPts)
            << "Invalid pts: number of points do not match";
          ptsTime += timer.MicroSeconds();
          timer.Start();
          // produce an instance of augmentation
          align_augmenter_->Augment(rawImage, (const cv::Mat&)rawPts, cv_img, cv_pts);
          warpTime += timer.MicroSeconds();
          timer.Start();
          // copy image data
          int offset = batch->data_.offset(item_id);
          transformed_data_.set_cpu_data(prefetch_data + offset);
          data_transformer_->Transform(cv_img, &(transformed_data_));
          // copy pts data
          DDtype* transformed_pts = prefetch_pts + item_id * numPts * 2;
          for(int i = 0; i < numPts; i ++)
          {
            const int idx = (i << 1);
            const double* rowPtr = (const double *)cv_pts.ptr(i);
            transformed_pts[idx] = static_cast<DDtype>(rowPtr[0]);
            transformed_pts[idx+1] = static_cast<DDtype>(rowPtr[1]); 
          }
          //copy label
          prefetch_label[item_id] = static_cast<DDtype>(taskList[item_id].second);
          restTime += timer.MicroSeconds();
          //LOG(WARNING) << "Worker " << worker_id_ << ": item[" << item_id << "].pts[0] = " 
          //  << transformed_pts[0] << ", " << transformed_pts[1];
        }
        //LOG(INFO) << "   Image: " << imgTime / 1000 << " ms."; 
        //LOG(INFO) << "     Pts: " << ptsTime / 1000 << " ms."; 
        //LOG(INFO) << "    Warp: " << warpTime / 1000 << " ms."; 
        //LOG(INFO) << "The Rest: " << restTime / 1000 << " ms."; 
        //LOG(WARNING) << "Worker " << worker_id_ << " complete task.";
        lock.unlock();
        toIdleState();
      }
    }
  } catch (boost::thread_interrupted&) { /* which is expected */ }
}

#endif // CAFFE_ALIGN_AUGMENT_WORKER_

INSTANTIATE_CLASS(AlignImageDataLayer);
REGISTER_LAYER_CLASS(AlignImageData);
 
} //namespace caffe



#endif // USE_OPENCV
