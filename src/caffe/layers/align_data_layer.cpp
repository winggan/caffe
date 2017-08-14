#ifdef USE_OPENCV
#include "caffe/layers/align_data_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include <map>
#include <string>
#include <boost/thread.hpp>

namespace caffe 
{
  
namespace AlignDataInternal 
{

class QueuePair {
 public:
  explicit QueuePair(int size);
  ~QueuePair();

  const Datum& startReading();
  inline void finishReading();
  Datum& startWriting();
  inline void finishWriting();
  
 private: 
  Datum* reading_ = NULL, *writing_ = NULL;
 
  BlockingQueue<Datum*> free_;
  BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
};
  
// key of DBLoader is defined by {layer_name}+{TRAIN ? 1 : 0}+{db_path}
class DBLoader : public InternalThread 
{
 public: 
  static DBLoader &GetOrCreateLoader(const string &key,const LayerParameter& param);
  inline static std::string buildKey(const LayerParameter& param)
  { return param.name() + (param.phase() == TRAIN ? "1" : "0") + param.align_data_param().source(); }
  inline QueuePair& getReadingQueue(unsigned int id)
  { 
    CHECK_LT(id, readingQueues_.size()) << "No readingQueue[" << id << "]";
    return *(readingQueues_[id].get());
  }
  
  void InitRand();
  
  virtual ~DBLoader();
  
 protected:
  explicit DBLoader(const LayerParameter& param);
  
  void InternalThreadEntry();
  
  virtual int Rand(int n);
 
 private:
  const LayerParameter param_;
  std::vector<shared_ptr<QueuePair> > readingQueues_;
  int interval_;
  shared_ptr<Caffe::RNG> rng_;
 
  static std::map<const string, shared_ptr<DBLoader> > allLoader_;  
  static boost::mutex  mapMutex_;
  DISABLE_COPY_AND_ASSIGN(DBLoader);
};
 
} // AlignDataInternal 

boost::mutex AlignDataInternal::DBLoader::mapMutex_;
std::map<const string, shared_ptr<AlignDataInternal::DBLoader> > AlignDataInternal::DBLoader::allLoader_;

AlignDataInternal::QueuePair::QueuePair(int size): reading_(NULL), writing_(NULL)
{
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

AlignDataInternal::QueuePair::~QueuePair() {
  // ensure only current thread is operating
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
  if (reading_)
    delete reading_;
  if (writing_)
    delete writing_;
}

const Datum& AlignDataInternal::QueuePair::startReading()
{
  // explicit check finishing last read
  finishReading();
  reading_ = full_.pop("Waiting for data");
  return *reading_;
}

Datum& AlignDataInternal::QueuePair::startWriting()
{
  // explicit check finishing last write
  finishWriting();
  writing_ = free_.pop(""); // may be to noisy
  return *writing_;
}

void AlignDataInternal::QueuePair::finishReading()
{
  if (reading_)
  {
    free_.push(reading_);
    reading_ = NULL;
  }
}

void AlignDataInternal::QueuePair::finishWriting()
{
  if (writing_)
  {
    full_.push(writing_);
    writing_ = NULL;
  }
}

AlignDataInternal::DBLoader::DBLoader(const LayerParameter& param)
  : param_(param)
{
  int queue_count = param.phase() == TRAIN ? Caffe::solver_count() : 1;
  int preload_length = param.align_data_param().batch_size() * param.align_data_param().prefetch();
  
  LOG(INFO) << "creating " << queue_count << " preloading queue";
  readingQueues_.resize(queue_count);
  for (size_t i = 0; i < queue_count; i ++)
    readingQueues_[i].reset(new QueuePair(preload_length));
  
  interval_ = param.phase() == TRAIN ? param.align_data_param().rand_interval() + 1 : 1;
  InitRand(); 
  StartInternalThread();
}

AlignDataInternal::DBLoader::~DBLoader()
{
  StopInternalThread();
}

void AlignDataInternal::DBLoader::InitRand()
{
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

int AlignDataInternal::DBLoader::Rand(int n)
{
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

AlignDataInternal::DBLoader& AlignDataInternal::DBLoader::GetOrCreateLoader(const string &key,const LayerParameter& param)
{
  {
    boost::mutex::scoped_lock lock(mapMutex_);
    if (allLoader_.end() == allLoader_.find(key))
    {
      allLoader_[key] = shared_ptr<DBLoader>(new DBLoader(param));
    }
  }
  return *(allLoader_[key]);
}

void static read_one(db::Cursor* cursor, AlignDataInternal::QueuePair* qp)
{
  Datum& datum = qp->startWriting();
  datum.ParseFromString(cursor->value());
  qp->finishWriting();
  
  // go to the next iter
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

void static skip_one(db::Cursor* cursor)
{
  cursor->Next();
  if (!cursor->valid()) {
    DLOG(INFO) << "Restarting data prefetching from start.";
    cursor->SeekToFirst();
  }
}

void AlignDataInternal::DBLoader::InternalThreadEntry()
{
  DataParameter::DB backend;
  switch(param_.align_data_param().backend())
  {
    case AlignDataParameter::LEVELDB:
      backend = DataParameter::LEVELDB;
      break;
    case AlignDataParameter::LMDB:
      backend = DataParameter::LMDB;
      break;
    default:
      LOG(FATAL) << "unknown database backend";
  }
  shared_ptr<db::DB> db(db::GetDB( backend ));
  db->Open(param_.align_data_param().source(), db::READ);
  shared_ptr<db::Cursor> cursor(db->NewCursor());
  
  try
  {
    int rand_skip = Rand(param_.align_data_param().rand_skip() + 1);
    for (int i = 0; i < rand_skip; i ++)
      skip_one(cursor.get());
    while (!must_stop())
    {
      for (size_t i = 0; i < readingQueues_.size(); i ++)
        read_one(cursor.get(), readingQueues_[i].get());
      int skip = Rand(interval_);
      for (int i = 0; i < skip; i ++)
        skip_one(cursor.get());
    }
  }
  catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template <typename Dtype>
AlignDataLayer<Dtype>::AlignDataLayer(const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      augmentation_param_(param.align_data_param().augment_param()),
      prefetch_(param.align_data_param().prefetch())
{
  for (size_t i = 0; i < prefetch_.size(); i ++)
  {
    prefetch_[i].reset(new AlignBatch);
    prefetch_free_.push(prefetch_[i].get());
  }
  
  loaderKey_ = AlignDataInternal::DBLoader::buildKey(param);
  
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&picPushStream_, cudaStreamNonBlocking));
  }
}

template <typename Dtype>
AlignDataLayer<Dtype>::~AlignDataLayer()
{
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(picPushStream_));
  }
  StopInternalThread();
}

template <typename Dtype>
void AlignDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  align_augmenter_.reset(
    new AlignAugmenter<Dtype>(augmentation_param_,
      this->layer_param_.transform_param(), this->phase_)
  );
  
  align_augmenter_->InitRand();
  
  AlignDataInternal::DBLoader& loader = AlignDataInternal::DBLoader::GetOrCreateLoader(loaderKey_, this->layer_param_);
  AlignDataInternal::QueuePair& dbQueue = loader.getReadingQueue(Caffe::solver_rank());
  const int batch_size = this->layer_param_.align_data_param().batch_size();
  
  std::vector<int> topShape(4);
  topShape[0] = batch_size;
  topShape[2] = align_augmenter_->height();
  topShape[3] = align_augmenter_->width();
  const Datum& datum = dbQueue.startReading();
  topShape[1] = datum.channels();
  expect_channels_ = datum.channels();
  expect_extra_data_ = datum.float_data().size() - 2 * augmentation_param_.num_points();
  CHECK_GE(expect_extra_data_, 0) << "Datum should hold at least 2 * num_points float elements";
  dbQueue.finishReading();
  
  // init warpBuffer_ and destination ptr for warpaffine in GPU
  {
    int buffer_in_bytes = expect_channels_ * align_augmenter_->height() * align_augmenter_->width();
    int buffer_in_floats = (buffer_in_bytes >> 2) + 1;
    warpBuffer_.Reshape(std::vector<int>(1, buffer_in_floats));
    pWarpDst_.resize(expect_channels_);
    unsigned char *start = (unsigned char *)warpBuffer_.mutable_gpu_data();
    for (int i = 0; i < expect_channels_; i ++)
      pWarpDst_[i] = start + i * align_augmenter_->height() * align_augmenter_->width();

#if ALING_DATA_USE_REMAP
    std::vector<int> mapShape;
    mapShape.push_back(2);
    mapShape.push_back(align_augmenter_->height());
    mapShape.push_back(align_augmenter_->width());
    xyMap_.Reshape(mapShape);
    xyMap_.mutable_gpu_data(); // explicit allocate gpu memory
#endif // ALING_DATA_USE_REMAP
  }
  // init data_mean_, copied from data_transformer.cpp
  {
    const TransformationParameter& trans_param = this->layer_param_.transform_param();
    if (trans_param.has_mean_file()) 
    {
      CHECK_EQ(trans_param.mean_value_size(), 0) <<
        "Cannot specify mean_file and mean_value at the same time";
      const string& mean_file = trans_param.mean_file();
      LOG(INFO) << "Loading mean file from: " << mean_file;
      BlobProto blob_proto;
      ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
      data_mean_.FromProto(blob_proto);
      CHECK_EQ(data_mean_.shape(1), expect_channels_) << "data_mean_ channel dismatch";
      CHECK_EQ(data_mean_.shape(2), align_augmenter_->height()) << "data_mean_ height dismatch";
      CHECK_EQ(data_mean_.shape(3), align_augmenter_->width()) << "data_mean_ width dismatch";
    }
    // check if we want to use mean_value
    else if (trans_param.mean_value_size() > 0) 
    {
      CHECK(trans_param.has_mean_file() == false) <<
        "Cannot specify mean_file and mean_value at the same time";
      std::vector<Dtype> mean_values;
      for (int c = 0; c < trans_param.mean_value_size(); ++c) {
        mean_values.push_back(trans_param.mean_value(c));
      }
      CHECK(mean_values.size() == 1 || mean_values.size() == expect_channels_) 
        << "mean values should be either 1 mean_value or as many as channels";
      if (mean_values.size() == 1)
        for (int i = 1; i < expect_channels_; i ++)
          mean_values.push_back(mean_values[0]);
        
      std::vector<int> data_mean_shape(4, 1);
      data_mean_shape[1] = expect_channels_;
      data_mean_shape[2] = align_augmenter_->height();
      data_mean_shape[3] = align_augmenter_->width();
      data_mean_.Reshape(data_mean_shape);
      Dtype *data_mean_data = data_mean_.mutable_cpu_data();
      const int channel_step = align_augmenter_->height() *  align_augmenter_->width();
      for (int i = 0; i < expect_channels_; i ++)
        caffe_set(channel_step, mean_values[i], data_mean_data + i * channel_step);
    }
    else
    {
      std::vector<int> data_mean_shape(4, 1);
      data_mean_shape[1] = expect_channels_;
      data_mean_shape[2] = align_augmenter_->height();
      data_mean_shape[3] = align_augmenter_->width();
      data_mean_.Reshape(data_mean_shape);
      Dtype *data_mean_data = data_mean_.mutable_cpu_data();
      caffe_set<Dtype>(data_mean_.count(), 0, data_mean_data);
    }
#ifndef CPU_ONLY
    data_mean_.gpu_data();
#endif
  }
  
  top[0]->Reshape(topShape);
  LOG(INFO) << "output data size: " << top[0]->num() << "," 
    << top[0]->channels() << "," << top[0]->height() << "," 
    << top[0]->width();
  
  std::vector<int> ptsShape(2);
  ptsShape[0] = batch_size;
  ptsShape[1] = 2 * augmentation_param_.num_points();
  top[1]->Reshape(ptsShape);
  
  std::vector<int> labelShape(1, batch_size);
  top[2]->Reshape(labelShape);

  std::vector<int> transShape(1, batch_size);
  transShape.push_back(6);
  if (top.size() > 3)
  { // should forward the affine transformation matrix
    top[3]->Reshape(transShape);
  }

  std::vector<int> extraDataShape(1, batch_size);
  // prevent empty blob
  extraDataShape.push_back(expect_extra_data_ > 0 ? expect_extra_data_ : 1);
  if (top.size() > 4)
  {
    CHECK_GT(expect_extra_data_, 0) << "no extra data to forward";
    top[4]->Reshape(extraDataShape);
  }
  
  for (size_t i = 0; i < prefetch_.size(); i ++)
  {
    prefetch_[i]->pts_.Reshape(ptsShape);
    prefetch_[i]->label_.Reshape(labelShape);
    prefetch_[i]->trans_blob_.Reshape(transShape);
    prefetch_[i]->extra_data_.Reshape(extraDataShape);
    prefetch_[i]->data_.resize(batch_size);
    for (size_t sample = 0; sample < prefetch_[i]->data_.size(); sample ++)
      prefetch_[i]->data_[sample].reset(new Blob<float>);
    prefetch_[i]->w_.resize(batch_size);
    prefetch_[i]->h_.resize(batch_size);
    prefetch_[i]->trans_.resize(batch_size);
  }
  
  for (size_t i = 0; i < prefetch_.size(); ++i) 
  {
    prefetch_[i]->pts_.mutable_cpu_data();
    prefetch_[i]->label_.mutable_cpu_data();
    prefetch_[i]->trans_blob_.mutable_cpu_data();
    prefetch_[i]->extra_data_.mutable_cpu_data();
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (size_t i = 0; i < prefetch_.size(); ++i) 
    {
      prefetch_[i]->pts_.mutable_gpu_data();
      prefetch_[i]->label_.mutable_gpu_data();
      prefetch_[i]->trans_blob_.mutable_gpu_data();
      prefetch_[i]->extra_data_.mutable_gpu_data();
    }
  }
#endif
  
  // start thread after initialzation
  StartInternalThread();
}

static void loadImgIntoAlignBatch(AlignBatch& batch, int sample, const Datum& datum)
{
  batch.w_[sample] = datum.width();
  batch.h_[sample] = datum.height();
  unsigned int size_in_bytes = datum.channels() * datum.width() * datum.height();
  unsigned int size_in_floats = (size_in_bytes >> 2) + 1;
  std::vector<int> shape(1, size_in_floats);
  // may call cudaMallocHost, which is not reported to be a problem according to base_data_layer.cpp
  batch.data_[sample]->Reshape(shape);
  memcpy(batch.data_[sample]->mutable_cpu_data(), datum.data().c_str(), size_in_bytes);
}

template <typename Dtype>
void AlignDataLayer<Dtype>::load_batch(AlignBatch& batch) 
{
  AlignDataInternal::DBLoader& loader = AlignDataInternal::DBLoader::GetOrCreateLoader(loaderKey_, this->layer_param_);
  AlignDataInternal::QueuePair& dbQueue = loader.getReadingQueue(Caffe::solver_rank());
  const int batch_size = this->layer_param_.align_data_param().batch_size();
  const int num_pt = augmentation_param_.num_points();
  
  float *pts_data = batch.pts_.mutable_cpu_data();
  float *label_data = batch.label_.mutable_cpu_data();
  float *trans_blob_data = batch.trans_blob_.mutable_cpu_data();
  float *extra_data_data = batch.extra_data_.mutable_cpu_data();
  for (int sample = 0; sample < batch_size; sample ++)
  {
    const Datum& datum = dbQueue.startReading();
    CHECK_EQ(expect_channels_, datum.channels()) << " #channel dismatch! ";
    CHECK_EQ(num_pt * 2 + expect_extra_data_, datum.float_data().size()) 
      << " float data should have length of num_pt * 2 + expect_extra_data_";
    // load datum into batch
    label_data[sample] = datum.label();
    loadImgIntoAlignBatch(batch, sample, datum);
    cv::Mat originPts(num_pt, 2, CV_32F, const_cast<float *>(datum.float_data().data()));
    cv::Mat aug_pts;
    batch.trans_[sample] = align_augmenter_->Augment(originPts, aug_pts);
    memcpy(pts_data + sample * 2 * num_pt, aug_pts.data, 2 * num_pt * sizeof(float));
    memcpy(trans_blob_data + sample * 6, batch.trans_[sample].data, 6 * sizeof(float));
    if (expect_extra_data_)
      memcpy(
        extra_data_data + sample * expect_extra_data_,
        datum.float_data().data() + 2 * num_pt,
        expect_extra_data_ * sizeof(float));
    // finish reading
    dbQueue.finishReading();
  }
  
}

template <typename Dtype>
void AlignDataLayer<Dtype>::InternalThreadEntry() 
{
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      AlignBatch* batch = prefetch_free_.pop();
      load_batch(*batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        //for (size_t i = 0; i < batch->data_.size(); 
        //  batch->data_[i].data().get()->async_gpu_push(stream);
        batch->label_.data().get()->async_gpu_push(stream);
        batch->pts_.data().get()->async_gpu_push(stream);
        batch->trans_blob_.data().get()->async_gpu_push(stream);
        batch->extra_data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void AlignDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  LOG(FATAL) << "NOT Implemented yet";
}

INSTANTIATE_CLASS(AlignDataLayer);
REGISTER_LAYER_CLASS(AlignData);

} // namespace caffe

#endif // USE_OPENCV
