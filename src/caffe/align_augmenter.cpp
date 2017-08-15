#ifdef USE_OPENCV
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>

#include <string>

#include "caffe/align_augmenter.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
AlignAugmenter<Dtype>::AlignAugmenter(const AlignAugmentationParameter &param, 
    const TransformationParameter &transParam, Phase phase)
    : param_(param), phase_(phase) 
{
  // ONLY allow scale, mean_file, mean of Transformer
  CHECK_EQ(transParam.has_mirror(), false) 
    << "Cannot use \"mirror\" of DataTransformer with Augmentation at the same time";
  CHECK_EQ(transParam.has_crop_size(), false)
    << "Cannot use \"crop_size\" of DataTransformer with Augmentation at the same time";
  CHECK_EQ(transParam.has_force_color(), false)
    << "Cannot use \"force_color\" of DataTransformer with Augmentation at the same time";
  CHECK_EQ(transParam.has_force_gray(), false)
    << "Cannot use \"force_gray\" of DataTransformer with Augmentation at the same time";
  
  CHECK_EQ(param_.has_num_points(), true)
    << "Number of points must be specified";
  CHECK_GT(param_.num_points(), 0)
    << "Number of points must be a positive integer";
  
  if (param_.has_mirror() && param_.mirror())
  {
    CHECK_GT(param_.idx_left_points_size(), 0) 
      << "Mirror is permitted, must specify indecies of points that is on the left";
    CHECK_GT(param_.idx_right_points_size(), 0) 
      << "Mirror is permitted, must specify indecies of points that is on the right";
    CHECK_EQ(param_.idx_right_points_size(), param_.idx_left_points_size()) 
      << "Number of points on the left must be same as that on the right";
    CHECK_LE(param_.idx_left_points_size() * 2, param_.num_points())
      << "points on the left + points on the right cannot be larger than total number of points";
      
    const int pairLength = param_.idx_left_points_size();
    std::vector<int> checkUniqueList(param_.num_points(), 0);
    for(int i = 0; i < pairLength; i ++)
    {
      //convert to 0-based indecies
      int leftIdx = param_.idx_left_points(i) - 1;
      int rightIdx = param_.idx_right_points(i) - 1;
      CHECK_LT(leftIdx, param_.num_points()) 
        << "Left [" << i << "] is larger than number of points";
      CHECK_LT(rightIdx, param_.num_points()) 
        << "Right [" << i << "] is larger than number of points";
      CHECK_GE(leftIdx, 0) 
        << "Left [" << i << "] is smaller than 1";
      CHECK_GE(rightIdx, 0) 
        << "Right [" << i << "] is smaller than 1";
      CHECK_EQ(checkUniqueList[leftIdx], 0) 
        << leftIdx << " is duplicated";
      CHECK_EQ(checkUniqueList[rightIdx], 0)
        << rightIdx << " is duplicated";
      checkUniqueList[leftIdx] ++;
      checkUniqueList[rightIdx] ++;
      mirrorPairs_.push_back(cv::Vec2i(leftIdx, rightIdx));
    }
    
  }
  
  if (param_.has_rotate() && param_.rotate())
  {
    CHECK_EQ(param_.has_min_rotate(), true)
      << "Rotation is permitted, must specify minimum angle of rotation allowed (can be minus)";
    CHECK_EQ(param_.has_max_rotate(), true)
      << "Rotation is permitted, must specify maximum angle of rotation allowed (can be minus)";
    CHECK_LT(param_.min_rotate(), param_.max_rotate())
      << "Min rotate must LESS than Max rotate";
  }
  
  if (param_.has_size())
  {
    CHECK_EQ(param_.has_height(), false) 
      << "Cannot specify \"height\" with \"size\" at the same time";
    CHECK_EQ(param_.has_width(), false) 
      << "Cannot specify \"width\" with \"size\" at the same time";
    height_ = param_.size();
    width_  = param_.size();
  }
  else
  {
    CHECK_EQ(param_.has_height(), true) 
      << "Without \"size\", \"height\" and \"width\" MUST be specified";
    CHECK_EQ(param_.has_width(), true) 
      << "Without \"size\", \"height\" and \"width\" MUST be specified";
    height_ = param.height();
    width_  = param.width();
  }
  
  CHECK_GT(param_.max_crop_size(), 0)
    << "max crop size must be positive";
  CHECK_GT(param_.min_crop_size(), 0)
    << "min crop size must be positive";
  CHECK_GE(param_.max_crop_size(), param_.min_crop_size())
    << "max crop size must larger than min crop size";
    
  //if (param_.has_ignore_start_inc() || param_.has_ignore_end_inc() || param_.ignore_idx_size() > 0)
    ptsMask_ = cv::Mat::ones(param_.num_points(), 1, CV_8U);
  //else
  //  ptsMask_ = cv::noArray();
  
  if (param_.has_ignore_start_inc())
  {
    CHECK_GE(param_.ignore_start_inc(), 1)
      << "ignore_start_inc must be larger thant 1";
    CHECK_LE(param_.ignore_start_inc(), param_.num_points())
      << "ignore_start_inc must be smaller than number of points " << param_.num_points();
  }
  
  if (param_.has_ignore_end_inc())
  {
    CHECK_GE(param_.ignore_end_inc(), 1)
      << "ignore_end_inc must be larger thant 1";
    CHECK_LE(param_.ignore_end_inc(), param_.num_points())
      << "ignore_end_inc must be smaller than number of points " << param_.num_points();
  }
  
  if (param_.has_ignore_start_inc() && param_.has_ignore_end_inc())
  {
    CHECK_LE(param_.ignore_start_inc(), param_.ignore_end_inc())
      << "ignore_start_inc > ignore_end_inc";
    for (int i = param_.ignore_start_inc() - 1; i < param_.ignore_end_inc(); i ++)
      ptsMask_.at<uchar>(i, 0) = 0;
  }
  else if (param_.has_ignore_start_inc())
  {
    for (int i = param_.ignore_start_inc() - 1; i < param_.num_points(); i ++)
      ptsMask_.at<uchar>(i, 0) = 0;
  }
  else if (param_.has_ignore_end_inc())
  {
    for (int i = 0; i < param_.ignore_end_inc(); i ++)
      ptsMask_.at<uchar>(i, 0) = 0;
  }
  
  if (param_.ignore_idx_size() > 0)
    for (int i = 0; i < param_.ignore_idx_size(); i ++)
    {
      int idx = param_.ignore_idx(i) - 1;
      // converted to 0-based index
      CHECK_GE(idx, 0)
        << "ignore_idx[" << i << "] must be larger thant 1";
      CHECK_LT(idx, param_.num_points())
        << "ignore_idx[" << i << "] must be smaller than number of points " << param_.num_points();
      ptsMask_.at<uchar>(idx, 0) = 0;
    }
    
  CHECK_GT(cv::sum(ptsMask_)[0], 0)
    << "You can not ignore all points!";
  
  if (param_.has_ignore_start_inc() || param_.has_ignore_end_inc() || param_.ignore_idx_size() > 0)
  {
    std::string ignoreList = "Ignoring points: ";
    char tmp[10];
    for (int i = 0; i < param_.num_points(); i ++)
      if (ptsMask_.at<uchar>(i, 0) == 0)
      {
        sprintf(tmp, "%d, ", i + 1);
        ignoreList += tmp;
      }
      //ignoreList += ( ptsMask_.at<uchar>(i, 0) ? "" : (std::to_string(i + 1) + ", ") ); 
    LOG(INFO) << ignoreList;
  }
  
}

template <typename Dtype>
void AlignAugmenter<Dtype>::Augment(const cv::Mat &cv_img, const cv::Mat &cv_pts, 
    cv::Mat &aug_cv_img, cv::Mat &aug_cv_pts)
{
  const bool doMirror = (phase_ == TRAIN) ? param_.mirror() && Rand(2) : false;
  
  double angle = ((phase_ == TRAIN) && param_.rotate()) ? 
      getUniformRand(param_.min_rotate(), param_.max_rotate()) : 0;
  double xMin, xMax, yMin, yMax;
  cv::minMaxLoc(cv_pts.col(0), &xMin, &xMax, NULL, NULL, ptsMask_);
  cv::minMaxLoc(cv_pts.col(1), &yMin, &yMax, NULL, NULL, ptsMask_);
  
  cv::Rect oriBoundBox(floor(xMin), floor(yMin), ceil(xMax-xMin), ceil(yMax - yMin));
  cv::Rect rotatedBoundBox, randBoundBox;
  cv::Mat trans1 = makeRotate(oriBoundBox, angle, rotatedBoundBox);
  randBoundBox = generateRandomBoundingRect(rotatedBoundBox, 
      param_.min_crop_size(), param_.max_crop_size());
  cv::Mat trans2 = makeRandomCropAndResize(randBoundBox, cv::Size(width_, height_));
  
  cv::Mat trans;
  if(doMirror)
  {
    cv::Mat transMirror = makeMirror(width_, height_);
    trans = transMirror * trans2 * trans1;
  }
  else
    trans = trans2 * trans1;
  cv::warpAffine(cv_img, aug_cv_img, trans.rowRange(0, 2), cv::Size(width_, height_));
  aug_cv_pts = warpPointMat(cv_pts, trans);
  
  if(doMirror)
    processMirrorPts(aug_cv_pts, mirrorPairs_);
  
  if(param_.normalize())
  {
    aug_cv_pts.col(0) = aug_cv_pts.col(0) * ( static_cast<Dtype>(1) / width_);
    aug_cv_pts.col(1) = aug_cv_pts.col(1) * ( static_cast<Dtype>(1) / height_);
  }
}

static void processMirrorExtra(cv::Mat &data, const std::vector<cv::Vec2i> &pairs)
{
  for (size_t i = 0; i < pairs.size(); i++)
  {
    int a = pairs[i][0];
    int b = pairs[i][1];
    float *pa = data.ptr<float>(a);
    float *pb = data.ptr<float>(b);
    for (int d = 0; d < data.cols; d++)
    {
      float t = pa[d];
      pa[d] = pb[d];
      pb[d] = t;
    }
  }
}

template <typename Dtype>
cv::Mat AlignAugmenter<Dtype>::Augment(const cv::Mat &cv_pts, cv::Mat &aug_cv_pts,
  const cv::Mat &cv_extra, cv::Mat &aug_cv_extra, Caffe::RNG *provided_rng)
{
  const bool doMirror = (phase_ == TRAIN) ? param_.mirror() && Rand(provided_rng, 2) : false;
  
  double angle = ((phase_ == TRAIN) && param_.rotate()) ? 
      getUniformRand(provided_rng, param_.min_rotate(), param_.max_rotate()) : 0;
  double xMin, xMax, yMin, yMax;
  cv::minMaxLoc(cv_pts.col(0), &xMin, &xMax, NULL, NULL, ptsMask_);
  cv::minMaxLoc(cv_pts.col(1), &yMin, &yMax, NULL, NULL, ptsMask_);
  
  cv::Rect oriBoundBox(floor(xMin), floor(yMin), ceil(xMax-xMin), ceil(yMax - yMin));
  cv::Rect rotatedBoundBox, randBoundBox;
  cv::Mat trans1 = makeRotate(oriBoundBox, angle, rotatedBoundBox);
  randBoundBox = generateRandomBoundingRect(rotatedBoundBox, 
      param_.min_crop_size(), param_.max_crop_size(), provided_rng);
  cv::Mat trans2 = makeRandomCropAndResize(randBoundBox, cv::Size(width_, height_));
  
  cv::Mat trans;
  if(doMirror)
  {
    cv::Mat transMirror = makeMirror(width_, height_);
    trans = transMirror * trans2 * trans1;
  }
  else
    trans = trans2 * trans1;
  trans.convertTo(trans, CV_32F);
  aug_cv_pts = warpPointMat(cv_pts, trans);
  
  if (doMirror)
  {
    processMirrorPtsf(aug_cv_pts, mirrorPairs_);
    if (cv_extra.cols)
    {
      aug_cv_extra = cv_extra.clone();
      processMirrorExtra(aug_cv_extra, mirrorPairs_);
    }
  }
  
  if(param_.normalize())
  {
    aug_cv_pts.col(0) = aug_cv_pts.col(0) * ( static_cast<Dtype>(1) / width_);
    aug_cv_pts.col(1) = aug_cv_pts.col(1) * ( static_cast<Dtype>(1) / height_);
  }
  return trans;
}

template <typename Dtype>
cv::Mat AlignAugmenter<Dtype>::makeRotate(const cv::Rect &originBoundBox, 
    double angle, cv::Rect &resBoundBox)
{
  const double Pi = 3.1415926;
  double rad = angle / 180 * Pi;
  //double absRad = (rad > 0) ? rad : -rad;
  double m1data[9] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  }, m2data[9], m3data[9];
  memcpy(m2data, m1data, 9 * sizeof(double));
  memcpy(m3data, m1data, 9 * sizeof(double));
  double width = originBoundBox.width;
  double height = originBoundBox.height;
  m1data[2] = -width / 2 - originBoundBox.x;
  m1data[5] = -height / 2 - originBoundBox.y;
  
  cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f((float)width, (float)height), float(angle));
  resBoundBox = rRect.boundingRect();
  //fprintf(stderr, "x = %d, y = %d, w = %d, h = %d\n", resBoundBox.x, resBoundBox.y, resBoundBox.width, resBoundBox.height);
  
  m2data[0] = m2data[4] = std::cos(rad);
  m2data[1] = -std::sin(rad);
  m2data[3] = std::sin(rad);
  
  cv::Mat trans = cv::Mat(3, 3, CV_64F, m2data) * cv::Mat(3, 3, CV_64F, m1data);
  
  return trans.clone();
}

template <typename Dtype>
cv::Mat AlignAugmenter<Dtype>::makeRandomCropAndResize(const cv::Rect &randomBoundingRect, 
    const cv::Size &targetSize)
{
  double m1data[9] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  }, m2data[9];
  memcpy(m2data, m1data, 9 * sizeof(double));
  m1data[2] = -randomBoundingRect.x;
  m1data[5] = -randomBoundingRect.y;
  
  double scaleX = (double)targetSize.width / randomBoundingRect.width;
  double scaleY = (double)targetSize.height / randomBoundingRect.height;
  
  m2data[0] = scaleX;
  m2data[4] = scaleY;
  
  //fprintf(stderr, "scaleX = %f, scaleY = %f\n", scaleX, scaleY);
  
  cv::Mat trans = cv::Mat(3, 3, CV_64F, m2data) * cv::Mat(3, 3, CV_64F, m1data);
  return trans.clone();
}

template <typename Dtype>
cv::Rect AlignAugmenter<Dtype>::generateRandomBoundingRect(const cv::Rect rotatedBoundingRect, 
    const float &extendBoxMin, const float &extendBoxMax, Caffe::RNG *provided_rng)
{
  float halfW = rotatedBoundingRect.width / 2.0f;
  float halfH = rotatedBoundingRect.height / 2.0f;
  // extend rotated rect to square first
  float half = (halfW > halfH) ? halfW : halfH;
  // then generate random square in the allowed region
  double X = -half * ((phase_ == TRAIN) ? 
      getUniformRand(provided_rng, extendBoxMin, extendBoxMax) : 0.5 * (extendBoxMin + extendBoxMax));
  //fprintf(stderr, "%f in (%f, %f)\n", X, -half * extendBoxMin, -half * extendBoxMax);
  double Y = -half * ((phase_ == TRAIN) ? 
      getUniformRand(provided_rng, extendBoxMin, extendBoxMax) : 0.5 * (extendBoxMin + extendBoxMax));
  double side;
  if(phase_ == TRAIN)
    side = ((X < Y) ? 
        getUniformRand(provided_rng, half * extendBoxMin, X - Y + half * extendBoxMax) :
        getUniformRand(provided_rng, X - Y + half * extendBoxMin, half * extendBoxMax))
    - X;
  else
    side = half * (extendBoxMin + extendBoxMax);
  
  return cv::Rect(
    floor(X + rotatedBoundingRect.x + halfW),
    floor(Y + rotatedBoundingRect.y + halfH),
    ceil(side),
    ceil(side)
  );
}

template <typename Dtype>
cv::Mat AlignAugmenter<Dtype>::makeMirror(const int &width, const int &height)
{
  double m1data[9] = {
  	1, 0, 0,
  	0, 1, 0,
  	0, 0, 1
  }, m2data[9], m3data[9];
  memcpy(m2data, m1data, 9 * sizeof(double));
  memcpy(m3data, m1data, 9 * sizeof(double));
  
  m1data[2] = -(double)width / 2;
  m1data[5] = -(double)height / 2;
  
  m3data[2] = (double)width / 2;
  m3data[5] = (double)height / 2;
  
  m2data[0] = -1;
  cv::Mat trans = cv::Mat(3, 3, CV_64F, m3data) * cv::Mat(3, 3, CV_64F, m2data) * cv::Mat(3, 3, CV_64F, m1data);
  return trans.rowRange(0, 2).clone();
}

template <typename Dtype>
void AlignAugmenter<Dtype>::processMirrorPts(cv::Mat &pts, const std::vector<cv::Vec2i> &pairs)
{
  for (size_t i = 0; i < pairs.size(); i++)
  {
    int a = pairs[i][0];
    int b = pairs[i][1];
    float tmp = pts.at<double>(a, 0);
    pts.at<double>(a, 0) = pts.at<double>(b, 0);
    pts.at<double>(b, 0) = tmp;
    tmp = pts.at<double>(a, 1);
    pts.at<double>(a, 1) = pts.at<double>(b, 1);
    pts.at<double>(b, 1) = tmp;
  }
}

template <typename Dtype>
void AlignAugmenter<Dtype>::processMirrorPtsf(cv::Mat &pts, const std::vector<cv::Vec2i> &pairs)
{
  for (size_t i = 0; i < pairs.size(); i++)
  {
    int a = pairs[i][0];
    int b = pairs[i][1];
    float tmp = pts.at<float>(a, 0);
    pts.at<float>(a, 0) = pts.at<float>(b, 0);
    pts.at<float>(b, 0) = tmp;
    tmp = pts.at<float>(a, 1);
    pts.at<float>(a, 1) = pts.at<float>(b, 1);
    pts.at<float>(b, 1) = tmp;
  }
}

template <typename Dtype>
cv::Mat AlignAugmenter<Dtype>::warpPointMat(const cv::Mat &ptsMat, const cv::Mat &trans)
{
  cv::Mat tmp = cv::Mat::ones(ptsMat.rows, 3, ptsMat.type());
	ptsMat.copyTo(tmp(cv::Rect(0, 0, 2, ptsMat.rows)));
  return tmp * trans.rowRange(0, 2).t();
}

template <typename Dtype>
void AlignAugmenter<Dtype>::InitRand()
{
  if (phase_ == TRAIN)
  {
    const unsigned int rngSeed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rngSeed));
  }
  else
  {
    rng_.reset();
  }
}

template <typename Dtype>
int AlignAugmenter<Dtype>::Rand(int n)
{
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t *rng = static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
int AlignAugmenter<Dtype>::Rand(Caffe::RNG *provided_rng, int n)
{
  CHECK(provided_rng);
  CHECK_GT(n, 0);
  caffe::rng_t *rng = static_cast<caffe::rng_t*>(provided_rng->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
Dtype AlignAugmenter<Dtype>::getUniformRand(const Dtype &lowerBound, const Dtype &upperBound)
{
  CHECK(rng_);
  caffe::rng_t *rng = static_cast<caffe::rng_t*>(rng_->generator());
  return lowerBound + (upperBound - lowerBound) * (Dtype)(*rng)() / (Dtype)(rng->max());
}

template <typename Dtype>
Dtype AlignAugmenter<Dtype>::getUniformRand(Caffe::RNG *provided_rng, const Dtype &lowerBound, const Dtype &upperBound)
{
  CHECK(provided_rng);
  caffe::rng_t *rng = static_cast<caffe::rng_t*>(provided_rng->generator());
  return lowerBound + (upperBound - lowerBound) * (Dtype)(*rng)() / (Dtype)(rng->max());
}

INSTANTIATE_CLASS(AlignAugmenter);

} // namespace caffe

#endif // USE_OPENCV
