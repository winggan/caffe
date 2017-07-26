#ifdef USE_OPENCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/util/align_data_2_lmdb.h"
#define EPSILON 0.001

namespace caffe
{
namespace data_preprocess
{
using boost::shared_ptr;

int scale_down(const cv::Mat &src_img, const cv::Mat &src_pts, const scale_down_param param,
  cv::Mat &output_img, cv::Mat &output_pts, const cv::Mat &mask)
{
  cv::Mat actualMask = mask;
  if (actualMask.empty())
    actualMask = cv::Mat::ones(src_pts.rows, 1, CV_8U);
  
  { // parameter checkings
    CHECK(src_img.data);
    CHECK_EQ(src_img.channels(), 3);
    CHECK(src_pts.data);
    CHECK(actualMask.data);

    CHECK_GT(src_pts.rows, 1);
    CHECK_EQ(src_pts.cols, 2);

    CHECK_EQ(actualMask.rows, src_pts.rows);
    CHECK_EQ(actualMask.cols, 1);

    CHECK_GE(param.max_rotate, 0);
    CHECK_GT(param.min_crop_size, 0);
    CHECK_GE(param.max_crop_size, param.min_crop_size);

    CHECK_GT(param.net_input_size, 0);
  }

  double cx, cy, halfSize;
  { // actual bbox 
    double xMin, xMax, yMin, yMax;
    cv::minMaxIdx(src_pts.col(0), &xMin, &xMax, NULL, NULL, actualMask);
    cv::minMaxIdx(src_pts.col(1), &yMin, &yMax, NULL, NULL, actualMask);
    if (xMax - xMin < EPSILON || yMax - yMin < EPSILON)
      return -2;
    halfSize = 0.5 * ((yMax - yMin > xMax - xMin) ? yMax - yMin : xMax - xMin);
    cx = 0.5 * (xMax + xMin);
    cy = 0.5 * (yMax + yMin);
  }
  //LOG(INFO) << "cx cy halfSize = " << cx << " " << cy << " " << halfSize;  
  double rotatedSize;
  { // get size of rotated actual box
    double d = (param.max_rotate > 45.f) ? 45.f : param.max_rotate;
    d *= 0.0174532925199432957; // pi / 180
    rotatedSize = 2 * halfSize * (sin(d) + cos(d));
  }
  //LOG(INFO) << "rotatedSize = " << rotatedSize;
  double scale;
  { // determine scale 
    scale = param.net_input_size / (2 * halfSize * param.min_crop_size);
    scale = (scale > 1.0) ? 1.0 : scale;
  }
  //LOG(INFO) << "scale = " << scale;
  cv::Rect extendActualBox;
  { // extend the crop to adjust max_crop_size
    double halfExtendSize = 0.5 * rotatedSize * param.max_crop_size;
    extendActualBox = cv::Rect(
      (int)floor(cx - halfExtendSize),
      (int)floor(cy - halfExtendSize),
      (int)ceil(2 * halfExtendSize),
      (int)ceil(2 * halfExtendSize)
    );
  }
  //LOG(INFO) << "extendActualBox = " << extendActualBox;
  cv::Mat src1;
  if (param.is_color)
    src1 = src_img;
  else
    cv::cvtColor(src_img, src1, CV_BGR2GRAY);

  { // do crop and scale
    cv::Rect roi = extendActualBox & cv::Rect(0, 0, src1.cols, src1.rows);
    //LOG(INFO) << "roi = " << roi;
    cv::Mat cropped = src1(roi);
    output_pts = src_pts.clone();
    output_pts.col(0) -= roi.x;
    output_pts.col(1) -= roi.y;
    if (scale < 1.0)
    {
      double transData[6] = { scale, 0, 0, 0, scale, 0 };
      cv::Mat trans(2, 3, CV_64F, transData);
      cv::warpAffine(cropped, output_img, trans, cv::Size(ceil(roi.width * scale), ceil(roi.height * scale)));
      output_pts *= scale;
    }
    else
      output_img = cropped.clone();
  }

  if (output_img.channels() > 1)
  { //NHWC to NCHW
    cv::Mat nchw(output_img.rows * output_img.channels(), output_img.cols, output_img.depth());
    std::vector<cv::Mat> channels;
    cv::split(output_img, channels);
    for (int i = 0; i < output_img.channels(); i++)
      channels[i].copyTo(nchw.rowRange(i * output_img.rows, i * output_img.rows + output_img.rows));
    output_img = nchw;
  }

  return 0;

}

int align_data_2_datum(const cv::Mat &data, const cv::Mat &pts, int channels, int label, Datum &out)
{
  CHECK(data.data);
  CHECK(pts.data);
  
  CHECK_EQ(1, data.channels());
  CHECK_EQ(0, data.rows % channels);

  CHECK_EQ(pts.cols, 2);

  out.set_width(data.cols);
  out.set_height(data.rows / channels);
  out.set_channels(channels);
  out.set_label(label);
  out.set_encoded(false);

  out.set_data(data.data, out.width() * out.height() * channels);
  out.clear_float_data();
  for (int i = 0; i < pts.rows; i++)
  {
    const float* row = pts.ptr<float>(i);
    out.add_float_data(row[0]);
    out.add_float_data(row[1]);
  }

  return 0;
}

LMDBWriter::LMDBWriter(const string &path) : dbinst_(db::GetDB("lmdb")), db_mutex_()
{
  {
    boost::mutex::scoped_lock lock(db_mutex_);
    dbinst_->Open(path, db::NEW);
    txn_.reset(dbinst_->NewTransaction());
  }
}

LMDBWriter::~LMDBWriter()
{
  {
    boost::mutex::scoped_lock lock(db_mutex_);
    txn_->Commit();
    txn_.reset();
    dbinst_->Close();
  }
}

void LMDBWriter::commit()
{
  boost::mutex::scoped_lock lock(db_mutex_);
  txn_->Commit();
  txn_.reset(dbinst_->NewTransaction());
}

void LMDBWriter::put(unsigned long long key, const Datum &data)
{
  char keyStr[20];
  sprintf(keyStr, "%016llx", key);
  std::string value;
  CHECK_EQ(data.SerializeToString(&value), true);
  {
    boost::mutex::scoped_lock lock(db_mutex_);
    txn_->Put(keyStr, value);
  }
}

} // namespcae data_preprocess
} // namespace caffe
#endif // USE_OPENCV
