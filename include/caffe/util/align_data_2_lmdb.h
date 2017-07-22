#ifndef ALIGN_DATA_2_LMDB_H
#define ALIGN_DATA_2_LMDB_H

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe
{
namespace data_preprocess
{

  struct scale_down_param
  {
    bool is_color;
    float max_rotate;
    int net_input_size;
    float max_crop_size; 
    float min_crop_size;
  };

  int scale_down(const cv::Mat &src_img, const cv::Mat &src_pts, const scale_down_param param,
    cv::Mat &output_img, cv::Mat &output_pts, const cv::Mat &mask = cv::Mat());

  int align_data_2_datum(const cv::Mat &data, const cv::Mat &pts, int channels, int label, Datum &out);

  class LMDBWriter
  {
  public:
    explicit LMDBWriter(const string &path);
    virtual ~LMDBWriter();
    void put(unsigned long long key, const Datum &data);
    void commit();
  private:
    boost::shared_ptr<db::DB> dbinst_;
    boost::shared_ptr<db::Transaction> txn_;
    boost::mutex db_mutex_;
  };

} // namespace data_preprocess
} // namespace caffe

#endif // USE_OPENCV

#endif // ALIGN_DATA_2_LMDB_H
