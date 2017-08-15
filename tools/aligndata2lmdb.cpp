#include <caffe/util/align_data_2_lmdb.h>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <sstream>

using namespace caffe::data_preprocess;

static scale_down_param param = {
  true,
  10.f,
  112,
  1.6f,
  0.85f
};

enum SrcType
{
  PTS, // one pts file for each image sample, two path and a label in each line
  CELEBA, // celaba annotation: image path and coordinates of points without label for each line
  OCCPTS //  tow pts file for each image sample, three path and a label in each line
};

static int(*lineProcessor)(const std::string &line, cv::Mat &img, cv::Mat &pts, cv::Mat &extra, int &label) = NULL;
static SrcType src_type;
static std::string src_path;
static int global_label = 1;

const static size_t commit_size = 1000;

static bool __extractPointFromLine(char * buf, cv::Point2d &p)
{
  double x, y;
  int match = std::sscanf(buf, "%lf %lf", &x, &y);
  if (match < 2 || cvIsNaN(x) || cvIsNaN(y) || cvIsInf(x) || cvIsInf(y))
    return false;
  p.x = x;
  p.y = y;
  return true;
}

static cv::Mat readPtsFile(const char* fileName) 
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
  cv::Mat ret(n, 2, CV_32F);
  for (int i = 0; i < n; i++)
  {
    ptsFile.getline(buffer, bufSize);
    cv::Point2d p;
    if (!__extractPointFromLine(buffer, p))
    {
      return cv::Mat();
    }
    //ret.push_back(p);
    ret.at<float>(i, 0) = static_cast<float>(p.x);
    ret.at<float>(i, 1) = static_cast<float>(p.y);
  }
  return ret;
}


static int ptsLineProcessor(const std::string &line, cv::Mat &img, cv::Mat &pts, cv::Mat &extra, int &label)
{
  std::string imgPath, ptsPath;
  {
    std::string streamSrc = line + " ";
    std::stringstream ss(streamSrc, std::ios_base::in);
    std::vector<std::string> parts;
    while(!ss.eof())
    {
      std::string line;
      std::getline(ss, line, ' ');
      if (line.length() > 0)
        parts.push_back(line); 
    }
    if (parts.size() != 3)
      return 1;
    if (1 != sscanf(parts[2].c_str(), "%d", &label) || label < 0)
      return 2;
    imgPath = parts[0];
    ptsPath = parts[1]; 
  }
  //LOG(INFO) << "pts path = " << ptsPath;
  pts = readPtsFile(ptsPath.c_str());
  if (pts.empty())
    return 3;

  img = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
  if (img.empty())
    return 4;

  return 0;
}

static int occluPtsLineProcessor(const std::string &line, cv::Mat &img, cv::Mat &pts, cv::Mat &extra, int &label)
{
  std::string imgPath, ptsPath, occPath;
  {
    std::string streamSrc = line + " ";
    std::stringstream ss(streamSrc, std::ios_base::in);
    std::vector<std::string> parts;
    while(!ss.eof())
    {
      std::string line;
      std::getline(ss, line, ' ');
      if (line.length() > 0)
        parts.push_back(line); 
    }
    if (parts.size() != 4)
      return 1;
    if (1 != sscanf(parts[3].c_str(), "%d", &label) || label < 0)
      return 2;
    imgPath = parts[0];
    ptsPath = parts[1]; 
    occPath = parts[2];
  }
  //LOG(INFO) << "pts path = " << ptsPath;
  pts = readPtsFile(ptsPath.c_str());
  if (pts.empty())
    return 3;

  img = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
  if (img.empty())
    return 4;
    
  cv::Mat occTmp;
  occTmp = readPtsFile(occPath.c_str());
  if (occTmp.empty())
    return 5;
  
  if (occTmp.rows != pts.rows)
    return 6;
    
  extra.create(1, occTmp.rows, CV_32F);
  float *dst = extra.ptr<float>();
  for (int i = 0; i < occTmp.rows; i ++)
  {
    const float *row = occTmp.ptr<float>(i);
    if (row[0] != row[1])
      return 7;
    dst[i] = row[0];
  }
  
  return 0;
}

static int celebaLineProcessor(const std::string &line, cv::Mat &img, cv::Mat &pts, cv::Mat &extra, int &label)
{
  int pos = line.find_first_of(" ");
  label = global_label;
  std::vector<std::string> numStr;
  {
    std::string streamSrc = line.substr(pos) + " ";
    std::stringstream ss(streamSrc, std::ios_base::in);
    //LOG(INFO) << "streamSrc = " << streamSrc;
    while (!ss.eof())
    {
      std::string line;
      std::getline(ss, line, ' ');
      if (line.length() > 0)
        numStr.push_back(line);
    }
  }
  if (numStr.size() % 2 || numStr.size() == 0)
    return 1;
  
  cv::Mat pts1(numStr.size() / 2, 2, CV_32F);
  float *ptr = pts1.ptr<float>();
  for (size_t i = 0; i < numStr.size(); i++)
  {
    if (1 != sscanf(numStr[i].c_str(), "%f", ptr + i))
      return 2;
  }

  pts = pts1;

  img = cv::imread(line.substr(0, pos), CV_LOAD_IMAGE_COLOR);
  if (img.empty())
    return 3;

  return 0;
}

static void parse_parameters(int argc, char **argv)
{
  { // src type
    std::string typeStr(argv[1]);
    if (typeStr == "pts")
      src_type = PTS;
    else if (typeStr == "celeba")
      src_type = CELEBA;
    else if (typeStr == "occpts")
      src_type = OCCPTS;
    else
      LOG(FATAL) << "unknown src type";

    if (src_type == PTS)
      lineProcessor = ptsLineProcessor;
    else if (src_type == CELEBA)
      lineProcessor = celebaLineProcessor;
    else if (src_type == OCCPTS)
      lineProcessor = occluPtsLineProcessor;
    else
      LOG(FATAL) << "unknown src type";
  }
  
  src_path = argv[2];

  if (argc > 3)
  { // label
    if (src_type == CELEBA)
    {
      if (1 != sscanf(argv[3], "%d", &global_label))
        LOG(FATAL) << "unknown label";
      CHECK_GE(global_label, 0);
    }
  }

  if (argc > 4)
  { // is_color
    std::string isColorStr(argv[4]);
    if (isColorStr == "gray")
      param.is_color = false;
    else if (isColorStr == "color")
      param.is_color = true;
    else
      LOG(FATAL) << "specify gray or color";
  }

  if (argc > 5)
  { // net input size
    if (1 != sscanf(argv[5], "%d", &(param.net_input_size)))
      LOG(FATAL) << "net input size should be a integer";
    CHECK_GT(param.net_input_size, 0);
  }

  if (argc > 6)
  { // max rotate
    if (1 != sscanf(argv[6], "%f", &(param.max_rotate)))
      LOG(FATAL) << "max rorate should be a positive float number";
    CHECK_GT(param.max_rotate, 0.f);
  }

  if (argc > 7)
  { // min crop
    if (1 != sscanf(argv[7], "%f", &(param.min_crop_size)))
      LOG(FATAL) << "min crop should be a positive float number";
    CHECK_GT(param.min_crop_size, 0.f);
  }

  if (argc > 8)
  { // min crop
    if (1 != sscanf(argv[8], "%f", &(param.max_crop_size)))
      LOG(FATAL) << "max crop should be a positive float number";
    CHECK_GE(param.max_crop_size, param.min_crop_size);
  }
}

int main(int argc, char **argv)
{
  if (argc < 3 || argc > 9)
  {//                           0  1                 2        3       4                  5                    6               7               8
    fprintf(stderr, "Usage:\n  %s pts/celeba/occpts src_path [label] [color/gray=color] [net_input_size=112] [max_rotate=10] [min_crop=0.85] [max_crop=1.6]\n", argv[0]);
    return 1;
  }
  
  parse_parameters(argc, argv);

  std::string dstPath = src_path + ".lmdb";

  std::ifstream list(src_path.c_str());
  if (!list.good())
    LOG(FATAL) << "fail to open list " << src_path;

  LMDBWriter lmdb(dstPath);
  std::string line;
  cv::Mat previousPts;  
  cv::Mat previousExtra;
 
  for (size_t i = 0; !list.eof(); i++)
  {
    std::getline(list, line);
    if (line.length() == 0)
      continue;

    cv::Mat img, pts, processed_img, processed_pts, extra;
    int label, code;
    if (0 != (code = lineProcessor(line, img, pts, extra, label)) )
    {
      LOG(ERROR) << "error process line with code " << code << ": " << line;
      continue;
    }

    if (previousPts.data && previousPts.rows != pts.rows)
      LOG(FATAL) << "#points dismatch: " << previousPts.rows << " vs " << pts.rows;
    previousPts = pts;
    
    if (!extra.empty() && extra.rows != 1)
      LOG(FATAL) << "extra_data should be stored in an array";
    
    if (previousExtra.data && (extra.empty() || previousExtra.cols != extra.cols))
      LOG(FATAL) << "#extra_data dismatch: " << previousExtra.cols << " vs " << extra.cols;
    previousExtra = extra;

    //LOG(INFO) << "pts.size " << pts.rows << ", " << pts.cols;
    scale_down(img, pts, param, processed_img, processed_pts);
    caffe::Datum msg;
    if (processed_img.empty())
    {
      LOG(INFO) << "Invalid input image or pts at line: " << line;
      continue;
    }
    align_data_2_datum(processed_img, processed_pts, extra, param.is_color ? 3 : 1, label, msg);
    lmdb.put(i, msg);
    if (i % commit_size == 0)
      lmdb.commit();
  }

}
