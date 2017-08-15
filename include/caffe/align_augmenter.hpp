#ifndef CAFFE_ALIGN_AUGMENTER_HPP_
#define CAFFE_ALIGN_AUGMENTER_HPP_

#ifdef USE_OPENCV

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"


#include <opencv2/core/core.hpp>


namespace caffe {

/**
 * @brief Applies augmentation to input images of face alignment, including 
 * mirror, rotation, scaling(zooming), translation
 */
template <typename Dtype>
class AlignAugmenter {
  public: 
    explicit AlignAugmenter(const AlignAugmentationParameter &param, const TransformationParameter &transParam, Phase phase);
    virtual ~AlignAugmenter() {}
    
   /**
    * @brief Initialize the Random number generations if needed by the
    *    transformation.
    */
    void InitRand();
    
   /**
	  * @brief Perform augmentation to input image to generate a augmentation instance of 
	  * the input image, together with points' corresponding transformation 
	  *
	  * @param cv_img 
	  *    The input image to be augmented
	  * @param cv_pts
	  *    The list of points stored in cv::Mat whose size is [n x 2], n is the 
	  *    number of points, CV_64F
	  * @param [out] aug_cv_img
	  *    The augmented image
	  * @param [out] aug_cv_pts
	  *    The transformed points according to augmentation, same format as cv_pts
	  * 
	  */
    
	void Augment(const cv::Mat &cv_img, const cv::Mat &cv_pts, cv::Mat &aug_cv_img, cv::Mat &aug_cv_pts);
  
  cv::Mat Augment(const cv::Mat &cv_pts, const cv::Mat &cv_extra, 
    cv::Mat &aug_cv_pts, cv::Mat &aug_cv_extra, Caffe::RNG *provided_rng);
  inline cv::Mat Augment(const cv::Mat &cv_pts, const cv::Mat &cv_extra,
    cv::Mat &aug_cv_pts, cv::Mat &aug_cv_extra, )
  { return Augment(cv_pts, cv_extra, aug_cv_pts, aug_cv_extra, rng_.get()); }
  
  inline int width() {return width_;}
  inline int height() {return height_;}
  
  protected:
    virtual int Rand(int n);
    int Rand(Caffe::RNG *provided_rng, int n);
	  virtual Dtype getUniformRand(const Dtype &lowerBound, const Dtype &upperBound);
    Dtype getUniformRand(Caffe::RNG *provided_rng, const Dtype &lowerBound, const Dtype &upperBound);
   /**
    * @brief make first transformation matrix.
    *    Correspond to move center to (0,0) and rotate the specified angle (in degree)
    */
    cv::Mat makeRotate(const cv::Rect &originBoundBox, double angle, cv::Rect &resBoundBox);
  
   /**
    * @brief make second transformation matrix.
    *    move back according to random extended bounding box, then scale
    */
    cv::Mat makeRandomCropAndResize(const cv::Rect &randomBoundingRect, const cv::Size &targetSize);
    
   /**
    * @brief generate the random crop box, that still bounding all points (after rotation)
    *    and satisfying given extend range
    */
    cv::Rect generateRandomBoundingRect(const cv::Rect rotatedBoundingRect, 
        const float &extendBoxMin, const float &extendBoxMax, Caffe::RNG *provided_rng);
    inline cv::Rect generateRandomBoundingRect(const cv::Rect rotatedBoundingRect, 
        const float &extendBoxMin, const float &extendBoxMax)
    { return generateRandomBoundingRect(rotatedBoundingRect, extendBoxMin, extendBoxMax, rng_.get()); }
        
   /**
    * @brief make finally transformation matrix (stand alone).
    *    mirror operation
    */
    cv::Mat makeMirror(const int &width, const int &height);
    
   /**
    * @brief handle order of points after MIRROR warping, assuming CV_64F
    */
    void processMirrorPts(cv::Mat &pts, const std::vector<cv::Vec2i> &pairs);
    void processMirrorPtsf(cv::Mat &pts, const std::vector<cv::Vec2i> &pairs);
    
   /**
    * @brief do points transformation according to given transformation matrix
    */
    cv::Mat warpPointMat(const cv::Mat &ptsMat, const cv::Mat &trans);
    
    int height_, width_;
    AlignAugmentationParameter param_;
    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    
    std::vector<cv::Vec2i> mirrorPairs_;
    cv::Mat ptsMask_;
};
    
} // namespace caffe

#endif // USE_OPENCV

#endif  // CAFFE_ALIGN_AUGMENTER_HPP

//FATAL
