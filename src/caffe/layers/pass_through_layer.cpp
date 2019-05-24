#include "caffe/layers/pass_through_layer.hpp"

namespace caffe {

template <typename Dtype>
void PassThroughLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(false, this->layer_param_.pass_through_param().has_num_output()) << "num_output is deprecate";
    CHECK(this->layer_param_.pass_through_param().has_block_height());
    CHECK(this->layer_param_.pass_through_param().has_block_width());

    int stride_h = this->layer_param_.pass_through_param().block_height();
    int stride_w = this->layer_param_.pass_through_param().block_width();
    CHECK_GT(stride_h, 0);
    CHECK_GT(stride_w, 0);

    ReorganizeParameter* reorg_param = this->layer_param_.mutable_reorg_param();
    reorg_param->set_is_flatten(false);
    reorg_param->set_stride_h(stride_h);
    reorg_param->set_stride_w(stride_w);

    ReorganizeLayer<Dtype>::LayerSetUp(bottom, top);
}

INSTANTIATE_CLASS(PassThroughLayer);
REGISTER_LAYER_CLASS(PassThrough);

} // namespace caffe