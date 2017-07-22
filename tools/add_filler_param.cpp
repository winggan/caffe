#include <string>
using std::string;

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/upgrade_proto.hpp>
#include <glog/logging.h>
#include <map>
#include <cmath>

static void resnet_conv_mod(caffe::NetParameter &net)
{
  for (int i = 0; i < net.layer().size(); i++)
  {
    if (net.layer(i).type() != "Convolution")
      continue;

    caffe::ConvolutionParameter *param = net.mutable_layer(i)->mutable_convolution_param();
    if (!param->has_weight_filler())
    {
      param->set_allocated_weight_filler(new caffe::FillerParameter);
    }
    param->mutable_weight_filler()->mutable_type()->operator=("gaussian");
    int kernel_size, stride;
    {
      CHECK_EQ(param->kernel_size_size(), 1);
      CHECK_EQ(param->stride_size(), 1);
      kernel_size = param->kernel_size(0);
      stride = param->stride(0);
      CHECK(param->has_num_output());
    }
    
    float std = sqrtf(2.f / (kernel_size * kernel_size * stride * param->num_output()));
    param->mutable_weight_filler()->set_std(std);
    LOG(INFO) << net.layer(i).name() << ": std = " << std;
  }
}

namespace caffe
{
  typedef void(*modifier)(caffe::NetParameter &net);
  static std::map<std::string, modifier> modifiers;

  struct modifierTuple { const char* name; modifier mod; };
  static modifierTuple modList[] =
  {
    {"resnet_conv", resnet_conv_mod},
    {0, 0}
  };

  void initModifiers()
  {
    for (int i = 0; modList[i].name; i++)
      modifiers[modList[i].name] = modList[i].mod;
  }
} // namespace caffe


int main(int argc, char **argv)
{
  caffe::initModifiers();
  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s {src_net.prototxt} {mod1} [{mod2} ... ]\n", argv[0]);
    return 1;
  }

  caffe::NetParameter net;
  caffe::ReadNetParamsFromTextFileOrDie(argv[1], &net);
  for (int i = 2; i < argc; i++)
  {
    if (caffe::modifiers.end() == caffe::modifiers.find(argv[i]))
    {
      LOG(ERROR) << "no mod called " << argv[i];
      continue;
    }
    caffe::modifiers[argv[i]](net);
  }

  fprintf(stdout, "%s\n", net.DebugString().c_str());
}
