#include <string>                           
#include <vector>                           
                                            
#include "boost/algorithm/string.hpp"       
#include "google/protobuf/text_format.h"   
 
#include "caffe/layers/dense_block_layer.hpp"                                            
#include "caffe/blob.hpp"                   
#include "caffe/common.hpp"                 
#include "caffe/net.hpp"                    
#include "caffe/proto/caffe.pb.h"           
#include "caffe/util/db.hpp"                
#include "caffe/util/format.hpp"            
#include "caffe/util/io.hpp"               

#include <ctime> 
#include <cmath>

typedef float Dtype;
using namespace caffe;

static void dump_array(const char *comment, int count, const Dtype* arr)
{
  vector<int> shape(1, count);
  Blob<Dtype> aa;
  aa.Reshape(shape);
  caffe_copy(count, arr, aa.mutable_cpu_data());
  const Dtype* a_ptr = aa.cpu_data();
  fprintf(stderr, "%s: ", comment);
  for (int i = 0; i < count; i ++)
    fprintf(stderr, "%f ", a_ptr[i]);
  fprintf(stderr, "\n");
}

template <typename T>
T array_max_diff(const int count, const T* a, const T*b)
{
  LOG(INFO) << a[0] << " " << b[0];
  T max_diff = T(0);
  for (int i = 0; i < count; i ++)
  {
    T diff = std::abs(a[i] - b[i]);
    max_diff = (diff > max_diff) ? diff : max_diff;
  }
  return max_diff;
}

static bool read_file(const char* filepath, std::string& data)
{
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "read_file %s failed\n", filepath);
        return false;
    }

    fseek(fp, 0, SEEK_END);
    int len = ftell(fp);
    rewind(fp);

    data.resize(len);
    fread((char*)data.data(), 1, len, fp);

    fclose(fp);

    return true;
}


int main1(int argc, char **argv)
{
  if (argc == 4 && std::string(argv[3]) == "GPU")
  {
    LOG(INFO) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  }
  unsigned int seed = time(NULL);
  fprintf(stderr, "set random seed: %u\n", seed);
  Caffe::set_random_seed(seed);

  boost::shared_ptr<Net<Dtype> > net(new Net<Dtype>(argv[1], caffe::TRAIN));
  for (int i = 0; i < net->layers().size(); i ++)
    fprintf(stderr, "%3d %s\n", i, net->layer_names()[i].c_str());

  std::string dense_block_name(argv[2]), cls_loss_name = "cls_loss";
  int dense_block_id = -1;
  int plain_dense_block_start = -1;
  int fc_id = -1, fc_2_id = -1;
  for (int i = 0; i < net->layers().size(); i ++)
  {
    if (net->layer_names()[i] == dense_block_name)
      dense_block_id = i;
    
    else if (net->layer_names()[i] == cls_loss_name)
      plain_dense_block_start = i + 1;
    
    else if (net->layer_names()[i] == "fc")
      fc_id = i;

    else if (net->layer_names()[i] == "fc_2")
      fc_2_id = i;
  }
 
  if (dense_block_id < 0 || plain_dense_block_start < 0 || fc_id < 0 || fc_2_id < 0)
    return -1;
 
  fprintf(stderr, "%3d %s\n", dense_block_id, net->layer_names()[dense_block_id].c_str());
  fprintf(stderr, "%3d %s\n", plain_dense_block_start, net->layer_names()[plain_dense_block_start].c_str());
  fprintf(stderr, "%3d %s\n", fc_id, net->layer_names()[fc_id].c_str());
  fprintf(stderr, "%3d %s\n", fc_2_id, net->layer_names()[fc_2_id].c_str());
  
  int num_param_plain = 0;
  for (int i = plain_dense_block_start; i < fc_2_id; i ++)
  {
    fprintf(stderr, "%s have %d param blobs\n", net->layer_names()[i].c_str(), (int)(net->layers()[i]->blobs().size()));
    num_param_plain += net->layers()[i]->blobs().size();
  }
  
  fprintf(stderr, "total have %d param blobs\n", num_param_plain);
  fprintf(stderr, "%s have %d param blobs\n", dense_block_name.c_str(), (int)(net->layers()[dense_block_id]->blobs().size()));
  
  for (int i = plain_dense_block_start, offset = 0; i < fc_2_id; i ++)
  {
    for (int b = 0; b < net->layers()[i]->blobs().size(); b ++, offset ++)
    {
      LOG(INFO) << net->layers()[dense_block_id]->blobs()[offset]->shape_string() << " vs " << 
        net->layers()[i]->blobs()[b]->shape_string();
      CHECK(net->layers()[dense_block_id]->blobs()[offset]->shape() == net->layers()[i]->blobs()[b]->shape())
        << "param blob shape not match at " << b << " of (" << i << ") " << net->layer_names()[i];
      caffe_copy(net->layers()[dense_block_id]->blobs()[offset]->count(),
                 net->layers()[dense_block_id]->blobs()[offset]->cpu_data(),
                 net->layers()[i]->blobs()[b]->mutable_cpu_data() );
    }
  }

  for (int b = 0; b < net->layers()[fc_id]->blobs().size(); b ++)
  {
    LOG(INFO) << "fc: " << net->layers()[fc_id]->blobs()[b]->shape_string() << " vs " <<
      net->layers()[fc_2_id]->blobs()[b]->shape_string();
    CHECK(net->layers()[fc_id]->blobs()[b]->shape() == net->layers()[fc_2_id]->blobs()[b]->shape())
      << "param blob shape not match between fc & fc_2"; 
    caffe_copy(net->layers()[fc_id]->blobs()[b]->count(),
               net->layers()[fc_id]->blobs()[b]->cpu_data(),
               net->layers()[fc_2_id]->blobs()[b]->mutable_cpu_data() );
  }

  fprintf(stderr, "param copy complete\n");
  
  { // store the initial parameters
    NetParameter __net_param;
    net->ToProto(&__net_param, false);
    WriteProtoToBinaryFile(__net_param, std::string(argv[1]) + ".caffemodel");
  }

  Dtype total_loss;
  for (int tt = 0; tt < 10; tt ++)
  {
    const vector<Blob<Dtype>*>& outputs = net->Forward(&total_loss);
    fprintf(stderr, "total_loss = %f\n", total_loss);
    for (size_t i = 0; i < outputs.size(); i ++)
    {
      fprintf(stderr, "%s: %f\n", outputs[i]->shape_string().c_str(), outputs[i]->cpu_data()[0]);
    }
    //const Dtype* d1 = ((DenseBlockLayer<Dtype>*)net->layers()[dense_block_id].get())->maps_diff_.cpu_data();
    //const Dtype* d2 = net->blob_by_name("dense_test_concat_0")->cpu_data();
    //const int d_count = net->blob_by_name("dense_test_concat_0")->count();
    //vector<int> shape = net->blob_by_name("dense_test_concat_0")->shape();
    //const int c_stride = shape[2]*shape[3], n_stride = c_stride * shape[1];
    //LOG(INFO) << "d1 = " << d1 << ", d2 = " << d2;
    //for (int d = 0; d < d_count; d ++)
    //  fprintf(stdout, "%4d %4d %4d  %f %f %f\n", d / n_stride, d % n_stride / c_stride, d % c_stride, d1[d] - d2[d], d1[d], d2[d]);
    LOG(INFO) << "output_diff = "
              << array_max_diff(net->blob_by_name("yixixi_output")->count(),
                                net->blob_by_name("yixixi_output")->cpu_data(),
                                net->blob_by_name("yixixi_output_2")->cpu_data());

    //dump_array("plain", net->blob_by_name("dense_test_concat_0")->count(), net->blob_by_name("dense_test_concat_0")->gpu_data());
 
    net->Backward();    
 
    for (int i = plain_dense_block_start, offset = 0; i < fc_2_id; i ++)
    {
      LOG(INFO) << "Layer " << net->layer_names()[i];
      for (int b = 0; b < net->layers()[i]->blobs().size(); b ++, offset ++)
      {
        CHECK(net->layers()[dense_block_id]->blobs()[offset]->shape() == net->layers()[i]->blobs()[b]->shape())
          << "param blob shape not match at " << b << " of (" << i << ") " << net->layer_names()[i];
        Dtype max_diff = array_max_diff(net->layers()[dense_block_id]->blobs()[offset]->count(),
                                        net->layers()[dense_block_id]->blobs()[offset]->cpu_diff(),
                                        net->layers()[i]->blobs()[b]->cpu_diff() );
        LOG(INFO) << "[" << b << "] " << net->layers()[dense_block_id]->blobs()[offset]->shape_string()  
                  << " max_diff = " << max_diff;  
      }
    }
     
  }
  return 0;
}
                    
int main(int argc, char **argv)
{
  if (argc >= 3)
    return main1(argc, argv);
  
  LayerParameter fc_param;
  LayerParameter softmax_param;

  {
    fc_param.set_name("fc");
    fc_param.set_type("InnerProduct");
    fc_param.add_bottom("yixixi_output");
    fc_param.add_top("predict");
    {
      InnerProductParameter* inner_param = fc_param.mutable_inner_product_param();
      inner_param->set_num_output(10);
      inner_param->mutable_weight_filler()->set_type("msra");
    }
  
    softmax_param.set_name("cls_loss");
    softmax_param.set_type("SoftmaxWithLoss");
    softmax_param.add_bottom("predict");
    softmax_param.add_bottom("wahaha_label");
    softmax_param.add_top("cls_loss");
  }

  boost::shared_ptr<Net<Dtype> > net(new Net<Dtype>(argv[1], caffe::TEST));    
  if (net->layers().size() <= 1)
  {
     LOG(ERROR) << "no layers";
     return 1;
  }
  std::string template_content;
  if (!read_file(argv[1], template_content))
  {
     LOG(ERROR) << "load template error";
     return -1;
  }
  boost::shared_ptr<Layer<Dtype> > layer = net->layer_by_name("dense_test");

  if (!layer || std::string(layer->type()) != "DenseBlock")
  {
    LOG(ERROR) << "not DenseBlockLayer";
    return -1;
  }
  
  DenseBlockLayer<Dtype>* ptr = (DenseBlockLayer<Dtype>*)(layer.get());
  
  vector<LayerParameter> params;
  ptr->convertToPlainLayers(params);
  
  NetParameter new_net;
  {
    size_t i = params.size() - 2;
    LayerParameter &p = params[i];
    CHECK(p.top(0) == "yixixi_output") << "invalid template";
    p.set_top(0, p.top(0) + "_2");
  }
  {
    size_t i = params.size() - 1;
    LayerParameter &p = params[i];
    CHECK(p.bottom(0) == "yixixi_output") << "invalid template";
    p.set_bottom(0, p.bottom(0) + "_2");
    CHECK(p.top(0) == "yixixi_output") << "invalid template";
    p.set_top(0, p.top(0) + "_2");
  }
  new_net.add_layer()->CopyFrom(fc_param);
  new_net.add_layer()->CopyFrom(softmax_param);
  for (size_t i = 0; i < params.size(); i ++)
    new_net.add_layer()->CopyFrom(params[i]);
  {
    fc_param.set_name(fc_param.name() + "_2");
    fc_param.set_bottom(0, fc_param.bottom(0) + "_2");
    fc_param.set_top(0, fc_param.top(0) + "_2");

    softmax_param.set_name(softmax_param.name() + "_2");
    softmax_param.set_bottom(0, softmax_param.bottom(0) + "_2");
    softmax_param.set_top(0, softmax_param.top(0) + "_2"); 
  }
  new_net.add_layer()->CopyFrom(fc_param);
  new_net.add_layer()->CopyFrom(softmax_param);
 
  std::string output_cfg("\n");
  output_cfg += new_net.DebugString();
  fprintf(stdout, template_content.c_str(), output_cfg.c_str());
  
  return 0; 
}
