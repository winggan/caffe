#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/io.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/dense_block_layer.hpp"
#include <cstdio>
#include <map>

using namespace caffe;

typedef float Dtype;

static Layer<Dtype>* get_layer(const Net<Dtype> &net, const std::string &name)
{
  Layer<Dtype>* ret = NULL;
  ret = net.layer_by_name(name).get();
  CHECK(ret) << "Layer " << name << " does not exists";
  return ret;
}

int main(int argc, char **argv)
{
  if (argc != 5)
  {
    LOG(ERROR) << "Usage: " << argv[0] << " {input.prototxt} {input.caffemodel} {output.prototxt} {output.caffemodel}";
    return argc;
  }
  

  NetParameter input_weights;
  ReadNetParamsFromBinaryFileOrDie(argv[2], &input_weights);
  NetParameter input_net;
  ReadNetParamsFromTextFileOrDie(argv[1], &input_net);
  
  boost::shared_ptr<Net<Dtype> > the_net(
    new Net<Dtype>(argv[1], caffe::TRAIN)
  );

  NetParameter output_net;
  
  {
    Net<Dtype> &net = *(the_net.get());
    net.CopyTrainedLayersFrom(input_weights);
    std::map<std::string, std::vector<std::string> > names_layer_dense;
     
    { // construct output prototxt
      output_net.CopyFrom(input_net);
      output_net.clear_layer();
      for (int layer_id = 0; layer_id < input_net.layer_size(); layer_id++)
      {
        if (input_net.layer(layer_id).type() == "DenseBlock")
        {
          const string &layer_name = input_net.layer(layer_id).name();
          DenseBlockLayer<Dtype>* dense_block =
            (DenseBlockLayer<Dtype>*)(net.layer_by_name(layer_name).get());
          std::vector<LayerParameter> dense_layers;
          dense_block->convertToPlainLayers(dense_layers);
          names_layer_dense[input_net.layer(layer_id).name()] = std::vector<std::string>();
          std::vector<std::string> &names = names_layer_dense[input_net.layer(layer_id).name()];
          for (size_t i = 0; i < dense_layers.size(); i++)
          {
            output_net.add_layer()->CopyFrom(dense_layers[i]);
            names.push_back(dense_layers[i].name());
          }
        }
        else
          output_net.add_layer()->CopyFrom(input_net.layer(layer_id));
      }
      FILE *fp = fopen(argv[3], "wb");
      CHECK(fp) << "fail to open file " << argv[3];
      fprintf(fp, "%s\n", output_net.DebugString().c_str());
      fclose(fp);
    }

    boost::shared_ptr<Net<Dtype> > the_out_net(
      new Net<Dtype>(argv[3], caffe::TRAIN)
    );

    Net<Dtype> &out_net = *(the_out_net.get());

    { // copy weights to output net
      for (int src_id = 0; src_id < net.layers().size(); src_id++)
      {
        Layer<Dtype>* src_layer = net.layers()[src_id].get();
        if (src_layer->layer_param().type() == "DenseBlock")
        {
          int src_offset = 0;
          CHECK(names_layer_dense.find(src_layer->layer_param().name()) != names_layer_dense.end())
            << "DenseBlockLayer " << src_layer->layer_param().name() << "does not exist";
          const std::vector<std::string> &names_dst_layers = names_layer_dense[src_layer->layer_param().name()];
          //while (src_offset < src_layer->blobs().size())
          for (size_t dense_i = 0; dense_i < names_dst_layers.size(); dense_i ++)
          {
            const std::string &dst_name(names_dst_layers[dense_i]);
            Layer<Dtype>* dst_layer = get_layer(out_net, dst_name);
            CHECK_LE(dst_layer->blobs().size() + src_offset, src_layer->blobs().size())
              << "#blobs dismatch within DenseBlock " << src_layer->layer_param().name();

            for (int blob_id = 0; blob_id < dst_layer->blobs().size(); blob_id++)
            {
              CHECK(src_layer->blobs()[blob_id + src_offset]->shape() 
                 == dst_layer->blobs()[blob_id]->shape()) 
                << "blob shape dismatch at src(" << (src_offset + blob_id) << ") vs dst(" << dst_name << ")"
                << " blob[" << blob_id << "]";
              caffe_copy(src_layer->blobs()[blob_id + src_offset]->count(),
                         src_layer->blobs()[blob_id + src_offset]->cpu_data(),
                         dst_layer->blobs()[blob_id]->mutable_cpu_data());
            }
            LOG(INFO) <<  dst_layer->blobs().size() << " " << dst_layer->layer_param().name();
            src_offset += dst_layer->blobs().size();
          }
          CHECK_EQ(src_offset, (int)(src_layer->blobs().size())) 
            << "DenseBlockLayer is not converted completely";
        }
        else
        {
          Layer<Dtype>* dst_layer = get_layer(out_net, src_layer->layer_param().name());
          CHECK_EQ(src_layer->blobs().size(), dst_layer->blobs().size())
            << "#blobs dismatch at " << src_layer->layer_param().name();
          for (int blob_id = 0; blob_id < src_layer->blobs().size(); blob_id++)
          {
            CHECK(src_layer->blobs()[blob_id]->shape() 
               == dst_layer->blobs()[blob_id]->shape()) 
              << "blob shape dismatch at " << src_layer->layer_param().name()
              << " blob[" << blob_id << "]";
            caffe_copy(src_layer->blobs()[blob_id]->count(),
                       src_layer->blobs()[blob_id]->cpu_data(),
                       dst_layer->blobs()[blob_id]->mutable_cpu_data());
          }
        }
      }
    }
    
    // save converted weights
    {
      NetParameter output_weights;
      out_net.ToProto(&output_weights, false);
      WriteProtoToBinaryFile(output_weights, argv[4]);
    }
  }

  return 0;
}
