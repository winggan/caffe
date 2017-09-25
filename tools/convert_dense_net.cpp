#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/dense_block_layer.hpp"
#include <cstdio>

using namespace caffe;

typedef float Dtype;

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    LOG(ERROR) << "Usage: " << argv[0] << " {input.prototxt} {input.caffemodel} {output.prototxt} {output.caffemodel}";
    return argc;
  }
  

  NetParameter input_weigths;
  ReadNetParamsFromBinaryFileOrDie(argv[2], &input_weights);
  NetParameter input_net;
  ReadNetParamsFromTextFileOrDie(argv[1], &input_net);
  
  boost::shared_ptr<Net<Dtype> > the_net(
    new Net<Dtype>(argv[1], caffe::TRAIN)
  );

  NetParameter output_net;
  
  {
    Net<Dtype> &net = *(the_net.get());
    net.CopyTrainedLayersFrom(input_weigths);
    
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
          for (size_t i = 0; i < dense_layers.size(); i++)
            output_net.add_layer()->CopyFrom(dense_layers[i]);
        }
        else
          output_net.add_layer()->CopyFrom(input_net.layer(layer_id));
      }
      FILE *fp = fopen(argv[3], "wb");
      CHECK_NE(fp, NULL) << "fail to open file " << argv[3];
      fprintf(fp, "%s\n", output_net.DebugString().c_str());
      fclose(fp);
    }

    boost::shared_ptr<Net<Dtype> > the_out_net(
      new Net<Dtype>(argv[3], caffe::TRAIN)
    );

    Net<Dtype> &out_net = *(the_out_net.get());

    { // copy weights to output net
      int dst_id = 0;
      for (int src_id = 0; src_id < net.layers().size(); src++)
      {
        Layer<Dtype>* src_layer = net.layers()[src_id].get();
        if (src_layer->layer_param().type() == "DenseBlock")
        {
          int src_offset = 0;
          while (src_offset < src_layer->blobs().size())
          {
            Layer<Dtype>* dst_layer = out_net.layers()[dst_id].get();
            CHECK_LE(dst_layer->blobs().size() + src_offset, src_layer->blobs().size())
              << "#blobs dismatch within DenseBlock " << src_layer->layer_param().name();

            for (int blob_id = 0; blob_id < dst_layer->blobs().size(); blob_id++)
            {
              CHECK(src_layer->blobs()[blob_id + src_offset]->shape() 
                 == dst_layer->blobs()[blob_id]->shape()) 
                << "blob shape dismatch at src(" << src_id << ") vs dst(" << dst_id ")"
                << " blob[" << blob_id << "]";
              caffe_copy(src_layer->blobs()[blob_id + src_offset]->count(),
                         src_layer->blobs()[blob_id + src_offset]->cpu_data(),
                         dst_layer->blobs()[blob_id]->mutable_cpu_data());
            }

            src_offset += dst_layer->blobs().size();
            dst_id++;
          }
        }
        else
        {
          Layer<Dtype>* dst_layer = out_net.layers()[dst_id].get();
          CHECK_EQ(src_layer->blobs().size(), dst_layer->blobs().size())
            << "#blobs dismatch at src(" << src_id << ") vs dst(" << dst_id ")";
          for (int blob_id = 0; blob_id < src_layer->blobs().size(); blob_id++)
          {
            CHECK(src_layer->blobs()[blob_id]->shape() 
               == dst_layer->blobs()[blob_id]->shape()) 
              << "blob shape dismatch at src(" << src_id << ") vs dst(" << dst_id ")"
              << " blob[" << blob_id << "]";
            caffe_copy(src_layer->blobs()[blob_id]->count(),
                       src_layer->blobs()[blob_id]->cpu_data(),
                       dst_layer->blobs()[blob_id]->mutable_cpu_data());
          }
          dst_id ++;
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
