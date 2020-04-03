//
// (c) 2020, Luke de Oliveira
// This code is licensed under MIT license (see LICENSE for details)
//

#include <spdlog/sinks/stdout_color_sinks-inl.h>

#include <memory>

#include "extern/optionparser.h"
#include "torch_serving/model_server.h"
#include "torch_serving/torch_jit_servable.h"

optionparser::OptionParser GetConfiguration(int argc, const char *argv[]) {
  optionparser::OptionParser parser(
      "torch-serving: a minimal HTTP serving layer for models created with "
      "PyTorch and exported to TorchScript.");

  parser.add_option("--host", "-h")
      .help("Host to run the server on.")
      .default_value("localhost")
      .mode(optionparser::STORE_VALUE);

  parser.add_option("--port", "-p")
      .help("Port to run the server on.")
      .default_value(8888)
      .mode(optionparser::STORE_VALUE);

  parser.add_option("--model-capacity", "-c")
      .help(
          "Maximum number of models or servables to remain in memory at any "
          "point in time.")
      .default_value(10)
      .mode(optionparser::STORE_VALUE);

  parser.add_option("--buffer-size", "-b")
      .help("Soft buffer around model capacity for eviction from cache (LRU)")
      .default_value(3)
      .mode(optionparser::STORE_VALUE);

  parser.add_option("--threads", "-t")
      .help("Number of concurrent threads to use in the serving layer.")
      .default_value(8)
      .mode(optionparser::STORE_VALUE);

  parser.add_option("--use-gpu")
      .help("Whether or not to use CUDA GPUs.")
      .mode(optionparser::STORE_TRUE);

  parser.eat_arguments(argc, argv);
  return parser;
}

int main(int argc, const char *argv[]) {
  auto logger = spdlog::stdout_color_mt("torch_serving");
  logger->info("Starting torch-serving");

  auto config = GetConfiguration(argc, argv);

  auto model_capacity = config.get_value<int>("model-capacity");
  auto buffer_size = config.get_value<int>("buffer-size");
  auto threads = config.get_value<int>("threads");
  auto host = config.get_value<std::string>("host");
  auto port = config.get_value<int>("port");
  auto use_gpu = config.get_value<bool>("use-gpu");

  if (!use_gpu) {
    torch_serving::ModelServer<torch_serving::TorchJITServable> model_server(
        model_capacity, buffer_size, threads);
    model_server.RunServer(host, port);
  } else {
    torch_serving::ModelServer<torch_serving::TorchJITCudaServable>
        model_server(model_capacity, buffer_size, threads);
    model_server.RunServer(host, port);
  }
}
