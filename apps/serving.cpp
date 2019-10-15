#include <memory>

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks-inl.h>

#include "extern/optionparser.h"

#include "torch_serving/model_server.h"

optionparser::OptionParser GetConfiguration(int argc, const char* argv[]) {
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

  parser.eat_arguments(argc, argv);
  return parser;
}

int main(int argc, const char* argv[]) {
  std::shared_ptr<spdlog::logger> logger =
      spdlog::stdout_color_mt("torch_serving");
  logger->info("Starting torch-serving");

  auto config = GetConfiguration(argc, argv);

  auto model_capacity = config.get_value<int>("model-capacity");
  auto buffer_size = config.get_value<int>("buffer-size");
  auto threads = config.get_value<int>("threads");
  auto host = config.get_value<std::string>("host");
  auto port = config.get_value<int>("port");

  torch_serving::ModelServer model_server(model_capacity, buffer_size, threads);
  model_server.RunServer(host, port);
}
