// TODO: Move this to a cmdline specified param
#define CPPHTTPLIB_THREAD_POOL_COUNT 8

#include "ModelServer.h"
#include <memory>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks-inl.h>

int main(int argc, const char *argv[]) {
  std::shared_ptr<spdlog::logger> logger =
      spdlog::stdout_color_mt("pytorch-serving");
  logger->info("Starting pytorch-serving");
  ModelServer model_server;
  model_server.RunServer();
}
