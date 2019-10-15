//
// Created by Luke de Oliveira on 2019-08-08.
//

#ifndef TORCH_SERVING__MODELSERVER_H_
#define TORCH_SERVING__MODELSERVER_H_

#include "extern/httplib.h"
#include "extern/json.hpp"
#include "servable_manager.h"

namespace json = nlohmann;

namespace torch_serving {

class ModelServer {
 public:
  explicit ModelServer(const size_t &model_capacity = 10,
                       const size_t &buffer = 0,
                       const size_t &thread_pool_size = 8);

  void RunServer(const std::string &host = "localhost", const int &port = 8888);

 private:
  static std::string GetHTTPMessageFromCode(const int &code);

  static void SetResponse(httplib::Response &response, const int &code,
                          const std::string &description = "",
                          const json::json &payload = json::json::object());

  void SetupEndpoints();

  httplib::Server server_;
  ServableManager servable_manager_;
  std::shared_ptr<spdlog::logger> logger_;
  std::shared_ptr<httplib::ThreadPool> thread_pool_;
};

}  // namespace torch_serving

#endif  // TORCH_SERVING__MODELSERVER_H_
