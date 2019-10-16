//
// Created by Luke de Oliveira on 2019-08-08.
//

#ifndef TORCH_SERVING__SERVABLEMANAGER_H_
#define TORCH_SERVING__SERVABLEMANAGER_H_

#include <spdlog/logger.h>
#include <torch/script.h>

#include <future>

#include "extern/LRUCache11.hpp"

namespace torch_serving {

class ServableManager {
 public:
  ServableManager();
  explicit ServableManager(const size_t &size, const size_t &buffer_size = 0);
  std::shared_ptr<torch::jit::script::Module> GetServable(
      const std::string &filepath);

  std::shared_ptr<torch::jit::script::Module> GetServable(
      const std::string &filepath, const float &invalidation_prob);

  size_t Size();

  torch::jit::IValue InferenceRequest(
      const std::string &filepath, const std::vector<torch::jit::IValue> &input,
      const float &invalidation_prob = 0.0);

  std::future<torch::jit::IValue> AsyncInferenceRequest(
      const std::string &filepath, const std::vector<torch::jit::IValue> &input,
      const float &invalidation_prob = 0.0,
      std::launch policy = std::launch::async);

 private:
  static std::shared_ptr<torch::jit::script::Module> LoadServableFromDisk(
      const std::string &filepath);

  std::shared_ptr<spdlog::logger> logger_;

  // N.B., this uses a mutex so the insertion and retrieval of models into
  // model_cache_ is thread safe/
  lru11::Cache<std::string, std::shared_ptr<torch::jit::script::Module>>
      model_cache_;
};

}  // namespace torch_serving

#endif  // TORCH_SERVING__SERVABLEMANAGER_H_
