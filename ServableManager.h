//
// Created by Luke de Oliveira on 2019-08-08.
//

#ifndef TORCH_SERVING__SERVABLEMANAGER_H_
#define TORCH_SERVING__SERVABLEMANAGER_H_

#include <torch/script.h>
#include "extern/LRUCache11.hpp"
#include <future>
#include <random>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks-inl.h>
#include <spdlog/spdlog.h>

class ServableManager {
public:
  ServableManager() : ServableManager(5, 0) {}
  explicit ServableManager(const size_t &size, const size_t &buffer_size = 0)
      : size_(size), model_cache_(size, buffer_size),
        logger_(spdlog::get("ServableManager")) {
    if (!logger_) {
      logger_ = spdlog::stdout_color_mt("ServableManager");
    }
  }

  std::shared_ptr<torch::jit::script::Module>
  GetServable(const std::string &filepath) {
    logger_->info("Loading servable from filename: " + filepath);
    if (model_cache_.contains(filepath)) {
      logger_->info("Found model with filename: " + filepath + " in cache");
      return model_cache_.get(filepath);
    }
    logger_->info("Cache miss detected for filename: " + filepath);
    std::shared_ptr<torch::jit::script::Module> servable;
    // Try to load the servable - if if fails, throw an error.
    try {
      servable = LoadServableFromDisk(filepath);
    } catch (const std::exception &err) {
      const std::string msg("Failed to load from filename: " + filepath);
      logger_->warn(msg);
      throw std::invalid_argument(msg + err.what());
    }
    model_cache_.insert(filepath, servable);
    logger_->info("Cache is now of size: " + std::to_string(Size()));
    return servable;
  }

  std::shared_ptr<torch::jit::script::Module>
  GetServable(const std::string &filepath, const float &invalidation_prob) {

    std::mt19937 generator(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    if (distribution(generator) < invalidation_prob) {
      logger_->info("Bypassing cache and invaliding for filename: " + filepath);
      if (model_cache_.contains(filepath)) {
        model_cache_.remove(filepath);
      }
    }
    return GetServable(filepath);
  }

  size_t Size() { return model_cache_.size(); }

  torch::jit::IValue
  InferenceRequest(const std::string &filepath,
                   const std::vector<torch::jit::IValue> &input,
                   const float &invalidation_prob = 0.0) {
    return (invalidation_prob > 1e-5 ? GetServable(filepath, invalidation_prob)
                                     : GetServable(filepath))
        ->forward(input);
  }

  std::future<torch::jit::IValue>
  AsyncInferenceRequest(const std::string &filepath,
                        const std::vector<torch::jit::IValue> &input,
                        const float &invalidation_prob = 0.0,
                        std::launch policy = std::launch::async) {
    logger_->info("Launching async inference request");
    return std::async(policy, [this, &filepath, &input, &invalidation_prob]() {
      logger_->info("Running inference in a future.");
      return InferenceRequest(filepath, input, invalidation_prob);
    });
  }

private:
  std::shared_ptr<torch::jit::script::Module>
  LoadServableFromDisk(const std::string &filepath) {
    auto module = torch::jit::load(filepath);
    if (module == nullptr || !module) {
      throw std::runtime_error("Failed to load Module from filename: " +
                               filepath);
    }
    logger_->info("Servable from filename: " + filepath +
                  " successfully loaded from disk");
    return module;
  }

  std::shared_ptr<spdlog::logger> logger_;

  size_t size_;

  // N.B., this uses a mutex so the insertion and retrieval of models into
  // model_cache_ is thread safe/
  lru11::Cache<std::string, std::shared_ptr<torch::jit::script::Module>>
      model_cache_;
};

#endif // TORCH_SERVING__SERVABLEMANAGER_H_
