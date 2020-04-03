//
// (c) 2019, Luke de Oliveira
// This code is licensed under MIT license (see LICENSE for details)
//

#ifndef TORCH_SERVING__SERVABLEMANAGER_H_
#define TORCH_SERVING__SERVABLEMANAGER_H_

#include <spdlog/logger.h>
//#include <torch/script.h>

#include <future>

#include "extern/LRUCache11.hpp"
#include "extern/json.hpp"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks-inl.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <future>
#include <random>
#include <type_traits>

namespace json = nlohmann;

namespace torch_serving {

template <typename ServableType>
class ServableManager {
  static_assert(std::is_same<decltype(std::declval<ServableType>().RunInference(
                                 std::declval<std::vector<json::json>>())),
                             json::json>::value,
                "ServableType must have method with signature "
                "RunInference(std::vector<json::json>) -> json::json.");
  static_assert(std::is_constructible<ServableType, std::string>::value,
                "ServableType must be constructible from single std::string.");

 public:
  ServableManager() : ServableManager(5, 0) {}

  explicit ServableManager(const size_t &size, const size_t &buffer_size = 0)
      : logger_(spdlog::get("servable_manager")),
        model_cache_(size, buffer_size) {
    if (!logger_) {
      logger_ = spdlog::stdout_color_mt("servable_manager");
    }
  }

  std::shared_ptr<ServableType> GetServable(
      const std::string &servable_identifier) {
    logger_->info("Loading servable from servable_identifier: " +
                  servable_identifier);
    if (model_cache_.contains(servable_identifier)) {
      logger_->info("Found model with servable_identifier: " +
                    servable_identifier + " in cache");
      return model_cache_.get(servable_identifier);
    }
    logger_->info("Cache miss detected for servable_identifier: " +
                  servable_identifier);
    std::shared_ptr<ServableType> servable;
    // Try to load the servable - if if fails, throw an error.
    try {
      servable = LoadServableFromIdentifier(servable_identifier);
    } catch (const std::exception &err) {
      const std::string msg("Failed to load from servable_identifier: " +
                            servable_identifier);
      logger_->warn(msg);
      throw std::invalid_argument(msg + err.what());
    }
    model_cache_.insert(servable_identifier, servable);
    logger_->info("Cache is now of size: " + std::to_string(Size()));
    return servable;
  }

  std::shared_ptr<ServableType> GetServable(
      const std::string &servable_identifier, const float &invalidation_prob) {
    std::mt19937 generator(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    if (distribution(generator) < invalidation_prob) {
      logger_->info("Bypassing cache and invaliding for servable_identifier: " +
                    servable_identifier);
      if (model_cache_.contains(servable_identifier)) {
        model_cache_.remove(servable_identifier);
      }
    }
    return GetServable(servable_identifier);
  }

  size_t Size() { return model_cache_.size(); }

  json::json InferenceRequest(const std::string &servable_identifier,
                              const json::json &input,
                              const float &invalidation_prob = 0.0) {
    try {
      return (invalidation_prob > 1e-5
                  ? GetServable(servable_identifier, invalidation_prob)
                  : GetServable(servable_identifier))
          ->RunInference(input);
    } catch (const std::exception &e) {
      if (model_cache_.contains(servable_identifier)) {
        logger_->warn("Removing servable_identifier: " + servable_identifier +
                      " from cache due to caught exception.");
        model_cache_.remove(servable_identifier);
      }
      throw;
    }
  }

  std::future<json::json> AsyncInferenceRequest(
      const std::string &servable_identifier, const json::json &input,
      const float &invalidation_prob = 0.0,
      std::launch policy = std::launch::async) {
    logger_->debug("Launching async inference request");
    return std::async(policy, [this, &servable_identifier, &input,
                               &invalidation_prob]() {
      logger_->debug("Running inference in a future.");
      return InferenceRequest(servable_identifier, input, invalidation_prob);
    });
  }

 private:
  static std::shared_ptr<ServableType> LoadServableFromIdentifier(
      const std::string &servable_identifier) {
    return std::make_shared<ServableType>(servable_identifier);
  }

  std::shared_ptr<spdlog::logger> logger_;

  // N.B., this uses a mutex so the insertion and retrieval of models into
  // model_cache_ is thread safe/
  lru11::Cache<std::string, std::shared_ptr<ServableType>> model_cache_;
};

}  // namespace torch_serving

#endif  // TORCH_SERVING__SERVABLEMANAGER_H_
