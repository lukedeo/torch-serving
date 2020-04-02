//
// Created by Luke deOliveira on 1/7/20.
//

#ifndef TORCH_SERVING_INCLUDE_TORCH_SERVING_TORCH_JIT_SERVABLE_H_
#define TORCH_SERVING_INCLUDE_TORCH_SERVING_TORCH_JIT_SERVABLE_H_

#include <torch/script.h>
#include <torch/torch.h>

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks-inl.h>

#include "extern/json.hpp"
#include "tensor_io.h"

namespace json = nlohmann;

namespace torch_serving {

class TorchJITServable {
 public:
  explicit TorchJITServable(std::string path)
      : m_path(std::move(path)),
        m_servable(LoadServable(m_path)),
        m_logger(spdlog::get("servable_manager")) {
    if (!m_logger) {
      m_logger = spdlog::stdout_color_mt("torch_jit_servable");
    }
  }

  json::json RunInference(const json::json &input) {
    return TorchValueToJson(
        m_servable.forward(JsonToTorchValue(input, m_device)));
  }

  torch::jit::script::Module LoadServable(const std::string &path) {
//    std::cout << "WE REACH HERE" << std::endl;
    auto module = torch::jit::load(path);

    module.eval();
    if (at::hasCUDA()) {
      m_logger->debug("CUDA detected. Moving to GPU.");
      module.to(at::kCUDA);
      m_device = at::kCUDA;
    } else {
      m_logger->debug("Running on CPU only.");
      m_device = at::kCPU;
    }
//    std::cout << "WE REACH HERE ENDDDD" << std::endl;
    return module;
  }
  ~TorchJITServable() { m_servable.to(at::kCPU, false); }

 private:
  std::string m_path;
  torch::jit::script::Module m_servable;
  std::shared_ptr<spdlog::logger> m_logger;
  at::Device m_device = at::kCPU;
};

}  // namespace torch_serving
#endif  // TORCH_SERVING_INCLUDE_TORCH_SERVING_TORCH_JIT_SERVABLE_H_
