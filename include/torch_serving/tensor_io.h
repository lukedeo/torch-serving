//
// (c) 2019, Luke de Oliveira
// This code is licensed under MIT license (see LICENSE for details)
//

#ifndef TORCH_SERVING__TENSOR_IO_H_
#define TORCH_SERVING__TENSOR_IO_H_

#include <torch/script.h>

#include <unordered_map>
#include <utility>

#include "extern/json.hpp"

namespace json = nlohmann;

namespace torch_serving {

// Use shallowly overloaded errors from the STL just to make error defs easier
class TensorIOError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TensorTypeError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TensorShapeError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

json::json TorchValueToJson(const torch::jit::IValue &torch_value);

std::vector<torch::jit::IValue> JsonToTorchValue(
    const json::json &payload, const at::Device &device = at::kCPU);

}  // namespace torch_serving

#endif
