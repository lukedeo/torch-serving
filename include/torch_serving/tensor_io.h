//
// Created by Luke de Oliveira on 2019-08-08.
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
class TensorIOError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TensorTypeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TensorShapeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

json::json TorchValueToJson(const torch::jit::IValue &torch_value);

std::vector<torch::jit::IValue> JsonToTorchValue(const json::json &payload);

}  // namespace torch_serving

#endif
