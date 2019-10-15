//
// Created by Luke de Oliveira on 2019-08-08.
//

#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/jit.h>
#include <torch/types.h>

#include "torch_serving/tensor_io.h"

namespace json = nlohmann;

namespace torch_serving {

template <typename T>
std::vector<T> TensorToStdVector(const torch::Tensor &t) {
  std::vector<T> tensor_vec(t.data<T>(), t.data<T>() + t.numel());
  return tensor_vec;
}

torch::ScalarType StringToScalarType(const std::string &scalar_type) {
  if (scalar_type == "uint8") {
    return torch::ScalarType::Byte;
  } else if (scalar_type == "int8") {
    return torch::ScalarType::Char;
  } else if (scalar_type == "float64") {
    return torch::ScalarType::Double;
  } else if (scalar_type == "float32") {
    return torch::ScalarType::Float;
  } else if (scalar_type == "int32") {
    return torch::ScalarType::Int;
  } else if (scalar_type == "int64") {
    return torch::ScalarType::Long;
  } else if (scalar_type == "int16") {
    return torch::ScalarType::Short;
  } else if (scalar_type == "float16") {
    return torch::ScalarType::Half;
  } else if (scalar_type == "bool") {
    return torch::ScalarType::Bool;
  } else {
    throw TensorTypeError("Invalid type for Tensor: " + scalar_type);
  }
}

json::json TensorToJson(const torch::Tensor &tensor) {
  torch::IntArrayRef tensor_shape_ref = tensor.sizes();
  std::vector<int> tensor_shape(tensor_shape_ref.begin(),
                                tensor_shape_ref.end());
  std::string data_type;

  torch::ScalarType dtype = torch::typeMetaToScalarType(tensor.dtype());

  json::json payload = {{"type", "tensor"}, {"shape", tensor_shape}};

  switch (dtype) {
    case c10::ScalarType::Byte:
      payload["data_type"] = "uint8";
      payload["value"] = TensorToStdVector<uint8_t>(tensor);
      break;
    case c10::ScalarType::Char:
      payload["data_type"] = "int8";
      payload["value"] = TensorToStdVector<int8_t>(tensor);
      break;
    case c10::ScalarType::Short:
      payload["data_type"] = "int16";
      payload["value"] = TensorToStdVector<int16_t>(tensor);
      break;
    case c10::ScalarType::Int:
      payload["data_type"] = "int32";
      payload["value"] = TensorToStdVector<int>(tensor);
      break;
    case c10::ScalarType::Long:
      payload["data_type"] = "int64";
      payload["value"] = TensorToStdVector<int64_t>(tensor);
      break;
    case c10::ScalarType::Half:
      payload["data_type"] = "float16";
      payload["value"] = TensorToStdVector<float>(tensor);
      break;
    case c10::ScalarType::Float:
      payload["data_type"] = "float32";
      payload["value"] = TensorToStdVector<float>(tensor);
      break;
    case torch::ScalarType::Double:
      payload["data_type"] = "float64";
      payload["value"] = TensorToStdVector<double>(tensor);
      break;
    case c10::ScalarType::Bool:
      payload["data_type"] = "bool";
      payload["value"] = TensorToStdVector<bool>(tensor);
      break;
    default:
      throw TensorIOError("Invalid scalar type");
  }
  return payload;
}

json::json ScalarToJson(const torch::jit::IValue &torch_value) {
  auto scalar = torch_value.toScalar();
  auto dtype = torch::typeMetaToScalarType(
      torch::scalar_to_tensor(torch_value.toScalar()).dtype());

  std::string data_type;
  switch (dtype) {
    case torch::ScalarType::Double:
      data_type = "float64";
      break;
    case c10::ScalarType::Byte:
      data_type = "uint8";
      break;
    case c10::ScalarType::Char:
      data_type = "int8";
      break;
    case c10::ScalarType::Short:
      data_type = "int16";
      break;
    case c10::ScalarType::Int:
      data_type = "int32";
      break;
    case c10::ScalarType::Long:
      data_type = "int64";
      break;
    case c10::ScalarType::Half:
      data_type = "float16";
      break;
    case c10::ScalarType::Float:
      data_type = "float32";
      break;
    case c10::ScalarType::Bool:
      data_type = "bool";
      break;
    default:
      throw TensorIOError("Invalid scalar type");
  }

  json::json payload = {{"type", "scalar"}, {"data_type", data_type}};
  if (scalar.isFloatingPoint()) {
    payload["value"] = scalar.toFloat();
  } else if (scalar.isIntegral()) {
    payload["value"] = scalar.toInt();
  } else {
    throw TensorTypeError("Unimplemented scalar type");
  }
  return payload;
}

json::json GenericDictToJson(const torch::jit::IValue &torch_value) {
  json::json payload = {{"type", "generic_dict"}};
  const auto &dict = torch_value.toGenericDictRef();
  for (auto &entry : dict) {
    if (!entry.first.isString()) {
      throw TensorIOError(
          "Can only convert GenericDicts to Json if keys are string type");
    }
    payload["value"][entry.first.toStringRef()] =
        TorchValueToJson(entry.second);
  }
  return std::move(payload);
}

json::json TorchValueToJson(const torch::jit::IValue &torch_value) {
  if (torch_value.isTensor()) {
//    return TorchValueToJson(torch_value.toTensor());
    return TensorToJson(torch_value.toTensor());
  } else if (torch_value.isTensorList()) {
    auto tensor_list = torch_value.toTensorListRef();
    json::json payload = json::json::array();
    for (const auto &tensor : tensor_list) {
      payload.emplace_back(TensorToJson(tensor));
    }
    return payload;
  } else if (torch_value.isString()) {
    return {{"type", "string"}, {"value", torch_value.toStringRef()}};
  } else if (torch_value.isTuple()) {
    json::json payload = json::json::array();
    for (const auto &value : torch_value.toTuple()->elements()) {
      payload.emplace_back(TorchValueToJson(value));
    }
    return payload;
  } else if (torch_value.isScalar()) {
    return ScalarToJson(torch_value);
  } else if (torch_value.isGenericDict()) {
    return GenericDictToJson(torch_value);
  } else {
    throw TensorIOError("Only supports Tensor and TensorList types");
  }
}

void CheckValidJsonObject(const json::json &payload) {
  if (!(payload.contains("type") && payload.contains("value"))) {
    throw TensorIOError(
        "Error parsing payload, missing required "
        "attributes 'type' and 'value'.");
  }
  if (!payload.at("type").is_string()) {
    throw TensorIOError("Field `type` must be a string");
  }
  if (payload.contains("data_type")) {
    if (!payload.at("data_type").is_string()) {
      throw TensorIOError("If specified, data_type must be a string");
    }
  }
}

torch::Tensor ParseJsonTensor(const json::json &payload) {
  if (!(payload.contains("shape") && payload.at("shape").is_array())) {
    throw TensorIOError(
        "Error parsing payload, expected 'shape' to be an array "
        "for 'type' of 'tensor'");
  }
  json::json tensor = payload.at("value");
  if (!tensor.is_array()) {
    throw TensorIOError(
        "Error parsing payload, expected 'value' to be an array "
        "to be converted to 'type' of 'tensor'");
  }
  std::string data_type = payload.contains("data_type")
                              ? payload.at("data_type").get<std::string>()
                              : "float32";
  auto flattened_tensor = tensor.get<std::vector<float>>();
  auto tensor_shape = payload.at("shape").get<std::vector<long long>>();
  long long total_elements = 1;
  for (const auto &dim : tensor_shape) {
    total_elements = total_elements * dim;
  }
  if (flattened_tensor.size() != total_elements) {
    throw TensorShapeError(
        "Dimension mismatch - shape expected " +
        std::to_string(total_elements) + " total elements, found " +
        std::to_string(flattened_tensor.size()) + " total elements");
  }
  return std::move(
      torch::tensor(flattened_tensor,
                    torch::TensorOptions().dtype(StringToScalarType(data_type)))
          .reshape(tensor_shape));
}

std::vector<torch::Tensor> ParseJsonTensorList(const json::json &payload) {
  if (!payload.is_array()) {
    throw TensorIOError("Type tensor_list must be an array");
  }
  std::vector<torch::Tensor> tensor_vec;
  for (const auto &tensor_elem : payload) {
    tensor_vec.emplace_back(ParseJsonTensor(tensor_elem));
  }
  return std::move(tensor_vec);
}

torch::ivalue::UnorderedMap ParseJsonTensorDict(const json::json &payload) {
  if (!payload.is_object()) {
    throw TensorIOError("Type tensor_dict must be an object");
  }
  torch::ivalue::UnorderedMap tensor_dict;
  for (auto &tensor_elem : payload.items()) {
    tensor_dict[tensor_elem.key()] = ParseJsonTensor(tensor_elem.value());
  }
  return tensor_dict;
}

std::string ParseJsonString(const json::json &payload) {
  if (!payload.is_string()) {
    throw TensorIOError("Type string must specify a string value");
  }
  return payload.get<std::string>();
}

torch::Scalar ParseJsonScalar(const json::json &payload) {
  if ((!payload.contains("data_type")) or
      !payload.at("data_type").is_string()) {
    throw TensorIOError("Type scalar must specify a `data_type` as a string");
  }
  const std::string scalar_type = payload.at("data_type").get<std::string>();
  const auto json_scalar = payload.at("value");

  if (!json_scalar.is_number()) {
    throw TensorIOError("Type scalar must be a number");
  }
  return torch::scalar_to_tensor(json_scalar.get<float>())
      .to(StringToScalarType(scalar_type))
      .item();
}

std::vector<torch::jit::IValue> JsonToTorchValue(const json::json &payload) {
  std::vector<torch::jit::IValue> inputs;
  // First, we check the case where it's a single input, and convert
  if (payload.is_object()) {
    CheckValidJsonObject(payload);

    const auto type = payload.at("type").get<std::string>();

    if (type == "tensor") {
      inputs.emplace_back(ParseJsonTensor(payload));
    } else if (type == "tensor_list") {
      const torch::IValue tensor_list(ParseJsonTensorList(payload.at("value")));
      inputs.emplace_back(tensor_list);
    } else if (type == "tensor_dict") {
      const torch::IValue ival_tensor_dict(
          ParseJsonTensorDict(payload.at("value")));
      inputs.emplace_back(ival_tensor_dict);
    } else if (type == "scalar") {
      torch::Scalar scalar_value = ParseJsonScalar(payload);
      const torch::IValue ival_scalar(scalar_value);
      inputs.emplace_back(ival_scalar);
    } else if (type == "string") {
      auto string_value = ParseJsonString(payload.at("value"));
      const torch::IValue ival_string(string_value);
      inputs.emplace_back(ival_string);
    } else {
      throw TensorIOError("Unsupported type: " + type);
    }
  }
  // Next, we check the case that it's an array (one element per function
  // argument to the JIT-traced function)
  else {
    for (const auto &input : payload) {
      if (!input.is_object()) {
        throw TensorIOError("Must be an array of objects");
      }
      for (const auto &processed_input : JsonToTorchValue(input)) {
        inputs.emplace_back(processed_input);
      }
    }
  }
  return inputs;
}

}  // namespace torch_serving
