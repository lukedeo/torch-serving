//
// Created by Luke de Oliveira on 2019-08-08.
//

#ifndef TORCH_SERVING__MODELSERVER_H_
#define TORCH_SERVING__MODELSERVER_H_

#include "ServableManager.h"
#include "extern/httplib.h"
#include "extern/json.hpp"
#include <iostream>
#include <torch/script.h>

using nlohmann::json;

// Use shallowly overloaded errors from the STL just to make error defs easier
class MarshalingError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TensorShapeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class ModelServer {
public:
  explicit ModelServer(const size_t &model_capacity = 10,
                       const size_t &buffer = 0)
      : servable_manager_(model_capacity, buffer),
        logger_(spdlog::get("ModelServer")) {
    if (!logger_) {
      logger_ = spdlog::stdout_color_mt("ModelServer");
    }
    SetupEndpoints();
  }

  void RunServer(const std::string &host = "localhost",
                 const int &port = 8888) {
    server_.listen(host.c_str(), port);
  }

private:
  json MarshalOutputs(const torch::Tensor &tensor) {
    torch::IntArrayRef tensor_shape_ref = tensor.sizes();
    std::vector<int> tensor_shape(tensor_shape_ref.begin(),
                                  tensor_shape_ref.end());

    json payload = {{"type", "tensor"}};
    std::vector<float> tensor_vec(tensor.data<float>(),
                                  tensor.data<float>() + tensor.numel());
    payload["value"] = tensor_vec;
    payload["shape"] = tensor_shape;
    return payload;
  }

  json MarshalOutputs(const torch::jit::IValue &output) {
    if (output.isTensor()) {
      return MarshalOutputs(output.toTensor());
    } else if (output.isTensorList()) {
      auto tensor_list = output.toTensorListRef();
      json payload = json::array();
      for (const auto &tensor : tensor_list) {
        payload.emplace_back(MarshalOutputs(tensor));
      }
      return payload;
    } else {
      throw MarshalingError("Only supports Tensor and TensorList types");
    }
  }

  static std::vector<torch::jit::IValue> MarshalInputs(const json &payload) {
    std::vector<torch::jit::IValue> inputs;
    // First, we check the case where it's a single input, and convert
    if (payload.is_object()) {
      if (!(payload.contains("type") && payload.contains("value"))) {
        throw MarshalingError("Error parsing payload, missing required "
                              "attributes 'type' and 'value'.");
      } else {
        if (payload.at("type").get<std::string>() == "tensor") {
          if (!(payload.contains("shape") && payload.at("shape").is_array())) {
            throw MarshalingError(
                "Error parsing payload, expected 'shape' to be an array "
                "for 'type' of 'tensor'");
          }
          json tensor = payload.at("value");
          if (!tensor.is_array()) {
            throw MarshalingError(
                "Error parsing payload, expected 'value' to be an array "
                "to be converted to 'type' of 'tensor'");
          }
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
          inputs.emplace_back(
              torch::tensor(flattened_tensor).reshape(tensor_shape));
        } else {
          throw MarshalingError("Unsupported type: " +
                                payload.at("type").get<std::string>());
        }
      }
    } else {
      for (const auto &input : payload) {
        if (!input.is_object()) {
          throw MarshalingError("Must be an array of objects");
        }
        for (const auto &processed_input : MarshalInputs(input)) {
          inputs.emplace_back(processed_input);
        }
      }
    }
    return inputs;
  }

  static std::string GetHTTPMessageFromCode(const int &code) {
    switch ((code + 1) / 100) {
    case 1:
      return "Info";
    case 2:
      return "OK";
    case 3:
      return "Redirection";
    case 4:
      return "Client Error";
    case 5:
      return "Server Error";
    default:
      return "Undefined";
    }
  }
  static void SetResponse(httplib::Response &response, const int &code,
                          const std::string &description = "",
                          const json &payload = json::object()) {
    json outbound_payload = {{"code", code},
                             {"message", GetHTTPMessageFromCode(code)}};

    if (!description.empty()) {
      outbound_payload["description"] = description;
    }

    if (!payload.empty()) {
      outbound_payload["result"] = payload;
    }

    response.status = code;
    response.set_content(outbound_payload.dump(), "application/json");
  }

  void SetupEndpoints() {
    // Recieves POST /v1/serve/{modelIdentifier} requests
    server_.Post(R"(/v1/serve/(.*))", [&](const httplib::Request &req,
                                          httplib::Response &res) {
      std::string servable_identifier = req.matches[1];
      // First, just make sure we sent *something* over the wire.
      if (req.body.empty()) {
        SetResponse(res, 400, "Empty body");
        return;
      } else {
        // Step 1: Parse the JSON
        json payload;
        try {
          payload = json::parse(req.body);
        } catch (const std::exception &err) {
          SetResponse(res, 400, "Invalid JSON");
          return;
        }

        // Step 2: Parse the inputs to the JIT-compiled model
        std::vector<torch::jit::IValue> inputs;
        try {
          inputs = MarshalInputs(payload);
        } catch (const MarshalingError &err) {
          SetResponse(res, 400, "Invalid Input JSON");
          return;
        } catch (const TensorShapeError &err) {
          SetResponse(res, 400, "Incompatible tensor shapes");
          return;
        } catch (const std::exception &err) {
          SetResponse(res, 500, "Unexpected server error");
          return;
        }

        // Step 3: Run the inputs through the model (identified by the
        // servable_identifier) in a future.
        std::future<torch::jit::IValue> async_inference_response;
        try {
          async_inference_response = servable_manager_.AsyncInferenceRequest(
              servable_identifier, inputs, 0.0, std::launch::async);
        } catch (const std::exception &err) {
          SetResponse(res, 500, "Unexpected server error");
          return;
        }

        // Step 4: Wait for the future to be done, and set the result
        try {
          async_inference_response.wait();
          auto result = MarshalOutputs(async_inference_response.get());
          SetResponse(res, 200, "Success", result);
        } catch (const std::invalid_argument &err) {
          SetResponse(res, 400, "Invalid servable identifier");
          return;
        } catch (const std::exception &err) {
          SetResponse(res, 500, "Unexpected server error");
          return;
        }
      }
    });

    server_.set_logger([this](const httplib::Request &req,
                              const httplib::Response &res) {
      logger_->info("Request: [" + req.method + " " + req.version + " " +
                    req.path + "] => Response: [" + std::to_string(res.status) +
                    " " + GetHTTPMessageFromCode(res.status) + "]");
    });
  }

  httplib::Server server_;
  ServableManager servable_manager_;
  std::shared_ptr<spdlog::logger> logger_;
};

#endif // TORCH_SERVING__MODELSERVER_H_
