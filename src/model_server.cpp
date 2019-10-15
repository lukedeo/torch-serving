//
// (c) 2019, Luke de Oliveira
// This code is licensed under MIT license (see LICENSE for details)
//

#include "torch_serving/model_server.h"

#include <spdlog/sinks/stdout_color_sinks-inl.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <memory>

#include "torch_serving/tensor_io.h"

namespace json = nlohmann;

namespace torch_serving {

ModelServer::ModelServer(const size_t &model_capacity, const size_t &buffer,
                         const size_t &thread_pool_size)
    : servable_manager_(model_capacity, buffer),
      logger_(spdlog::get("model_server")),
      thread_pool_(std::make_shared<httplib::ThreadPool>(thread_pool_size)) {
  if (!logger_) {
    logger_ = spdlog::stdout_color_mt("model_server");
  }
  logger_->info("Allocated thread pool of size " +
                std::to_string(thread_pool_size));
  SetupEndpoints();
}

void ModelServer::RunServer(const std::string &host, const int &port) {
  server_.new_task_queue = [this] { return thread_pool_.get(); };
  logger_->info("Listening on " + host + ":" + std::to_string(port));
  server_.listen(host.c_str(), port);
}

std::string ModelServer::GetHTTPMessageFromCode(const int &code) {
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
void ModelServer::SetResponse(httplib::Response &response, const int &code,
                              const std::string &description,
                              const json::json &payload,
                              const std::string &detail) {
  json::json outbound_payload = {{"code", code},
                                 {"message", GetHTTPMessageFromCode(code)}};

  if (!description.empty()) {
    outbound_payload["description"] = description;
  }

  if (!payload.empty()) {
    outbound_payload["result"] = payload;
  }

  if (!detail.empty()) {
    outbound_payload["detail"] = detail;
  }

  response.status = code;
  response.set_content(outbound_payload.dump(), "application/json");
}

void ModelServer::SetupEndpoints() {
  // Receives GET /healthcheck requests
  server_.Get("/healthcheck",
              [&](const httplib::Request &req, httplib::Response &res) {
                SetResponse(res, 200, "OK");
              });
  // Receives POST /serve requests
  server_.Post("/serve", [&](const httplib::Request &req,
                             httplib::Response &res) {
    auto p_count = req.params.count("servable_identifier");
    if (!p_count) {
      SetResponse(res, 400, "Missing required parameter `servable_identifier`");
      return;
    } else if (p_count > 1) {
      SetResponse(res, 400,
                  "Required parameter `servable_identifier` must only be "
                  "passed in once");
      return;
    }
    auto servable_identifier = req.params.find("servable_identifier")->second;

    // First, just make sure we sent *something* over the wire.
    if (req.body.empty()) {
      SetResponse(res, 400, "Empty body");
      return;
    } else {
      // Step 1: Parse the JSON
      json::json payload;
      try {
        payload = json::json::parse(req.body);
      } catch (const std::exception &err) {
        SetResponse(res, 400, "Invalid JSON", err.what());
        return;
      }

      // Step 2: Parse the inputs to the JIT-compiled model
      std::vector<torch::jit::IValue> inputs;
      try {
        inputs = JsonToTorchValue(payload);
      } catch (const TensorIOError &err) {
        SetResponse(res, 400, "Invalid Input JSON", err.what());
        return;
      } catch (const TensorShapeError &err) {
        SetResponse(res, 400, "Incompatible tensor shapes", err.what());
        return;
      } catch (const TensorTypeError &err) {
        SetResponse(res, 400, "Incompatible tensor data type", err.what());
        return;
      } catch (const std::exception &err) {
        SetResponse(res, 500, "Unexpected server error", err.what());
        return;
      }

      // Step 3: Run the inputs through the model (identified by the
      // servable_identifier) in a future.
      std::future<torch::jit::IValue> async_inference_response;
      try {
        async_inference_response = servable_manager_.AsyncInferenceRequest(
            servable_identifier, inputs, 0.0, std::launch::async);
      } catch (const std::exception &err) {
        SetResponse(res, 500, "Unexpected server error", err.what());
        return;
      }

      // Step 4: Wait for the future to be done, and set the result
      try {
        async_inference_response.wait();
        auto result = TorchValueToJson(async_inference_response.get());
        SetResponse(res, 200, "Success", result);
      } catch (const std::invalid_argument &err) {
        SetResponse(res, 400, "Invalid servable identifier");
        return;
      } catch (const std::exception &err) {
        logger_->error(err.what());
        SetResponse(res, 500, "Unexpected server error", err.what());
        return;
      }
    }
  });

  server_.set_logger(
      [this](const httplib::Request &req, const httplib::Response &res) {
        auto msg = "Request: [" + req.method + " " + req.version + " " +
                   req.path + "] => Response: [" + std::to_string(res.status) +
                   " " + GetHTTPMessageFromCode(res.status) + "]";
        switch ((res.status + 1) / 100) {
          case 3:
            logger_->warn(msg);
            break;
          case 4:
            logger_->warn(msg);
            break;
          case 5:
            logger_->error(msg);
            break;
          default:
            logger_->info(msg);
        }
      });
}

}  // namespace torch_serving
