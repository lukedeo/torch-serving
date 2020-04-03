//
// (c) 2020, Luke de Oliveira
// This code is licensed under MIT license (see LICENSE for details)
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "torch_serving/model_server.h"
#include "torch_serving/tensor_io.h"
#include "torch_serving/torch_jit_servable.h"

std::string GetEnvVar(const std::string &variable_name,
                      const std::string &default_value) {
  const char *value = std::getenv(variable_name.c_str());
  return value ? value : default_value;
}

TEST_CASE("Test Torch Tensor to JSON") {
  auto t = torch::tensor({1, 2, 3, 4});
  auto result = torch_serving::TorchValueToJson(t);
  MESSAGE("Shape is correct representation");
  CHECK(result.contains("shape") && result.at("shape").is_array());

  auto shape = result.at("shape").get<std::vector<long long>>();
  MESSAGE("Shape is correct value");
  CHECK_EQ(shape[0], 4);
  CHECK_EQ(shape.size(), 1);

  MESSAGE("Result has value of right type");
  CHECK(result.contains("value") && result.at("value").is_array());

  auto value = result.at("value").get<std::vector<int>>();
  MESSAGE("Check consistent value");
  CHECK_EQ(value[0], 1);
  CHECK_EQ(value[1], 2);
  CHECK_EQ(value[2], 3);
  CHECK_EQ(value[3], 4);
  CHECK_EQ(value.size(), 4);

  MESSAGE("Check valid type");
  CHECK(result.contains("type") && result.at("type").is_string());
  CHECK_EQ(result.at("type"), "tensor");

  MESSAGE("Check valid data type");
  CHECK(result.contains("data_type") && result.at("data_type").is_string());
  CHECK_EQ(result.at("data_type"), "int64");
}

TEST_CASE("Test JSON Torch Tensor") {
  json::json jt = {{"shape", {2, 3}},
                   {"value", {1.2, 3.4, 4.4, 1.2, 6.2, 7.2}},
                   {"data_type", "float32"},
                   {"type", "tensor"}};

  auto vt = torch_serving::JsonToTorchValue(jt);
  MESSAGE("Check return singleton vector of Tensors");
  CHECK_EQ(vt.size(), 1);

  torch::Tensor t = vt.at(0).toTensor();
  MESSAGE("Check correct shape");
  torch::IntArrayRef sh = {2, 3};
  CHECK_EQ(t.sizes(), sh);

  MESSAGE("Check correct values");
  CHECK(torch::allclose(t, torch::tensor({{1.2, 3.4, 4.4}, {1.2, 6.2, 7.2}})));
}

TEST_CASE("Test servable manager inference") {
  torch_serving::ServableManager<torch_serving::TorchJITServable> manager;

  std::string servable_payload = GetEnvVar(
      "TS_TEST_PAYLOAD", "../tests/assets/test-servable-payload.json");
  std::string servable_response = GetEnvVar(
      "TS_TEST_RESPONSE", "../tests/assets/test-servable-response.json");
  std::string servable_model =
      GetEnvVar("TS_TEST_MODEL", "../tests/assets/test-servable.pt");

  json::json payload = json::json::parse(std::ifstream(servable_payload));
  json::json response = json::json::parse(std::ifstream(servable_response));

  MESSAGE("Verify sync result");
  auto result = manager.InferenceRequest(servable_model, payload);
  CHECK_EQ(response, result);

  MESSAGE("Verify async result");
  auto result_async = manager.AsyncInferenceRequest(servable_model, payload);
  CHECK_EQ(response, result_async.get());
}
