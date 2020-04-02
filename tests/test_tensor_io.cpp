#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "torch_serving/tensor_io.h"
#include "torch_serving/model_server.h"
#include "torch_serving/torch_jit_servable.h"


TEST_CASE("Test Torch Tensor to JSON") {
  auto t = torch::tensor({1, 2, 3, 4});
  auto result = torch_serving::TorchValueToJson(t);
  CHECK(result.contains("shape") && result.at("shape").is_array());
  auto shape = result.at("shape").get<std::vector<long long>>();
  CHECK_EQ(shape[0], 4);
  CHECK_EQ(shape.size(), 1);

  CHECK(result.contains("value") && result.at("value").is_array());
  auto value = result.at("value").get<std::vector<int>>();
  CHECK_EQ(value[0], 1);
  CHECK_EQ(value[1], 2);
  CHECK_EQ(value[2], 3);
  CHECK_EQ(value[3], 4);
  CHECK_EQ(value.size(), 4);

  CHECK(result.contains("type") && result.at("type").is_string());
  CHECK_EQ(result.at("type"), "tensor");

  CHECK(result.contains("data_type") && result.at("data_type").is_string());
  CHECK_EQ(result.at("data_type"), "int64");
}

TEST_CASE("Test JSON Torch Tensor") {
  json::json jt = {
      {"shape", {2, 3}},
      {"value", {1.2, 3.4, 4.4, 1.2, 6.2, 7.2}},
      {"data_type", "float32"},
      {"type", "tensor"}
  };

  auto vt = torch_serving::JsonToTorchValue(jt);
  CHECK_EQ(vt.size(), 1);
  torch::Tensor t = vt.at(0).toTensor();

  torch::IntArrayRef sh = {2, 3};
  CHECK_EQ(t.sizes(), sh);

  CHECK(torch::allclose(t,
      torch::tensor({{1.2, 3.4, 4.4}, {1.2, 6.2, 7.2}})));
}

TEST_CASE("Test servable manager inference") {
  torch_serving::ServableManager<torch_serving::TorchJITServable> manager;


  json::json payload = torch_serving::TorchValueToJson(torch::tensor({{1, 2, 3, 4}}));

  auto result = manager.InferenceRequest("tests/test-servable.pt", payload);




}


//
//TEST_CASE("testing torsch to JSON") {
//  json::json j = {};
//  auto t = torch::tensor({1, 2, 3, 4});
//  auto result = torch_serving::TorchValueToJson(t);
//  CHECK(result.contains("shape") && result.at("shape").is_array());
//  auto shape = result.at("shape").get<std::vector<long long>>();
//  CHECK_EQ(shape[0], 4);
//  CHECK_EQ(shape.size(), 1);
//}