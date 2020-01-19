#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "torch_serving/tensor_io.h"

int factorial(int number) { return number <= 1 ? number : factorial(number - 1) * number; }

TEST_CASE("testing the factorial function") {
CHECK(factorial(1) == 1);
CHECK(factorial(2) == 2);
CHECK(factorial(3) == 6);
CHECK(factorial(10) == 3628800);
}
TEST_CASE("testing JSON to torch") {
  auto t = torch::tensor({1, 2, 3, 4});
  auto result = torch_serving::TorchValueToJson(t);
  CHECK(result.contains("shape") && result.at("shape").is_array());
  auto shape = result.at("shape").get<std::vector<long long>>();
  CHECK_EQ(shape[0], 4);
  CHECK_EQ(shape.size(), 1);
}