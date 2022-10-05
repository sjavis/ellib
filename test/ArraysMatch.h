#include "gtest/gtest.h"
#include <vector>


template<typename T>
::testing::AssertionResult ArraysMatch(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) {
    return ::testing::AssertionFailure() << "array 1 has size " << a.size() << " != expected size " << b.size();
  }

  for (size_t i(0); i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return ::testing::AssertionFailure() << "a["<<i<< "] ("<<a[i]<<") != b["<< i<<"] ("<<b[i]<<")";
    }
  }

  return ::testing::AssertionSuccess();
}


template<typename T>
::testing::AssertionResult ArraysNear(const std::vector<T>& a, const std::vector<T>& b, float delta) {
  if (a.size() != b.size()) {
    return ::testing::AssertionFailure() << "array 1 has size " << a.size() << " != expected size " << b.size();
  }

  for (size_t i(0); i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > delta) {
      return ::testing::AssertionFailure() << "a["<<i<< "] ("<<a[i]<<") != b["<< i<<"] ("<<b[i]<<")";
    }
  }

  return ::testing::AssertionSuccess();
}
