#include "Bitss.h"
#include "minim/Lj3d.h"
#include "minim/State.h"
#include "minim/Lbfgs.h"
#include "minim/utils/mpi.h"

#include "gtest/gtest.h"


TEST(BitssTest, Test1) {
  ellib::Lj3d pot;
  ellib::State s1 = pot.newState({0,0});
  ellib::State s2 = pot.newState({0,0});
  ellib::Bitss bitss(s1, s2);
  EXPECT_EQ(1, 1);
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ellib::mpiInit(&argc, &argv);

  // Ensure only one processor prints
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (ellib::mpi.rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
  }

  return RUN_ALL_TESTS();
}
