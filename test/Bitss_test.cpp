#include "Bitss.h"

#include <vector>
#include "minim/Lj3d.h"
#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/utils/mpi.h"

#include "gtest/gtest.h"

using namespace ellib;


TEST(BitssTest, DefaultInitialisation) {
  Lj3d pot;
  State s1 = pot.newState({0,0});
  State s2 = pot.newState({1,0});
  Bitss bitss(s1, s2);
  // Check minimiser is Lbfgs
  EXPECT_NO_THROW(dynamic_cast<Lbfgs&>(*bitss.minimiser));
  EXPECT_THROW(dynamic_cast<Fire&>(*bitss.minimiser), std::bad_cast);
  // State coordinates
  auto coords = bitss.state.getCoords();
  auto coords1 = std::vector<double>(coords.begin(), coords.begin()+s1.ndof);
  auto coords2 = std::vector<double>(coords.begin()+s1.ndof, coords.end());
  EXPECT_EQ(bitss.state.ndof, s1.ndof+s2.ndof);
  EXPECT_EQ(coords1, s1.getCoords());
  EXPECT_EQ(coords2, s2.getCoords());
  // Check single state potentials are correct (will be possible if Args is removed and state.pot made public)
  // ASSERT_NO_THROW(dynamic_cast<Bitss::BitssPotential&>(*bitss.state._pot));
  // auto bitssPot = static_cast<Bitss::BitssPotential&>(*bitss.state._pot);
  // EXPECT_NO_THROW(dynamic_cast<Lj3d&>(*bitssPot.state1._pot));
  // EXPECT_NO_THROW(dynamic_cast<Lj3d&>(*bitssPot.state2._pot));
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  mpiInit(&argc, &argv);

  // Ensure only one processor prints
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (mpi.rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
  }

  return RUN_ALL_TESTS();
}
