#include "Bitss.h"
#include "minim/Lj3d.h"
#include "minim/State.h"
#include "minim/Lbfgs.h"

#include "gtest/gtest.h"

TEST(BitssTest, Test1) {
  minim::Lj3d pot;
  minim::State s1 = pot.newState({0,0});
  minim::State s2 = pot.newState({0,0});
  minim::Lbfgs minimiser;
  ellib::Bitss bitss(s1, s2, minimiser);
  EXPECT_EQ(1, 1);
}
