#include "GenAlg.h"

#include "gtest/gtest.h"
#include "gtest-mpi-listener.hpp"
#include "ArraysMatch.h"
#include "minim/utils/mpi.h"

#include "minim/Lj3d.h"
#include "minim/Lbfgs.h"
#include "minim/Fire.h"

using namespace ellib;
typedef std::vector<double> Vector;


TEST(GenAlgTest, InitClass) {
  Lj3d pot;
  GenAlg ga(pot);

  // Check default values
  EXPECT_EQ(ga.maxIter, 100);
  EXPECT_EQ(ga.popSize, 100);
  EXPECT_EQ(ga.numElites, 1);
  EXPECT_FLOAT_EQ(ga.selectionRate, 0.3);
  EXPECT_FLOAT_EQ(ga.mutationRate, 0.1);
  EXPECT_EQ(ga.stateGen, nullptr);
  EXPECT_EQ(ga.bounds.size(), 0);
  EXPECT_EQ(ga.min, nullptr);

  // Check set functions
  ga.setMaxIter(1000).setPopSize(10).setNumElites(0);
  EXPECT_EQ(ga.maxIter, 1000);
  EXPECT_EQ(ga.popSize, 10);
  EXPECT_EQ(ga.numElites, 0);

  ga.setSelectionRate(0.5).setMutationRate(0.5);
  EXPECT_FLOAT_EQ(ga.selectionRate, 0.5);
  EXPECT_FLOAT_EQ(ga.mutationRate, 0.5);

  auto stateGen = []() -> State { Lj3d pot; return State(pot, {0}); };
  ga.setStateGen(stateGen);
  EXPECT_TRUE(ArraysNear(ga.stateGen().coords(), {0}, 1e-6));

  ga.setBounds({0,0,0}, {1,1,1});
  EXPECT_EQ(ga.bounds.size(), 2);
  EXPECT_TRUE(ArraysNear(ga.bounds[1], {1,1,1}, 1e-6));

  ga.setMinimiser("lbfgs");
  EXPECT_NO_THROW(dynamic_cast<Lbfgs&>(*ga.min));
  ga.setMinimiser(std::unique_ptr<Minimiser>(new Fire));
  EXPECT_NO_THROW(dynamic_cast<Fire&>(*ga.min));
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  mpi.getSizeRank(MPI_COMM_WORLD);

  // Add an MPI listener (https://github.com/LLNL/gtest-mpi-listener)
  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  ::testing::TestEventListener *l = listeners.Release(listeners.default_result_printer());
  listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

  return RUN_ALL_TESTS();
}
