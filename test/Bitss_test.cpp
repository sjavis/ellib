#include "Bitss.h"

#include "gtest/gtest.h"
#include "gtest-mpi-listener.hpp"
#include "ArraysMatch.h"

#include "minim/Lj3d.h"
#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"
#include "minim/utils/mpi.h"
#include <math.h>

using namespace ellib;

typedef std::vector<double> Vector;


TEST(BitssTest, DefaultInitialisation) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
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

  // Check single state potentials are correct
  ASSERT_NO_THROW(dynamic_cast<Bitss::BitssPotential&>(*bitss.state.pot));
  auto bitssPot = static_cast<Bitss::BitssPotential&>(*bitss.state.pot);
  EXPECT_NO_THROW(dynamic_cast<Lj3d&>(*bitssPot.state1.pot));
  EXPECT_NO_THROW(dynamic_cast<Lj3d&>(*bitssPot.state2.pot));
}


TEST(BitssTest, MinimiserConstructor1) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  // Explicitly pass minimiser object
  auto min = std::unique_ptr<Fire>(new Fire);
  Bitss bitss(s1, s2, std::move(min));
  EXPECT_NO_THROW(dynamic_cast<Fire&>(*bitss.minimiser));
  EXPECT_THROW(dynamic_cast<Lbfgs&>(*bitss.minimiser), std::bad_cast);
}


TEST(BitssTest, MinimiserConstructor2) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  // Lbfgs
  Bitss bitss1(s1, s2, "Lbfgs");
  EXPECT_NO_THROW(dynamic_cast<Lbfgs&>(*bitss1.minimiser));
  EXPECT_THROW(dynamic_cast<Fire&>(*bitss1.minimiser), std::bad_cast);
  // Fire
  Bitss bitss2(s1, s2, "fire");
  EXPECT_NO_THROW(dynamic_cast<Fire&>(*bitss2.minimiser));
  // GradDescent
  Bitss bitss3(s1, s2, "GRADDESCENT");
  EXPECT_NO_THROW(dynamic_cast<GradDescent&>(*bitss3.minimiser));
  // Anneal
  Bitss bitss4(s1, s2, "Anneal");
  EXPECT_NO_THROW(dynamic_cast<Anneal&>(*bitss4.minimiser));
  // Undefined
  EXPECT_THROW(Bitss(s1, s2, "FooBar"), std::invalid_argument);
}


TEST(BitssTest, MaxIter) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  EXPECT_EQ(bitss.maxIter, 10);
  bitss.setMaxIter(5);
  EXPECT_EQ(bitss.maxIter, 5);
  // Check final iter
  bitss.run();
  EXPECT_EQ(bitss.iter(), 5);
}


TEST(BitssTest, DistStep) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  EXPECT_EQ(bitss.distStep, 0.5);
  bitss.setDistStep(0.1);
  EXPECT_EQ(bitss.distStep, 0.1);
  // Test one iteration
  bitss.setMaxIter(1);
  bitss.run();
  auto bitssPot = static_cast<Bitss::BitssPotential&>(*bitss.state.pot);
  EXPECT_EQ(bitssPot.di, 0.9);
}


TEST(BitssTest, ConvergenceDist) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  EXPECT_EQ(bitss.convergenceDist, 0.01);
  bitss.setConvergenceDist(0.5);
  EXPECT_EQ(bitss.convergenceDist, 0.5);
  // Check convergence
  bitss.setDistStep(0.5);
  bitss.run();
  EXPECT_EQ(bitss.iter(), 0);
}


TEST(BitssTest, EScaleMax) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  EXPECT_EQ(bitss.eScaleMax, 0);
  bitss.setEScaleMax(0.1);
  EXPECT_EQ(bitss.eScaleMax, 0.1);
}


TEST(BitssTest, CoefIter) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  auto bitssPot = static_cast<Bitss::BitssPotential*>(bitss.state.pot.get());
  EXPECT_EQ(bitssPot->coefIter, 100);
  bitss.setCoefIter(10);
  EXPECT_EQ(bitssPot->coefIter, 10);
}


TEST(BitssTest, Alpha) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  auto bitssPot = static_cast<Bitss::BitssPotential*>(bitss.state.pot.get());
  EXPECT_EQ(bitssPot->alpha, 10);
  bitss.setAlpha(2);
  EXPECT_EQ(bitssPot->alpha, 2);
}


TEST(BitssTest, Beta) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value and setter
  auto bitssPot = static_cast<Bitss::BitssPotential*>(bitss.state.pot.get());
  EXPECT_EQ(bitssPot->beta, 0.1);
  bitss.setBeta(0.01);
  EXPECT_EQ(bitssPot->beta, 0.01);
}


TEST(BitssTest, DistFunc) {
  Lj3d pot;
  State s1 = pot.newState({0,0,0});
  State s2 = pot.newState({1,0,0});
  Bitss bitss(s1, s2);
  // Check default value
  auto bitssPot = static_cast<Bitss::BitssPotential*>(bitss.state.pot.get());
  typedef std::vector<double> Vector;
  EXPECT_EQ(bitssPot->dist({3,0}, {0,4}), 5);
  EXPECT_TRUE(ArraysNear(bitssPot->distGrad({3,0}, {0,4}),  {0.6, -0.8}, 1e-6));
  // Check setter
  auto newDist = [](const Vector& x1, const Vector& x2) -> double { return x1[0]-x2[0]; };
  auto newGrad = [](const Vector& x1, const Vector& x2) -> Vector { return {1, 0}; };
  bitss.setDistFunc(newDist, newGrad);
  EXPECT_EQ(bitssPot->dist({3,0}, {0,4}), 3);
  EXPECT_TRUE(ArraysNear(bitssPot->distGrad({3,0}, {0,4}), {1, 0}, 1e-6));
}


TEST(BitssTest, BitssPotential) {
  Lj3d pot;
  Vector x1 = {0,0,0, 1,0,0};
  Vector x2 = {10,10,10, 12,10,10};
  State s1 = pot.newState(x1);
  State s2 = pot.newState(x2);
  Bitss bitss(s1, s2);
  auto bitssPot = static_cast<Bitss::BitssPotential*>(bitss.state.pot.get());
  bitssPot->ke = 2;
  bitssPot->kd = 2;
  bitssPot->di = 20;
  // Test single state energies
  double e1 = 0;
  double e2 = 1./1024 - 1./16;
  EXPECT_FLOAT_EQ(s1.energy(), e1);
  EXPECT_FLOAT_EQ(s2.energy(), e2);
  Vector grad1 = {24,0,0, -24,0,0};
  Vector grad2 = {3./512-3./16,0,0, 3./16-3./512,0,0};
  EXPECT_TRUE(ArraysNear(s1.gradient(), grad1, 1e-6));
  EXPECT_TRUE(ArraysNear(s2.gradient(), grad2, 1e-6));
  // Total energy
  double d = sqrt(621.);
  double ee = 2 * pow(e1-e2, 2);
  double ed = 2 * pow(d-20, 2);
  EXPECT_FLOAT_EQ(bitss.state.energy(), e1+e2+ee+ed);
  Vector ge1 = (1 + 4*(e1-e2)) * grad1;
  Vector ge2 = (1 + 4*(e2-e1)) * grad2;
  Vector gd = 4 * (1-20/d) * (x1-x2);
  Vector gtot2 = ge2 - gd;
  Vector gtot = ge1 + gd;
  gtot.insert(gtot.end(), gtot2.begin(), gtot2.end());
  EXPECT_TRUE(ArraysNear(bitss.state.gradient(), gtot, 1e-6));
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
