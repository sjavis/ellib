#include "ellib.h"

int main(int argc, char** argv) {
  ellib::mpiInit(&argc, &argv);

  ellib::Lj3d pot;
  ellib::GenAlg ga(pot);
  ga.setMinimiser("lbfgs").setBounds({-1,-1,-1,-1,-1,-1}, {1,1,1,1,1,1});
  auto result = ga.run();
  ellib::print(result);

  return 0;
}
