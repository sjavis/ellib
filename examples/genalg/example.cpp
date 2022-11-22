#include "ellib.h"

using namespace ellib;
typedef std::vector<double> Vector;

int main(int argc, char** argv) {
  mpiInit(&argc, &argv);

  auto eFunc = [](const Vector& x){ return x[0]*x[0] + x[1]*x[1]; };
  auto gFunc = [](const Vector& x){ return Vector{2*x[0], 2*x[1]}; };
  Potential pot(eFunc, gFunc);
  GenAlg ga(pot);
  ga.setBounds({-1,-1}, {2,2});
  auto result = ga.run();
  print(result);

  return 0;
}
