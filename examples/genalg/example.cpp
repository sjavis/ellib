#include "ellib.h"
#include <fstream>

using namespace ellib;
typedef std::vector<double> Vector;


void outputPop(std::vector<State>& pop) {
  std::ofstream file;
  if (mpi.rank==0) file = std::ofstream("output.txt", std::ios_base::app);
  for (auto s: pop) {
    auto coords = s.coords();
    if (mpi.rank == 0) {
      for (auto x: coords) {
        file << " " << x;
      }
    }
  }
  if (mpi.rank == 0) {
    file << std::endl;
    file.close();
  }
}


int main(int argc, char** argv) {
  mpiInit(&argc, &argv);

  // Clear output file
  if (mpi.rank==0) {
    std::ofstream file("output.txt");
    file.close();
  }

  auto eFunc = [](const Vector& x){ return x[0]*x[0] + x[1]*x[1]; };
  auto gFunc = [](const Vector& x){ return Vector{2*x[0], 2*x[1]}; };
  Potential pot(eFunc, gFunc);
  GenAlg ga = GenAlg(pot).setBounds({-1,-1}, {2,2}).setMaxIter(10).setIterFn(outputPop);
  auto result = ga.run();
  print(result);

  return 0;
}
