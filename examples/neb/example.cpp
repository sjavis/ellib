#include "ellib.h"
#include <fstream>

using namespace ellib;
using std::vector;


vector<vector<double>> peaks = {
  {-3, -1.4, 0, 1, 1},
  {-2,  1.4, 0, 1, 1},
  {-1, 0.07, 1, 1, 1}
};
void pathwayEG(const vector<double>& coords, double* e, vector<double>* g) {
  if (e) *e = 0;
  if (g) *g = vector<double>(2);
  for (auto peak: peaks) {
    double dx = (coords[0] - peak[1]) / peak[3];
    double dy = (coords[1] - peak[2]) / peak[4];
    double epeak = peak[0] * exp(- dx*dx - dy*dy);
    if (e) *e += epeak;
    if (g) *g -= vector<double>{2*dx/peak[3], 2*dy/peak[4]} * epeak;
   }
}


int main(int argc, char** argv) {
  mpiInit(&argc, &argv);
  print();

  // Get minima
  Lbfgs min;
  auto state1 = State(Potential(pathwayEG), {-1,0});
  auto state2 = State(Potential(pathwayEG), {1,0});
  auto min1 = min.minimise(state1);
  auto min2 = min.minimise(state2);

  Neb neb(Potential(pathwayEG), min1, min2, 5);
  neb.setHybrid(1, 100);
  auto chain = neb.run();

  std::ofstream file("path.txt");
  for (auto state: chain) {
    for (double x: state) {
      file << " " << x;
    }
    file << std::endl;
  }

  return 0;
}
