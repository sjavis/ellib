#include "ellib.h"
#include <fstream>
#include <iterator>

typedef std::vector<double> Vector;
using namespace ellib;

// Beam parameters
double kBar = 1e-3;
double kHinge = 1e-5;
double length = 6;
double totalStrain = 0.1;

// Initial ramp
int iterMax = 1000;
double initForce = 0.00001;
double iterStrain = std::pow(1-totalStrain, 1.0/iterMax);


void outputData(Vector coords, std::string filename) {
  std::ofstream file(filename);
  for (size_t i=0; i<coords.size()/3; i++) {
    file << coords[3*i] << " " << coords[3*i+1] << " " << coords[3*i+2] << std::endl;
  }
  file.close();
}


Vector loadCoords() {
  std::ifstream file("coords.txt");
  std::istream_iterator<double> start(file), end;
  Vector coords(start, end);
  file.close();
  return coords;
}


std::vector<std::vector<int>> loadTri() {
  std::vector<std::vector<int>> nodes;
  std::ifstream file("tlist.txt");
  while (!file.eof()) {
    int n1, n2, n3;
    file >> n1 >> n2 >> n3;
    nodes.push_back({n1, n2, n3});
    if (file.eof()) break;
  }
  file.close();
  return nodes;
}


std::vector<bool> loadFixedNodes() {
  std::vector<bool> fixed;
  std::ifstream file("regions.txt");
  while (!file.eof()) {
    double region;
    file >> region;
    if (region < 0) {
      fixed.push_back(true);
      fixed.push_back(true);
      fixed.push_back(true);
    } else {
      fixed.push_back(false);
      fixed.push_back(false);
      fixed.push_back(false);
    }
    if (file.eof()) break;
  }
  return fixed;
}


void applyStrain(int iter, State& state) {
  if (iter < iterMax) {
    Vector coords = state.coords();
    for (size_t i=0; i<state.ndof/3; i++) {
      coords[3*i] = coords[3*i] * iterStrain;
      coords[3*i+1] = coords[3*i+1] * iterStrain;
    }
    state.coords(coords);
    state.convergence = 0;

  } else if (iter == iterMax) {
    dynamic_cast<BarAndHinge&>(*state.pot).setForce({0,0,0});
    state.convergence = 1e-6;
  }
}


int main(int argc, char** argv) {
  // mpiInit(&argc, &argv);

  // Initialise the flat structure
  print("Initialising...");
  BarAndHinge pot;
  pot.setRigidity(kBar, kHinge);
  pot.setTriangulation(loadTri());
  pot.setFixed(loadFixedNodes());
  State initState(pot, loadCoords());

  // Get the minima under an applied strain
  State upState = initState;
  State downState = initState;
  dynamic_cast<BarAndHinge&>(*upState.pot).setForce({0,0,initForce});
  dynamic_cast<BarAndHinge&>(*downState.pot).setForce({0,0,-initForce});
  Lbfgs min;
  min.setMaxIter(5000);

  print("Finding minimum 1...");
  min.minimise(upState, applyStrain);
  outputData(upState.coords(), "buckled_up.txt");

  print("Finding minimum 2...");
  min.minimise(downState, applyStrain);
  outputData(downState.coords(), "buckled_down.txt");

  // Find the transition state with BITSS
  print("Finding transition state...");
  Bitss bitss(upState, downState);
  State tsPair = bitss.run();
  outputData(tsPair.coords(), "transition_state.txt");

  return 0;
}
