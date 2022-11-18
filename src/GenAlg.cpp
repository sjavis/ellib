#include "GenAlg.h"

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"

namespace ellib {

  typedef std::vector<double> Vector;


  GenAlg::GenAlg(Potential& pot)
    : pot(pot.clone())
  {}


  GenAlg& GenAlg::setMaxIter(int maxIter) {
    this->maxIter = maxIter;
    return *this;
  }

  GenAlg& GenAlg::setPopSize(int popSize) {
    this->popSize = popSize;
    return *this;
  }

  GenAlg& GenAlg::setNumElites(int numElites) {
    this->numElites = numElites;
    return *this;
  }

  GenAlg& GenAlg::setSelectionRate(double selectionRate) {
    this->selectionRate = selectionRate;
    return *this;
  }

  GenAlg& GenAlg::setMutationRate(double mutationRate) {
    this->mutationRate = mutationRate;
    return *this;
  }

  GenAlg& GenAlg::setStateGen(StateFn stateGen) {
    this->stateGen = stateGen;
    return *this;
  }

  GenAlg& GenAlg::setBounds(Vector bound1, Vector bound2) {
    bounds = std::vector<Vector>(bound1.size());
    for (size_t i=0; i<bound1.size(); i++) {
      if (bound1[i]<=bound2[i]) {
        bounds[i] = {bound1[i], bound2[i]};
      } else {
        bounds[i] = {bound2[i], bound1[i]};
      }
    }
    return *this;
  }

  GenAlg& GenAlg::setMinimiser(const std::string& min) {
    std::string string = min;
    std::transform(string.begin(), string.end(), string.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (string == "lbfgs") {
      this->min= std::unique_ptr<Minimiser>(new Lbfgs);
    } else if (string == "fire") {
      this->min= std::unique_ptr<Minimiser>(new Fire);
    } else if (string == "graddescent") {
      this->min= std::unique_ptr<Minimiser>(new GradDescent);
    } else if (string == "anneal") {
      this->min= std::unique_ptr<Minimiser>(new Anneal(1, 0.0001));
    } else {
      throw std::invalid_argument("Invalid minimiser chosen");
    }
    return *this;
  }

  GenAlg& GenAlg::setMinimiser(std::unique_ptr<Minimiser> min) {
    this->min = std::move(min);
    return *this;
  }


  Vector GenAlg::run() {
    initialise();
    for (int iter=0; iter<maxIter; iter++) {
      if (iter > 0) {
        auto parents = select();
        crossover(parents);
        mutate();
      }
      minimise();
      if (checkComplete()) break;
    }

    auto bestState = getBestStates(1, getEnergies())[0];
    return bestState.coords();
  }


  void GenAlg::initialise() {
  }


  std::vector<State> GenAlg::select() {
    return std::vector<State>();
  }


  void GenAlg::crossover(const std::vector<State>& parents) {
  }


  void GenAlg::mutate() {
  }


  void GenAlg::minimise() {
    if (min != nullptr) {
      for (auto state: pop) {
        min->minimise(state);
      }
    }
  }


  bool GenAlg::checkComplete() {
    return false;
  }


  Vector GenAlg::getEnergies() {
    Vector energies(popSize);
    for (int i=0; i<popSize; i++) {
      energies[i] = pop[i].energy();
    }
    return energies;
  }


  std::vector<State> GenAlg::getBestStates(int n, Vector energies) {
    // Get the indicies of the lowest energy states
    std::vector<int> index(popSize);
    std::iota(index.begin(), index.end(), 0);
    std::partial_sort(index.begin(), index.begin()+n, index.end(), [energies](int i, int j){ return energies[i]<energies[j]; });
    // Get the vector of the best states
    std::vector<State> best(n, pop[0]); // Assign pop[0] because State has no default initialiser
    for (int i=0; i<n; i++) {
      best[i] = pop[index[i]];
    }
    return best;
  }

}
