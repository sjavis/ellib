#include "GenAlg.h"

#include <algorithm>
#include <stdexcept>
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
    return Vector();
  }


  void GenAlg::initialise() {
  }


  void GenAlg::select() {
  }


  void GenAlg::crossover() {
  }


  void GenAlg::mutate() {
  }


  void GenAlg::checkComplete() {
  }

}
