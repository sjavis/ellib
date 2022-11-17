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

  GenAlg& GenAlg::setBounds(Vector bound1, Vector bound2) {
    return *this;
  }

  GenAlg& GenAlg::setStateGen(StateFn stateGen) {
    return *this;
  }


  Vector GenAlg::run() {
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
