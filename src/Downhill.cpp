#include "Downhill.h"

#include <stdexcept>

#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"

#include "minim/utils/vec.h"

namespace ellib {


  Downhill::Downhill(const vector<State>& tsPair, double stepSize, std::string minimiser)
    : stepSize(stepSize), tsPair(tsPair)
  {
    std::string string = minimiser;
    std::transform(string.begin(), string.end(), string.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (string == "lbfgs") {
      this->minimiser = std::unique_ptr<Minimiser>(new Lbfgs);
    } else if (string == "fire") {
      this->minimiser = std::unique_ptr<Minimiser>(new Fire);
    } else if (string == "graddescent") {
      this->minimiser = std::unique_ptr<Minimiser>(new GradDescent);
    } else if (string == "anneal") {
      this->minimiser = std::unique_ptr<Minimiser>(new Anneal(1, 0.0001));
    } else {
      throw std::invalid_argument("Invalid minimiser chosen");
    }
  }

  Downhill::Downhill(const vector<State>& tsPair, double stepSize, const Minimiser& minimiser)
    : stepSize(stepSize), tsPair(tsPair), minimiser(minimiser.clone())
  {}


  vector<vector<double>> Downhill::run() {
    auto trackPath = [this](int iter, State& state) {
      auto coords = state.coords();
      if ((iter==0 && mep.empty()) || (vec::norm(coords - mep.back()) > stepSize)) {
        mep.push_back(coords);
        mepEnergy.push_back(state.energy());
      }
    };

    minimiser->minimise(tsPair[0], trackPath);
    std::reverse(mep.begin(), mep.end());
    minimiser->minimise(tsPair[1], trackPath);
    return mep;
  }
}
