#ifndef DOWNHILL_H
#define DOWNHILL_H

#include <vector>
#include <string>
#include <memory>
#include "minim/State.h"
#include "minim/Minimiser.h"

namespace ellib {

  using namespace minim;
  using std::vector;

  class Downhill {
    public:
      double stepSize;
      vector<State> tsPair;
      std::unique_ptr<Minimiser> minimiser;
      vector<vector<double>> mep;
      vector<double> mepEnergy;

      Downhill(const vector<State>& tsPair, double stepSize, std::string minimiser="GradDescent");
      Downhill(const vector<State>& tsPair, double stepSize, const Minimiser& minimiser);
      ~Downhill() {};

      vector<vector<double>> run();
  };

}

#endif
