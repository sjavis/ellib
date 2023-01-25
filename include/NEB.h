#ifndef NEB_H
#define NEB_H

#include <vector>
#include <memory>
#include "minim/State.h"
#include "minim/Minimiser.h"

namespace ellib {

  using namespace minim;

  class NEB {
    public:
      State state;
      std::unique_ptr<Minimiser> minimiser;

      NEB(const State& state1, const State& state2, int nImage, bool dneb=false);
      NEB(const std::vector<State>& chain, bool dneb=false);
      std::vector<State> interpolate(const State& state1, const State& state2, int nImage);
      State run();

      class NEBPotential : public NewPotential<NEBPotential> {
        public:
          double kSpring = 1;
          bool dneb = false;
          int hybrid = 0;
          std::vector<State> chain;

          NEBPotential(std::vector<State> chain, bool dneb);
          void energyGradient(const std::vector<double> &coords, double* e, std::vector<double>* g) const override;
      };
  };

}
#endif
