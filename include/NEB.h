#ifndef NEB_H
#define NEB_H

#include <vector>

namespace ellib {

  using namespace minim;

  class NEB {
    using std::vector;

    public:
      State state;
      std::unique_ptr<Minimiser> minimiser;

      NEB(const State& state1, const State& state2, int nImage);
      NEB(const vector<State>& stateList);
      vector<State> interpolate(const State& state1, const State& state2, int nImage);
      State run();

      class NEBPotential : public NewPotential<NEBPotential> {
        public:
          double kSpring = 1;
          bool dneb = false;
          int hybrid = 0;
          vector<State> chain;

          void energyGradient(const Vector &coords, double* e, vector<double>* g) const override;
      };
  };

}
#endif
