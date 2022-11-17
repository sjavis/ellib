#ifndef GENALG_H
#define GENALG_H

#include <vector>
#include <string>
#include "minim/State.h"
// #include "minim/Potential.h"
#include "minim/Minimiser.h"

namespace ellib {
  using namespace minim;

  class GenAlg {
    typedef std::vector<double> Vector;
    typedef State (*StateFn)();

    public:
      GenAlg(Potential& pot);
      ~GenAlg() {};

      GenAlg& setMinimiser(const std::string& min="Lbfgs");
      GenAlg& setMinimiser(std::unique_ptr<Minimiser> min);
      GenAlg& setBounds(Vector bound1, Vector bound2);
      GenAlg& setStateGen(StateFn stateGen);

      Vector run();

      void initialise();
      void select();
      void crossover();
      void mutate();
      void checkComplete();

      int maxIter = 100;
      int populationSize = 100;
      int numElites = 1;
      double selectionRate = 0.3;
      double mutationRate = 0.1;
      StateFn stateGen;
      std::vector<Vector> bounds;
      std::vector<State> pop;
      std::unique_ptr<Minimiser> min;
      std::unique_ptr<Potential> pot;
  };
}

#endif
