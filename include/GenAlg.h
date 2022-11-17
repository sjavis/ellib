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

      Vector run();

      void initialise();
      void select();
      void crossover();
      void mutate();
      void checkComplete();

      GenAlg& setMaxIter(int maxIter);
      GenAlg& setPopSize(int popSize);
      GenAlg& setNumElites(int numElites);
      GenAlg& setSelectionRate(double selectionRate);
      GenAlg& setMutationRate(double mutationRate);
      GenAlg& setStateGen(StateFn stateGen);
      GenAlg& setBounds(Vector bound1, Vector bound2);
      GenAlg& setMinimiser(const std::string& min);
      GenAlg& setMinimiser(std::unique_ptr<Minimiser> min);

      int maxIter = 100;
      int popSize = 100;
      int numElites = 1;
      double selectionRate = 0.3;
      double mutationRate = 0.1;
      StateFn stateGen = nullptr;
      std::vector<Vector> bounds = std::vector<Vector>();
      std::unique_ptr<Minimiser> min = nullptr;

      std::unique_ptr<Potential> pot;
      std::vector<State> pop;
  };
}

#endif
