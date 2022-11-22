#ifndef GENALG_H
#define GENALG_H

#include <vector>
#include <string>
#include <random>
#include <functional>
#include "minim/State.h"
#include "minim/Minimiser.h"

namespace ellib {
  using namespace minim;

  class GenAlg {
    typedef std::vector<double> Vector;
    typedef std::vector<Vector> Vector2d;
    typedef State (*StateFn)();

    public:
      GenAlg(Potential& pot);
      ~GenAlg() {};

      Vector run();

      void initialise();
      std::vector<State> select();
      void newGeneration(const std::vector<State>& parents);
      void minimise();
      bool checkComplete();

      GenAlg& setMaxIter(int maxIter);
      GenAlg& setPopSize(int popSize);
      GenAlg& setNumElites(int numElites);
      GenAlg& setSelectionRate(double selectionRate);
      GenAlg& setMutationRate(double mutationRate);
      GenAlg& setStateGen(StateFn stateGen);
      GenAlg& setBounds(Vector bound1, Vector bound2);
      GenAlg& setPertubation(Vector pertubation);
      GenAlg& setMinimiser(const std::string& min);
      GenAlg& setMinimiser(std::unique_ptr<Minimiser> min);
      GenAlg& setIterFn(std::function<void(std::vector<State>&)> iterFn);

      int maxIter = 100;
      int popSize = 100;
      int numElites = 1;
      double selectionRate = 0.3;
      double mutationRate = 0.1;
      Vector pertubation = Vector();
      Vector2d bounds = Vector2d();
      StateFn stateGen = nullptr;
      std::unique_ptr<Minimiser> min = nullptr;
      std::function<void(std::vector<State>&)> iterFn;

      std::unique_ptr<Potential> pot;
      std::vector<State> pop;

    private:
      Vector getEnergies();
      std::vector<State> getBestStates(int n, Vector energies);

      static thread_local std::mt19937 randEng;
      static float randF();
      static int randI(int n);
  };

}

#endif
