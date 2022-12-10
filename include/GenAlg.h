#ifndef GENALG_H
#define GENALG_H

#include <vector>
#include <string>
#include <random>
#include <limits>
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
      std::vector<State> pop;
      std::unique_ptr<Potential> pot;

      GenAlg(Potential& pot);
      GenAlg(const GenAlg& genAlg);
      GenAlg& operator=(const GenAlg& genAlg);
      ~GenAlg() {};

      Vector run();

      void initialise();
      std::vector<State> select();
      void newGeneration(const std::vector<State>& parents);
      void minimise();
      bool checkComplete();
      Vector getEnergies();
      std::vector<State> getBestStates(int n);


      // General GA parameters
      int maxIter = 100;
      int popSize = 100;
      int numElites = 1;
      double selectionRate = 0.3;
      double mutationRate = 0.1;
      Vector pertubation = Vector();

      GenAlg& setMaxIter(int maxIter);
      GenAlg& setPopSize(int popSize);
      GenAlg& setNumElites(int numElites);
      GenAlg& setSelectionRate(double selectionRate);
      GenAlg& setMutationRate(double mutationRate);
      GenAlg& setPertubation(Vector pertubation);

      // Convergence parameters
      int noImprovementConvergence = 0;
      double energyConvergence = - std::numeric_limits<double>::infinity();

      GenAlg& setConvergence(const std::string& method, double value);
      GenAlg& setNoImprovementConvergence(int noImprovementConvergence);
      GenAlg& setEnergyConvergence(double energyConvergence);

      // Initialisation parameters
      Vector2d bounds = Vector2d();
      StateFn stateGen = nullptr;

      GenAlg& setStateGen(StateFn stateGen);
      GenAlg& setBounds(Vector bound1, Vector bound2);

      // Other parameters
      std::unique_ptr<Minimiser> min = nullptr;
      std::function<void(std::vector<State>&)> iterFn;

      GenAlg& setMinimiser(const std::string& min);
      GenAlg& setMinimiser(std::unique_ptr<Minimiser> min);
      GenAlg& setIterFn(std::function<void(std::vector<State>&)> iterFn);

    private:
      int noImprovementIter;
      double bestEnergy;
      Vector popEnergies;

      static thread_local std::mt19937 randEng;
      static float randF();
      static int randI(int n);
  };

}

#endif
