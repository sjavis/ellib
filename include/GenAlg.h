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

    public:
      Vector popEnergies;
      std::vector<State> pop;
      std::unique_ptr<Potential> pot;

      GenAlg(Potential& pot);
      GenAlg(const GenAlg& genAlg);
      GenAlg& operator=(const GenAlg& genAlg);
      ~GenAlg() {};

      Vector run();

      void initialise();
      std::vector<int> select();
      void newGeneration(std::vector<int> parents);
      void minimise();
      bool checkComplete();
      Vector getEnergies();
      std::vector<int> getBest(int n);


      // General GA parameters
      int maxIter = 100;
      int popSize = 100;
      int numElites = 1;
      double selectionRate = 0.3;
      double mutationRate = 0.1;
      std::string selectionMethod = "roulette";
      Vector pertubation = Vector();

      GenAlg& setMaxIter(int maxIter);
      GenAlg& setPopSize(int popSize);
      GenAlg& setNumElites(int numElites);
      GenAlg& setSelectionRate(double selectionRate);
      GenAlg& setMutationRate(double mutationRate);
      GenAlg& setSelectionMethod(std::string selectionMethod);
      GenAlg& setPertubation(Vector pertubation);

      // Convergence parameters
      int noImprovementConvergence = 0;
      double energyConvergence = - std::numeric_limits<double>::infinity();

      GenAlg& setConvergence(const std::string& method, double value);
      GenAlg& setNoImprovementConvergence(int noImprovementConvergence);
      GenAlg& setEnergyConvergence(double energyConvergence);

      // Initialisation parameters
      Vector2d bounds = Vector2d();
      std::function<State()> stateGen = nullptr;

      GenAlg& setStateGen(std::function<State()> stateGen);
      GenAlg& setBounds(Vector bound1, Vector bound2);

      // Other parameters
      std::unique_ptr<Minimiser> min = nullptr;
      std::function<void(int,GenAlg&)> iterFn;

      GenAlg& setMinimiser(const std::string& min);
      GenAlg& setMinimiser(const Minimiser& min);
      GenAlg& setIterFn(std::function<void(int,GenAlg&)> iterFn);

    private:
      int noImprovementIter;
      double bestEnergy;

      static thread_local std::mt19937 randEng;
      static float randF();
      static int randI(int n);
  };

}

#endif
