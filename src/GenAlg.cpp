#include "GenAlg.h"

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"
#include "minim/utils/vec.h"

namespace ellib {

  typedef std::vector<double> Vector;
  typedef std::vector<Vector> Vector2d;


  GenAlg::GenAlg(Potential& pot)
    : pot(pot.clone())
  {}

  GenAlg::GenAlg(const GenAlg& genAlg)
    : pop(genAlg.pop),
      maxIter(genAlg.maxIter),
      popSize(genAlg.popSize),
      numElites(genAlg.numElites),
      selectionRate(genAlg.selectionRate),
      mutationRate(genAlg.mutationRate),
      pertubation(genAlg.pertubation),
      noImprovementConvergence(genAlg.noImprovementConvergence),
      energyConvergence(genAlg.energyConvergence),
      bounds(genAlg.bounds),
      iterFn(genAlg.iterFn)
  {
    if (genAlg.stateGen) stateGen = genAlg.stateGen;
    if (genAlg.min) min = genAlg.min->clone();
    if (genAlg.pot) pot = genAlg.pot->clone();
  }

  GenAlg& GenAlg::operator=(const GenAlg& genAlg) {
    pop = genAlg.pop;
    maxIter = genAlg.maxIter;
    popSize = genAlg.popSize;
    numElites = genAlg.numElites;
    selectionRate = genAlg.selectionRate;
    mutationRate = genAlg.mutationRate;
    pertubation = genAlg.pertubation;
    noImprovementConvergence = genAlg.noImprovementConvergence;
    energyConvergence = genAlg.energyConvergence;
    bounds = genAlg.bounds;
    iterFn = genAlg.iterFn;
    if (genAlg.stateGen) stateGen = genAlg.stateGen;
    if (genAlg.min) min = genAlg.min->clone();
    if (genAlg.pot) pot = genAlg.pot->clone();
    return *this;
  }


  GenAlg& GenAlg::setMaxIter(int maxIter) {
    this->maxIter = maxIter;
    return *this;
  }

  GenAlg& GenAlg::setPopSize(int popSize) {
    this->popSize = popSize;
    return *this;
  }

  GenAlg& GenAlg::setNumElites(int numElites) {
    this->numElites = numElites;
    return *this;
  }

  GenAlg& GenAlg::setSelectionRate(double selectionRate) {
    this->selectionRate = selectionRate;
    return *this;
  }

  GenAlg& GenAlg::setMutationRate(double mutationRate) {
    this->mutationRate = mutationRate;
    return *this;
  }

  GenAlg& GenAlg::setPertubation(Vector pertubation) {
    this->pertubation = pertubation;
    return *this;
  }

  GenAlg& GenAlg::setStateGen(StateFn stateGen) {
    this->stateGen = stateGen;
    return *this;
  }

  GenAlg& GenAlg::setBounds(Vector bound1, Vector bound2) {
    bounds = {bound1, bound2};
    for (size_t i=0; i<bound1.size(); i++) {
      if (bound1[i] > bound2[i]) {
        bounds[0][i] = bound2[i];
        bounds[1][i] = bound1[i];
      }
    }
    return *this;
  }

  GenAlg& GenAlg::setConvergence(const std::string& method, double value) {
    std::string string = method;
    std::transform(string.begin(), string.end(), string.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (string == "noimprovement") {
      noImprovementConvergence = (int) value;
    } else if (string == "energy") {
      energyConvergence = value;
    } else {
      throw std::invalid_argument("Invalid convergence method");
    }
    return *this;
  }

  GenAlg& GenAlg::setNoImprovementConvergence(int noImprovementConvergence) {
    this->noImprovementConvergence = noImprovementConvergence;
    return *this;
  }

  GenAlg& GenAlg::setEnergyConvergence(double energyConvergence) {
    this->energyConvergence = energyConvergence;
    return *this;
  }

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

  GenAlg& GenAlg::setIterFn(std::function<void(std::vector<State>&)> iterFn) {
    this->iterFn = iterFn;
    return *this;
  }


  Vector GenAlg::run() {
    initialise();
    for (int iter=0; iter<maxIter; iter++) {
      if (iter > 0) {
        auto parents = select();
        newGeneration(parents);
      }
      minimise();
      if (iterFn) iterFn(pop);
      getEnergies();
      if (checkComplete()) break;
    }

    auto bestState = getBestStates(1)[0];
    return bestState.coords();
  }


  void GenAlg::initialise() {
    // Initialise states
    if (stateGen != nullptr) {
      pop = std::vector<State>(popSize, stateGen());
      for (int i=1; i<popSize; i++) {
        pop[i] = stateGen();
      }
    } else if (!bounds.empty()) {
      State state1 = pot->newState( (bounds[1]+bounds[0])/2 );
      pop = std::vector<State>(popSize, state1);
      int ndof = state1.ndof;
      for (int i=1; i<popSize; i++) {
        Vector coords = vec::random(ndof, 0.5) + 0.5;
        coords = coords * (bounds[1] - bounds[0]) + bounds[0];
        pop[i] = pot->newState(coords);
      }
    } else {
      throw std::logic_error("Either a state generator function or coordinate boundaries must be supplied");
    }
    
    // Assign pertubation size
    if (pertubation.empty()) {
      if (bounds.empty()) {
        int ndof = pop[0].ndof;
        Vector2d allCoords(ndof, Vector(popSize));
        for (int j=0; j<popSize; j++) {
          Vector stateCoords = pop[j].coords();
          for (int i=0; i<ndof; i++) {
            allCoords[i][j] = stateCoords[i];
          }
        }
        bounds = Vector2d(2, Vector(ndof));
        for (int i=0; i<ndof; i++) {
          bounds[0][i] = *std::min_element(allCoords[i].begin(), allCoords[i].end());
          bounds[1][i] = *std::max_element(allCoords[i].begin(), allCoords[i].end());
        }
      }
      setPertubation(0.01*(bounds[1] - bounds[0]));
    }

    // Initialise other private variables
    noImprovementIter = 0;
    bestEnergy = std::numeric_limits<double>::infinity();
    popEnergies = Vector(popSize);
    getEnergies();
  }


  std::vector<State> GenAlg::select() {
    int nParents = selectionRate * popSize;
    nParents = std::max({nParents, numElites, 1});
    return getBestStates(nParents);
  }


  void GenAlg::newGeneration(const std::vector<State>& parents) {
    // Directly set elites
    std::copy(parents.begin(), parents.begin()+numElites, pop.begin());
    // Fill the remaining population
    for (int iPop=numElites; iPop<popSize; iPop++) {
      int i1 = randI(parents.size());
      int i2 = (randI(parents.size()-1) + i1+1) % parents.size(); // Ensure that i2 != i1
      pop[iPop].comm.bcast(i1);
      pop[iPop].comm.bcast(i2);
      auto block = parents[i1].blockCoords();
      auto block2 = parents[i2].blockCoords();
      int nblock = pop[iPop].comm.nblock;
      for (int iCoord=0; iCoord<nblock; iCoord++) {
        // Crossover
        if (randF() < 0.5) block[iCoord] = block2[iCoord];
        // Mutation
        if (randF() < mutationRate) block[iCoord] += pertubation[iCoord] * (2*randF()-1);
      }
      pop[iPop].blockCoords(block);
      pop[iPop].communicate();
    }
  }


  void GenAlg::minimise() {
    if (min != nullptr) {
      for (auto &state: pop) {
        min->minimise(state);
      }
    }
  }


  bool GenAlg::checkComplete() {
    // Calculate test parameters
    double bestEnergyNew = *std::min_element(popEnergies.begin(), popEnergies.end());
    if (bestEnergyNew < bestEnergy) {
      bestEnergy = bestEnergyNew;
      noImprovementIter = 0;
    } else {
      noImprovementIter ++;
    }
    // Get result
    if (bestEnergy <= energyConvergence) return true;
    if (noImprovementIter >= noImprovementConvergence) return true;
    return false;
  }


  Vector GenAlg::getEnergies() {
    for (int i=0; i<popSize; i++) {
      popEnergies[i] = pop[i].energy();
    }
    return popEnergies;
  }


  std::vector<State> GenAlg::getBestStates(int n) {
    // Get the indicies of the lowest energy states
    std::vector<int> index(popSize);
    std::iota(index.begin(), index.end(), 0);
    std::partial_sort(index.begin(), index.begin()+n, index.end(), [this](int i, int j){ return popEnergies[i]<popEnergies[j]; });
    // Get the vector of the best states
    std::vector<State> best(n, pop[0]); // Assign pop[0] because State has no default initialiser
    for (int i=0; i<n; i++) {
      best[i] = pop[index[i]];
    }
    return best;
  }

  float GenAlg::randF() {
    return std::uniform_real_distribution<>{0, 1}(randEng);
  }

  int GenAlg::randI(int n) {
    return std::uniform_int_distribution<>{0, n-1}(randEng);
  }

  unsigned seed = std::random_device()();
  thread_local std::mt19937 GenAlg::randEng(seed);

}
