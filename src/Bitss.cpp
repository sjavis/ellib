#include "Bitss.h"

#include <math.h>
#include <algorithm>
#include <stdexcept>

#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"

#include "minim/utils/vec.h"

namespace ellib {

  typedef std::vector<double> Vector;


  Bitss::Bitss(const State& state1, const State& state2, const std::string& minimiser)
    : state(BitssPotential::newState(state1, state2))
  {
    _pot = static_cast<BitssPotential*>(state.pot.get());

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
      throw std::invalid_argument("Unknown minimiser chosen");
    }
  }

  Bitss::Bitss(const State& state1, const State& state2, std::unique_ptr<Minimiser> minimiser)
    : state(BitssPotential::newState(state1, state2)), minimiser(std::move(minimiser))
  {
    _pot = static_cast<BitssPotential*>(state.pot.get());
  }


  Bitss& Bitss::setMaxIter(int maxIter) {
    this->maxIter = maxIter;
    return *this;
  }

  Bitss& Bitss::setDistStep(double distStep) {
    this->distStep = distStep;
    return *this;
  }

  Bitss& Bitss::setDistCutoff(double distCutoff) {
    this->distCutoff = distCutoff;
    return *this;
  }
 
  Bitss& Bitss::setCoefIter(int coefIter) {
    _pot->coefIter = coefIter;
    return *this;
  }
 
  Bitss& Bitss::setAlpha(double alpha) {
    _pot->alpha = alpha;
    return *this;
  }
 
  Bitss& Bitss::setBeta(double beta) {
    _pot->beta = beta;
    return *this;
  }
 
  Bitss& Bitss::setEScaleMax(double eScaleMax) {
    this->eScaleMax = eScaleMax;
    return *this;
  }
 
  Bitss& Bitss::setDistFunc(DFunc dist, DGFunc distGrad) {
    _pot->dist = dist;
    _pot->distGrad = distGrad;
    return *this;
  }


  State Bitss::run() {
    _pot->d0 = _pot->dist(_pot->state1.getCoords(), _pot->state2.getCoords());
    _pot->di = _pot->d0;
    for (_iter=0; _iter<maxIter; _iter++) {
      _pot->di = _pot->di * (1 - distStep);
      minimiser->minimise(state, &adjustState);
      if (checkConvergence()) break;
    }
    return state;
  }


  void Bitss::adjustState(int iter, State& state) {
    auto pot = static_cast<BitssPotential*>(state.pot.get());
    // Update the coordinates for the individual states
    Vector coords = state.getCoords();
    auto coordsMid = coords.begin() + pot->state1.ndof;
    pot->state1.setCoords(Vector(coords.begin(), coordsMid));
    pot->state2.setCoords(Vector(coordsMid, coords.end()));
    // Recompute the BITSS coefficients
    if (iter % pot->coefIter == 0) recomputeCoefficients(state);
  }


  void Bitss::recomputeCoefficients(State& state) {
    auto pot = static_cast<BitssPotential*>(state.pot.get());
    double e1 = pot->state1.energy();
    double e2 = pot->state2.energy();
    Vector g1 = pot->state1.gradient();
    Vector g2 = pot->state2.gradient();
    // Estimate energy barrier
    int nInterp = 10;
    double emin = std::min(e1, e2);
    double emax = emin;
    Vector coords1 = pot->state1.blockCoords();
    Vector coords2 = pot->state2.blockCoords();
    for (int i=1; i<nInterp; i++) {
      double t = double(i) / nInterp;
      Vector xtmp = (1-t)*coords1 + t*coords2;
      emax = std::max(emax, pot->state1.energy(xtmp));
    }
    double eb = emax - emin;

    // Compute gradient magnitude in separation direction
    Vector dg = pot->distGrad(pot->state1.getCoords(), pot->state2.getCoords());
    double dgm = vec::norm(dg);
    double grad1 = abs(vec::dotProduct(dg, g1)) / dgm;
    double grad2 = abs(vec::dotProduct(dg, g2)) / dgm;
    double grad = std::max(sqrt(grad1+grad2), 2.828*eb/pot->di);

    // Coefficients
    pot->ke = pot->alpha / (2 * eb);
    pot->kd = grad / (2.828 * pot->beta * pot->di);
  }
  

  bool Bitss::checkConvergence() {
    return false;
  }


  // Potential
  Bitss::BitssPotential::BitssPotential(const State& state1, const State& state2)
    : state1(state1), state2(state2)
  {}

  State Bitss::BitssPotential::newState(const State& state1, const State& state2) {
    Vector coords = state1.getCoords();
    Vector coords2 = state2.getCoords();
    coords.insert(coords.end(), coords2.begin(), coords2.end());
    return State(BitssPotential(state1, state2), coords);
  }


  // TODO: Combine energy and gradient
  double Bitss::BitssPotential::energy(const Vector &coords) const {
    // Single state energies
    double e1 = state1.energy();
    double e2 = state2.energy();
    double d = dist(state1.getCoords(), state2.getCoords());
    // Constraints
    double ee = ke * pow(e1-e2, 2);
    double ed = kd * pow(d-di, 2);
    // Total energy
    return e1 + e2 + ee + ed;
  }

  Vector Bitss::BitssPotential::gradient(const Vector &coords) const {
    double e1 = state1.energy();
    double e2 = state2.energy();
    double d = dist(state1.getCoords(), state2.getCoords());
    // Single state gradients
    Vector gs1 = state1.gradient();
    Vector gs2 = state2.gradient();
    // Energy constraint
    Vector ge1 = (1 + 2*ke*(e1-e2)) * gs1;
    Vector ge2 = (1 + 2*ke*(e2-e1)) * gs2;
    // Distance constraint
    Vector distg = distGrad(state1.getCoords(), state2.getCoords());
    Vector gd = 2 * kd * (d - di) * distg;
    // Total gradient
    Vector g = ge1 + gd;
    Vector g2 = ge2 - gd;
    g.insert(g.end(), g2.begin(), g2.end());
    return g;
  }


}
