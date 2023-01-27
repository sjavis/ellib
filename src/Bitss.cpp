#include "Bitss.h"

#include <math.h>
#include <algorithm>
#include <stdexcept>

#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"

#include "minim/utils/vec.h"
#include "minim/utils/mpi.h"

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
      throw std::invalid_argument("Invalid minimiser chosen");
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

  Bitss& Bitss::setConvergenceDist(double convergenceDist) {
    this->convergenceDist = convergenceDist;
    return *this;
  }

  Bitss& Bitss::setConvergenceMethod(std::string convergenceMethod) {
    if (convergenceMethod != "distance" &&
        convergenceMethod != "relative distance" &&
        convergenceMethod != "midpoint change" &&
        convergenceMethod != "midpoint gradient") {
      throw std::invalid_argument("Invalid convergence method chosen");
    }
    this->convergenceMethod = convergenceMethod;
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
    _pot->d0 = _pot->dist(_pot->state1.coords(), _pot->state2.coords());
    _pot->di = _pot->d0;
    for (_iter=0; _iter<maxIter; _iter++) {
      _pot->di = _pot->di * (1 - distStep);
      minimiser->minimise(state, &adjustState);
      if (checkConvergence()) break;
    }
    return state;
  }


  State Bitss::getTS() {
    auto bitssPot = static_cast<BitssPotential&>(*state.pot);
    auto pot = *(bitssPot.state1.pot);
    return State(pot, getTSCoords());
  }


  Vector Bitss::getTSCoords() {
    auto bitssPot = static_cast<BitssPotential&>(*state.pot);
    auto coords1 = bitssPot.state1.coords();
    auto coords2 = bitssPot.state2.coords();
    return interp(coords1, coords2, 0.5);
  }


  std::vector<State> Bitss::getPair() {
    auto bitssPot = static_cast<BitssPotential&>(*state.pot);
    return {bitssPot.state1, bitssPot.state2};
  }


  std::vector<Vector> Bitss::getPairCoords() {
    auto pair = getPair();
    return {pair[0].coords(), pair[1].coords()};
  }


  void Bitss::adjustState(int iter, State& state) {
    auto pot = static_cast<BitssPotential*>(state.pot.get());
    // Update the coordinates for the individual states
    // TODO: Make these point to the same value, if the coords are updated the single states are only updated on the next loop
    Vector coords = state.coords();
    auto coordsMid = coords.begin() + pot->state1.ndof;
    pot->state1.coords(Vector(coords.begin(), coordsMid));
    pot->state2.coords(Vector(coordsMid, coords.end()));
    // Recompute the BITSS coefficients
    if (iter % pot->coefIter == 0) recomputeCoefficients(state);
  }


  void Bitss::recomputeCoefficients(State& state) {
    auto pot = static_cast<BitssPotential*>(state.pot.get());
    Vector coords = state.coords();
    auto coordsMid = coords.begin() + pot->state1.ndof;
    Vector coords1(coords.begin(), coordsMid);
    Vector coords2(coordsMid, coords.end());
    double e1;
    double e2;
    Vector g1;
    Vector g2;
    pot->state1.energyGradient(coords1, &e1, &g1);
    pot->state2.energyGradient(coords2, &e2, &g2);
    // Estimate energy barrier
    int nInterp = 10;
    double emin = std::min(e1, e2);
    double emax = std::max(e1, e2);
    for (int i=1; i<nInterp; i++) {
      double t = double(i) / nInterp;
      Vector xtmp = interp(coords1, coords2, t);
      emax = std::max(emax, pot->state1.energy(xtmp));
    }
    double eb = emax - emin; // TODO: ensure eb is not zero

    // Compute gradient magnitude in separation direction
    Vector dg = pot->distGrad(coords1, coords2);
    double dgm = vec::norm(dg);
    double grad1 = abs(vec::dotProduct(dg, g1)) / dgm;
    double grad2 = abs(vec::dotProduct(dg, g2)) / dgm;
    double grad = std::max(sqrt(grad1+grad2), 2.828*eb/pot->di);

    // Coefficients
    pot->ke = pot->alpha / (2 * eb);
    pot->kd = grad / (2.828 * pot->beta * pot->di);
  }
  

  bool Bitss::checkConvergence() {
    bool converged = false;
    if (convergenceMethod == "distance") {
      converged = (_pot->di <= convergenceDist);
    } else if (convergenceMethod == "relative distance") {
      converged = (_pot->di/_pot->d0 <= convergenceDist);
    } else if (convergenceMethod == "midpoint gradient") {
      Vector coords1 = _pot->state1.blockCoords();
      Vector coords2 = _pot->state2.blockCoords();
      Vector ts = interp(coords1, coords2, 0.5);
      Vector g = _pot->state1.gradient(ts);
      double grms = vec::norm(g) / sqrt(g.size());
      converged = (grms <= state.convergence);
    } else if (convergenceMethod == "midpoint change") {
      Vector coords1 = _pot->state1.blockCoords();
      Vector coords2 = _pot->state2.blockCoords();
      Vector ts = interp(coords1, coords2, 0.5);
      if (_iter > 0) {
        double diff = _pot->dist(ts, _tsOld);
        converged = (diff <= convergenceDist);
      }
      _tsOld = ts;
    }
    return converged;
  }


  Vector Bitss::interp(const Vector& coords1, const Vector& coords2, double t) {
    // if (coords1.size() != coords2.size()) {
    //   throw std::invalid_argument("Attempting linear interpolation with different sized vectors");
    // }
    return (1-t)*coords1 + t*coords2;
  }


  // Potential
  Bitss::BitssPotential::BitssPotential(const State& state1, const State& state2)
    : state1(state1), state2(state2)
  {
    _energyGradientDef = true;
  }

  State Bitss::BitssPotential::newState(const State& state1, const State& state2) {
    Vector coords = state1.coords();
    Vector coords2 = state2.coords();
    coords.insert(coords.end(), coords2.begin(), coords2.end());
    return State(BitssPotential(state1, state2), coords);
  }


  void Bitss::BitssPotential::energyGradient(const Vector &coords, double* e, Vector* g) const {
    auto coordsMid = coords.begin() + state1.ndof;
    Vector coords1(coords.begin(), coordsMid);
    Vector coords2(coordsMid, coords.end());
    double d = dist(coords1, coords2);
    // Single state energies / gradients
    double e1, e2;
    Vector g1, g2;
    state1.energyGradient(coords1, &e1, (g==nullptr)?nullptr:&g1);
    state2.energyGradient(coords2, &e2, (g==nullptr)?nullptr:&g2);
    // Total energy
    if (e != nullptr) {
      double ee = ke * pow(e1-e2, 2);
      double ed = kd * pow(d-di, 2);
      *e = e1 + e2 + ee + ed;
    }
    // Total gradient
    if (g != nullptr) {
      *g = Vector(coords.size());
      Vector dGrad = distGrad(coords1, coords2);
      Vector gd1 = 2 * kd * (d - di) * dGrad;
      Vector gtot1 = (1 + 2*ke*(e1-e2))*g1 + gd1;
      Vector gtot2 = (1 + 2*ke*(e2-e1))*g2 - gd1;
      std::copy(gtot1.begin(), gtot1.end(), (*g).begin());
      std::copy(gtot2.begin(), gtot2.end(), (*g).begin()+state1.ndof);
    }
  }


}
