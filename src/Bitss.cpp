#include "Bitss.h"

#include <string>
#include <math.h>
#include <algorithm>

#include "minim/State.h"
#include "minim/Minimiser.h"

#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"

#include "minim/utils/print.h"

namespace ellib {


  Bitss::Bitss(State state1, State state2, std::string minimiser) {
    Minimiser& min;
    switch (minimiser) {
      case "lbfgs":
        min = minim::Lbfgs();
        break;
      case "fire":
        min = minim::Fire();
        break;
      case "grad descent":
        min = minim::GradDescent();
        break;
      case "anneal":
        min = minim::Anneal();
        break;
      default:
        minim::print("Error: Unknown minimiser chosen.");
        stop;
    }
    Bitss(state1, state2, min);
  }


  Bitss::Bitss(State state1, State state2, Minimiser minimiser)
    : state1(state1), state2(state2), minimiser(minimiser)
  {
    Potential pot = BitssPotential();
    Vector coords = ;
    Args args();
    state = State(pot, coords, args);
  }


  void Bitss::run() {
    state.minimise()
  }


  void Bitss::recomputeCoefficients() {
    // Estimate energy barrier
    int n_interp = 10;
    double emin = std::min(state1.energy(), state2.energy());
    double emax = emin;
    for (int i=1; i<n_interp; i++) {
      double t = double(i) / n_interp;
      Vector xtmp = state1.blockCoords() + state2.blockCoords();
      emax = std::max(emax, state1.energy(xtmp));
    }
    eb = emax - emin;

    // Compute gradient magnitude in separation direction
    Vector dg = _dist_grad();
    double dgm = minim::vec::norm(dg);
    double grad1 = minim::vec::dotProduct(dg, state1.gradient()) / dgm;
    double grad2 = minim::vec::dotProduct(dg, state2.gradient()) / dgm;
    double grad = std::max(sqrt(dgrad1+dgrad2), 2.828*eb/_di)

    // Coefficients
    _ke = _alpha / (2 * eb);
    _kd = grad / (2.828 * _beta * _di);
  }


  // Potential
  Bitss::BitssPotential::energy(const Vector &coords, const Args &args) {
    e1 = state1.energy() // Need to update the
    e2 = state2.energy() // individual coordinates
  }


}
