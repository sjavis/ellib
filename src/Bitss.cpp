#include "Bitss.h"

#include <string>

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


  void Bitss::recomputeKe() {
  }


  void Bitss::recomputeKd() {
    int n_interp = 10;
    for (int i=0; i<n_interp; i++) {
    }
  }


  // Potential
  Bitss::BitssPotential::energy(const Vector &coords, const Args &args) {
    e1 = state1.energy() // Need to update the
    e2 = state2.energy() // individual coordinates
  }


}
