#include "Bitss.h"

#include <string>
#include <math.h>
#include <algorithm>
#include <stdexcept>

#include "minim/State.h"
#include "minim/Minimiser.h"

#include "minim/Lbfgs.h"
#include "minim/Fire.h"
#include "minim/GradDescent.h"
#include "minim/Anneal.h"

#include "minim/utils/vec.h"
#include "minim/utils/print.h"

namespace ellib {

  typedef std::vector<double> Vector;


  Bitss::Bitss(State state1, State state2)
    : state(createState(state1, state2)), minimiser(_default_minimiser)
  {}


  Bitss::Bitss(State state1, State state2, Minimiser& minimiser)
    : state(createState(state1, state2)), minimiser(minimiser)
  {}


  State Bitss::createState(const State& state1, const State& state2) {
    BitssPotential pot;
    Vector coords = state1.getCoords();
    Vector coords2 = state2.getCoords();
    coords.insert(coords.end(), coords2.begin(), coords2.end());
    Args args(state1.ndof + state2.ndof);
    return State(pot, coords, args);
  }


  State Bitss::run() {
    BitssArgs &args = static_cast<BitssArgs&> (state.args);
    args.d0 = args.dist(args.state1->getCoords(), args.state2->getCoords());
    args.di = args.d0;
    for (int iter=0; iter<_max_iter; iter++) {
      args.di = args.di * (1 - _dist_step);
      minimiser.minimise(state, &adjustState);
    }
    return state;
  }


  void Bitss::adjustState(int iter, State& state) {
    BitssArgs &args = static_cast<BitssArgs&> (state.args);
    // Update the coordinates for the individual states
    Vector coords = state.getCoords();
    auto coordsMid = coords.begin() + args.state1->ndof;
    args.state1->setCoords(Vector(coords.begin(), coordsMid));
    args.state2->setCoords(Vector(coordsMid, coords.end()));
    // Recompute the BITSS coefficients
    if (iter % args.coef_iter == 0) recomputeCoefficients(state);
  }


  void Bitss::recomputeCoefficients(State& state) {
    BitssArgs &args = static_cast<BitssArgs&> (state.args);
    double e1 = args.state1->energy();
    double e2 = args.state2->energy();
    Vector g1 = args.state1->gradient();
    Vector g2 = args.state2->gradient();
    // Estimate energy barrier
    int n_interp = 10;
    double emin = std::min(e1, e2);
    double emax = emin;
    Vector coords1 = args.state1->blockCoords();
    Vector coords2 = args.state2->blockCoords();
    for (int i=1; i<n_interp; i++) {
      double t = double(i) / n_interp;
      Vector xtmp = (1-t)*coords1 + t*coords2;
      emax = std::max(emax, args.state1->energy(xtmp));
    }
    double eb = emax - emin;

    // Compute gradient magnitude in separation direction
    Vector dg = args.dist_grad(args.state1->getCoords(), args.state2->getCoords());
    double dgm = vec::norm(dg);
    double grad1 = vec::dotProduct(dg, g1) / dgm;
    double grad2 = vec::dotProduct(dg, g2) / dgm;
    double grad = std::max(sqrt(grad1+grad2), 2.828*eb/args.di);

    // Coefficients
    args.ke = args.alpha / (2 * eb);
    args.kd = grad / (2.828 * args.beta * args.di);
  }


  // Potential
  double Bitss::BitssPotential::energy(const Vector &coords, const Args &args_tmp) const {
    const BitssArgs &args = static_cast<const BitssArgs&> (args_tmp);
    // coords
    double e1 = args.state1->energy();
    double e2 = args.state2->energy();
    double e = e1 + e2;
    return e;
  }

  Vector Bitss::BitssPotential::gradient(const Vector &coords, const Args &args_tmp) const {
    const BitssArgs &args = static_cast<const BitssArgs&> (args_tmp);
    // coords
    Vector g1 = args.state1->gradient();
    Vector g2 = args.state2->gradient();
    Vector g = g1;
    g.insert(g.end(), g2.begin(), g2.end());
    return g;
  }


}
