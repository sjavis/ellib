#include "Bitss.h"

#include <string>
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


  Bitss::Bitss(const State& state1, const State& state2)
    : Bitss(state1, state2, std::move(std::unique_ptr<Lbfgs>(new Lbfgs)))
  {}


  Bitss::Bitss(const State& state1, const State& state2, std::unique_ptr<Minimiser> minimiser)
    : state(createState(state1, state2)), minimiser(std::move(minimiser))
  {}


  State Bitss::createState(const State& state1, const State& state2) {
    std::unique_ptr<Potential> pot(new BitssPotential);
    Vector coords = state1.getCoords();
    Vector coords2 = state2.getCoords();
    coords.insert(coords.end(), coords2.begin(), coords2.end());
    int ndof = state1.ndof + state2.ndof;
    std::unique_ptr<Potential::Args> args(new BitssArgs(state1, state2, ndof));
    return State(pot, coords, args);
  }


  State Bitss::run() {
    BitssArgs &args = static_cast<BitssArgs&> (*state.args);
    args.d0 = args.dist(args.state1.getCoords(), args.state2.getCoords());
    args.di = args.d0;
    for (int iter=0; iter<max_iter; iter++) {
      args.di = args.di * (1 - dist_step);
      minimiser->minimise(state, &adjustState);
    }
    return state;
  }


  void Bitss::adjustState(int iter, State& state) {
    BitssArgs &args = static_cast<BitssArgs&> (*state.args);
    // Update the coordinates for the individual states
    Vector coords = state.getCoords();
    auto coordsMid = coords.begin() + args.state1.ndof;
    args.state1.setCoords(Vector(coords.begin(), coordsMid));
    args.state2.setCoords(Vector(coordsMid, coords.end()));
    // Recompute the BITSS coefficients
    if (iter % args.coef_iter == 0) recomputeCoefficients(state);
  }


  void Bitss::recomputeCoefficients(State& state) {
    BitssArgs &args = static_cast<BitssArgs&> (*state.args);
    double e1 = args.state1.energy();
    double e2 = args.state2.energy();
    Vector g1 = args.state1.gradient();
    Vector g2 = args.state2.gradient();
    // Estimate energy barrier
    int n_interp = 10;
    double emin = std::min(e1, e2);
    double emax = emin;
    Vector coords1 = args.state1.blockCoords();
    Vector coords2 = args.state2.blockCoords();
    for (int i=1; i<n_interp; i++) {
      double t = double(i) / n_interp;
      Vector xtmp = (1-t)*coords1 + t*coords2;
      emax = std::max(emax, args.state1.energy(xtmp));
    }
    double eb = emax - emin;

    // Compute gradient magnitude in separation direction
    Vector dg = args.dist_grad(args.state1.getCoords(), args.state2.getCoords());
    double dgm = vec::norm(dg);
    double grad1 = vec::dotProduct(dg, g1) / dgm; // TODO: make these positive
    double grad2 = vec::dotProduct(dg, g2) / dgm; // vec::abs() needed
    double grad = std::max(sqrt(grad1+grad2), 2.828*eb/args.di);

    // Coefficients
    args.ke = args.alpha / (2 * eb);
    args.kd = grad / (2.828 * args.beta * args.di);
  }


  // Potential
  // TODO: Combine energy and gradient
  double Bitss::BitssPotential::energy(const Vector &coords, const Args &args_tmp) const {
    const BitssArgs &args = static_cast<const BitssArgs&> (args_tmp);
    // Single state energies
    double e1 = args.state1.energy();
    double e2 = args.state2.energy();
    double dist = args.dist(args.state1.getCoords(), args.state2.getCoords());
    // Constraints
    double ee = args.ke * pow(e1-e2, 2);
    double ed = args.kd * pow(dist-args.di, 2);
    // Total energy
    return e1 + e2 + ee + ed;
  }

  Vector Bitss::BitssPotential::gradient(const Vector &coords, const Args &args_tmp) const {
    const BitssArgs &args = static_cast<const BitssArgs&> (args_tmp);
    double e1 = args.state1.energy();
    double e2 = args.state2.energy();
    double dist = args.dist(args.state1.getCoords(), args.state2.getCoords());
    // Single state gradients
    Vector gs1 = args.state1.gradient();
    Vector gs2 = args.state2.gradient();
    // Energy constraint
    Vector ge1 = (1 + 2*args.ke*(e1-e2)) * gs1;
    Vector ge2 = (1 + 2*args.ke*(e2-e1)) * gs2;
    // Distance constraint
    Vector distg = args.dist_grad(args.state1.getCoords(), args.state2.getCoords());
    Vector gd = 2 * args.kd * (dist - args.di) * distg;
    // Total gradient
    Vector g = ge1 + gd;
    Vector g2 = ge2 - gd;
    g.insert(g.end(), g2.begin(), g2.end());
    return g;
  }


}
