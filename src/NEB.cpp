#include "NEB.h"
#include "minim/Lbfgs.h"
#include "minim/utils/vec.h"
#include "minim/utils/mpi.h"

using std::vector;

namespace ellib {

  NEB::NEB(const State& state1, const State& state2, int nImage, bool dneb)
    : NEB(interpolate(state1, state2, nImage), dneb)
  {}

  NEB::NEB(const vector<State>& chain, bool dneb) {
    auto pot = NEBPotential(chain, dneb);
    vector<double> coords;
    for (auto& state: chain) {
      auto x = state.coords();
      coords.insert(coords.end(), x.begin(), x.end());
    }
    this->state = State(pot, coords);
    this->minimiser = std::unique_ptr<Minimiser>(new Lbfgs);
  }


  vector<State> NEB::interpolate(const State& state1, const State& state2, int nImage) {
    auto x1 = state1.coords();
    auto x2 = state2.coords();
    vector<State> chain(nImage);
    for (int i=1; i<nImage-1; i++) {
      double t = (double)i / nImage;
      auto x = (1-t)*x1 + t*x2;
      chain[i] = State(*state1.pot, x);
    }
    return chain;
  }


  State NEB::run() {
    minimiser->minimise(state);
    return state;
  }


  NEB::NEBPotential::NEBPotential(vector<State> chain, bool dneb)
    : dneb(dneb), chain(chain)
  {}

  void NEB::NEBPotential::energyGradient(const vector<double> &coords, double* e, vector<double>* g) const {
    int nImage = chain.size();
    vector<double> eList(nImage);
    vector<vector<double>> gList(nImage);

    // Split coords into each state
    vector<vector<double>> xList(nImage);
    auto xStart = coords.begin();
    for (int iState=0; iState<nImage; iState++) {
      auto xEnd = xStart + chain[iState].ndof;
      xList[iState] = vector<double>(xStart, xEnd);
      xStart = xEnd;
    }

    // Get differences
    vector<vector<double>> xDiffs(nImage-1);
    vector<double> xDiffMags(nImage-1);
    for (int iPair=0; iPair<nImage-1; iPair++) {
      xDiffs[iPair] = xList[iPair+1] - xList[iPair];
      xDiffMags[iPair] = vec::norm(xDiffs[iPair]);
    }

    // Get tangent vectors using bisection method (J Chem Phys 113, 9978 (2000))
    vector<vector<double>> tau(nImage);
    tau[0] = xDiffs[0] / xDiffMags[0];
    tau[nImage] = xDiffs[nImage-1] / xDiffMags[nImage-1];
    for (int iState=1; iState<nImage-1; iState++) {
      tau[iState] = xDiffs[iState-1] / xDiffMags[iState-1] +  xDiffs[iState] / xDiffMags[iState];
      tau[iState] = tau[iState] / vec::norm(tau[iState]);
    }

    // Get single state energies + gradients (perpendicular to tau)
    for (int iState=1; iState<nImage-1; iState++) {
      if (chain[iState].usesThisProc) {
        chain[iState].energyGradient(xList[iState], (e)?&eList[iState]:nullptr, (g)?&gList[iState]:nullptr);
      }
    }
    // Using a second loop allows the states to first be evaluated without blocking
    for (int iState=1; iState<nImage-1; iState++) {
      int root = chain[iState].comm.ranks[0];
      if (e) mpi.bcast(eList[iState], root);
      if (g) mpi.bcast(gList[iState], root);
    }

    // Total energy
    if (e) {
      *e = vec::sum(eList);
      // Spring energy
      for (auto xDiffMag: xDiffMags) {
        *e += 0.5 * kSpring * pow(xDiffMag, 2);
      }
    }

    // Total gradient
    if (!g) return;
    for (int iState=1; iState<nImage-1; iState++) {
      // Spring gradient
      vector<double> gS = kSpring * (xDiffMags[iState-1] - xDiffMags[iState]) * tau[iState];
      // Parallel and perpendicular components
      vector<double> gPara = vec::dotProduct(gList[iState], tau[iState]) * tau[iState];
      vector<double> gPerp = gList[iState] - gPara;
      vector<double> gSPara = vec::dotProduct(gS, tau[iState]) * tau[iState];
      if (dneb) {
        vector<double> gSPerp = gS - gSPara;
        gList[iState] = gPerp + gS - vec::dotProduct(gSPerp, gPerp) * gPerp;
      } else {
        gList[iState] = gPerp + gSPara;
      }
    }
    (*g).reserve(coords.size());
    for (auto& gImage: gList) {
      (*g).insert(g->end(), gImage.begin(), gImage.end());
    }
  }

}
