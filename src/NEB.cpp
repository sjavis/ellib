#include "NEB.h"
#include "minim/Lbfgs.h"
#include "minim/utils/vec.h"
#include "minim/utils/mpi.h"

using std::vector;

namespace ellib {

  std::vector<std::vector<int>> getRanks(int nImage) {
    std::vector<std::vector<int>> ranks(nImage);
    if (nImage < mpi.size) {
      // Multiple processors per state
      int nBase = mpi.size / nImage;
      int nRemainder = mpi.size - nBase*nImage;
      int iProc = 0;
      for (int i=0; i<nImage; i++) {
        int nProc = (i<nRemainder) ? nBase+1 : nBase;
        ranks[i] = std::vector<int>(nProc);
        std::iota(ranks[i].begin(), ranks[i].end(), iProc);
        iProc += nProc;
      }
    } else {
      // Multiple states per processor
      for (int i=0; i<nImage; i++) {
        int iProc = (i * mpi.size) / nImage;
        ranks[i] = {iProc};
      }
    }
    return ranks;
  }


  NEB::NEB(Potential pot, vector<vector<double>> coordList, bool dneb, int hybrid)
    : nImage(coordList.size()), state(State(pot, coordList[0])) // TODO: Add default State constructor so this is not needed
  {
    auto ranks = getRanks(nImage);
    vector<double> allCoords;
    vector<State> chain;
    for (int i=0; i<nImage; i++) {
      allCoords.insert(allCoords.end(), coordList[i].begin(), coordList[i].end());
      chain.push_back(State(pot, coordList[i], ranks[i]));
    }
    this->state = State(NEBPotential(chain, dneb, hybrid), allCoords);
    this->minimiser = std::unique_ptr<Minimiser>(new Lbfgs);
  }

  NEB::NEB(Potential pot, vector<double> coords1, vector<double> coords2, int nImage, bool dneb, int hybrid)
    : NEB(pot, interpolate(coords1, coords2, nImage), dneb, hybrid)
  {}


  std::vector<std::vector<double>> NEB::interpolate(vector<double> coords1, vector<double> coords2, int nImage) {
    vector<vector<double>> chain(nImage);
    for (int i=0; i<nImage; i++) {
      double t = i / (nImage - 1.0);
      chain[i] = (1-t)*coords1 + t*coords2;
    }
    return chain;
  }


  vector<State> NEB::run() {
    minimiser->minimise(state);
    auto nebPot = dynamic_cast<NEBPotential&>(*state.pot);
    nebPot.setChainCoords(state.coords());
    return nebPot.chain;
  }

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

    // Get single state energies + gradients
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
    tau[nImage-1] = xDiffs[nImage-2] / xDiffMags[nImage-2];
    for (int iState=1; iState<nImage-1; iState++) {
      tau[iState] = xDiffs[iState-1] / xDiffMags[iState-1] +  xDiffs[iState] / xDiffMags[iState];
      tau[iState] = tau[iState] / vec::norm(tau[iState]);
    }

    // Hybrid method
    int iMaxState;
    if (hybrid) {
      // Get highest energy state
      iMaxState = 1;
      double eMax = eList[1];
      for (int iState=2; iState<nImage-1; iState++) {
        if (eList[iState] > eMax) {
          iMaxState = iState;
          eMax = eList[iState];
        }
      }
      // Climbing image
      if (hybrid==1) gList[iMaxState] -= 2 * vec::dotProduct(gList[iMaxState], tau[iMaxState]) * tau[iMaxState];
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
      if (hybrid && iState==iMaxState) continue;
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
    gList[0] = vector<double>(chain[0].ndof);
    gList[nImage-1] = vector<double>(chain[nImage-1].ndof);
    (*g).clear();
    (*g).reserve(coords.size());
    for (auto& gImage: gList) {
      (*g).insert(g->end(), gImage.begin(), gImage.end());
    }
  }


  // Split coords among each state
  void NEB::NEBPotential::setChainCoords(const std::vector<double>& coords) {
    auto xStart = coords.begin();
    for (auto& state: chain) {
      auto xEnd = xStart + state.ndof;
      if (state.usesThisProc) state.coords(vector<double>(xStart, xEnd));
      xStart = xEnd;
    }
  }

}
