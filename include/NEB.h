#ifndef NEB_H
#define NEB_H

#include <vector>
#include <memory>
#include "minim/State.h"
#include "minim/Minimiser.h"

namespace ellib {

  using namespace minim;

  class NEB {
    public:
      int hybrid = 0;
      int hybridIter = 100;
      int nImage;
      State state;
      std::unique_ptr<Minimiser> minimiser;

      NEB(Potential pot, std::vector<std::vector<double>> coordList, bool dneb=false);
      NEB(Potential pot, std::vector<double> coords1, std::vector<double> coords2, int nImage, bool dneb=false);
      NEB& setHybrid(int method, int onIter);
      std::vector<State> run();

      class NEBPotential : public NewPotential<NEBPotential> {
        public:
          double kSpring = 1;
          bool dneb = false;
          int hybrid = 0;
          std::vector<State> chain;

          NEBPotential(std::vector<State> chain, bool dneb) : dneb(dneb), chain(chain) { _energyGradientDef = true; };
          void energyGradient(const std::vector<double>& coords, double* e, std::vector<double>* g) const override;
          void setChainCoords(const std::vector<double>& coords);
      };
  };

}
#endif
