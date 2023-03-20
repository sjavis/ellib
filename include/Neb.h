#ifndef NEB_H
#define NEB_H

#include <vector>
#include <memory>
#include "minim/State.h"
#include "minim/Minimiser.h"

namespace ellib {

  using namespace minim;

  class Neb {
    public:
      int hybrid = 0;
      int hybridIter = 100;
      int nImage;
      State state;
      std::unique_ptr<Minimiser> minimiser;

      Neb(const Potential& pot, std::vector<std::vector<double>> coordList, bool dneb=false);
      Neb(const Potential& pot, std::vector<double> coords1, std::vector<double> coords2, int nImage, bool dneb=false);
      Neb& setHybrid(int method, int onIter);
      Neb& setKSpring(double kSpring);

      std::vector<std::vector<double>> run();
      std::vector<State> getChain();
      std::vector<double> getEnergies();
      std::vector<std::vector<double>> getCoords();

      class NebPotential : public NewPotential<NebPotential> {
        public:
          double kSpring = 1;
          bool dneb = false;
          int hybrid = 0;
          std::vector<State> chain;

          NebPotential(std::vector<State> chain, bool dneb) : dneb(dneb), chain(chain) { _energyGradientDef = true; };
          void energyGradient(const std::vector<double>& coords, double* e, std::vector<double>* g) const override;
          std::vector<std::vector<double>> setChainCoords(const std::vector<double>& coords);
      };
      NebPotential* pot;
  };


  class Dneb : public Neb {
    public:
      Dneb(const Potential& pot, std::vector<std::vector<double>> coordList)
        : Neb(pot, coordList, true) {};
      Dneb(const Potential& pot, std::vector<double> coords1, std::vector<double> coords2, int nImage)
        : Neb(pot, coords1, coords2, nImage, true) {};
  };

}
#endif
