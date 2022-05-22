#ifndef BITSS_H
#define BITSS_H

#include <vector>
#include <string>
#include "minim/State.h"
#include "minim/Potential.h"
#include "minim/Minimiser.h"

namespace ellib {

  using minim::State;
  using minim::Args;
  using minim::Potential;
  using minim::Minimiser;

  class Bitss {
    private:
      typedef std::vector<double> Vector;
      typedef double (*DFunc)(const Vector&, const Vector&)
      typedef Vector (*DGFunc)(const Vector&, const Vector&)

    public:
      State state;
      State state1;
      State state2;
      Minimiser minimiser;

      Bitss(State state1, State state2, std::string minimiser="lbfgs");
      Bitss(State state1, State state2, Minimiser minimiser);
      ~Bitss() {};

      void run();

      Bitss& setMaxIter(int max_iter);
      Bitss& setDistStep(double dist_step);
      Bitss& setDistCutoff(double dist_cutoff);
      Bitss& setAlpha(double alpha);
      Bitss& setBeta(double beta);
      Bitss& setEScaleMax(double e_scale_max);
      Bitss& setDistFunc(DFunc dist, DGFunc dist_grad);


    private:
      int _max_iter = 10;
      double _dist_step = 0.5;
      double _dist_cutoff = 0.01;
      double _alpha = 10;
      double _beta = 0.1;
      double _e_scale_max = 0;

      int _iter;
      double _di;
      double _d0;
      double _ke;
      double _kd;
      DFunc _dist;
      DGFunc _dist_grad;

      void recomputeCoefficients();

      class BitssPotential : public Potential {
        public:
          double energy(const Vector &coords, const Args &args) override;
          Vector gradient(const Vector &coords, const Args &args) override;
      }
  };

}

#endif
