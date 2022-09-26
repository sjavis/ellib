#ifndef BITSS_H
#define BITSS_H

#include <vector>
#include <string>
#include <memory>
#include "minim/State.h"
#include "minim/Potential.h"
#include "minim/Lbfgs.h"

namespace ellib {

  using namespace minim;

  class Bitss {
    typedef std::vector<double> Vector;
    typedef double (*DFunc)(const Vector&, const Vector&);
    typedef Vector (*DGFunc)(const Vector&, const Vector&);

    public:
      State state;
      Minimiser &minimiser;

      Bitss(State state1, State state2);
      Bitss(State state1, State state2, Minimiser& minimiser);
      ~Bitss() {};

      State run();

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
      double _e_scale_max = 0;
      Lbfgs _default_minimiser;
      int _iter;

      static State createState(const State& state1, const State& state2);
      static void adjustState(int iter, State& state);
      static void recomputeCoefficients(State& state);

      class BitssArgs : public Args {
        public:
          int coef_iter = 100;
          double alpha = 10;
          double beta = 0.1;
          double di;
          double d0;
          double ke;
          double kd;
          DFunc dist;
          DGFunc dist_grad;
          std::shared_ptr<State> state1;
          std::shared_ptr<State> state2;
      };

      class BitssPotential : public Potential {
        typedef std::vector<double> Vector;
        public:
          double energy(const Vector &coords, const Args &args) const override;
          Vector gradient(const Vector &coords, const Args &args) const override;
      };
  };

}

#endif
