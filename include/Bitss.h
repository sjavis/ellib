#ifndef BITSS_H
#define BITSS_H

#include <vector>
#include <string>
#include <memory>
#include "minim/State.h"
#include "minim/Minimiser.h"
#include "minim/Potential.h"
#include "minim/utils/vec.h"

namespace ellib {

  using namespace minim;

  class Bitss {
    typedef std::vector<double> Vector;
    typedef double (*DFunc)(const Vector&, const Vector&);
    typedef Vector (*DGFunc)(const Vector&, const Vector&);

    public:
      State state;
      std::unique_ptr<Minimiser> minimiser;

      Bitss(const State& state1, const State& state2);
      Bitss(const State& state1, const State& state2, std::unique_ptr<Minimiser> minimiser);
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
      int _iter;

      static State createState(const State& state1, const State& state2);
      static void adjustState(int iter, State& state);
      static void recomputeCoefficients(State& state);

      class BitssArgs : public Potential::Args {
        public:
          int coef_iter = 100;
          double alpha = 10;
          double beta = 0.1;
          double di;
          double d0;
          double ke;
          double kd;
          DFunc dist = [](const Vector &x1, const Vector &x2) -> double { return vec::norm(x1-x2); };
          DGFunc dist_grad = [](const Vector &x1, const Vector &x2) -> Vector { return 1/vec::norm(x1-x2) * (x1-x2); };
          State state1;
          State state2;

          BitssArgs(const State& state1, const State& state2, int ndof) : Args(ndof), state1(state1), state2(state2) {};
      };

      class BitssPotential : public NewPotential<BitssPotential> {
        typedef std::vector<double> Vector;
        public:
          double energy(const Vector &coords, const Args &args) const override;
          Vector gradient(const Vector &coords, const Args &args) const override;
      };
  };

}

#endif
