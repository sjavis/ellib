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
      class BitssPotential;

      int maxIter = 10;
      double distStep = 0.5;
      double convergenceDist = 0.01;
      double convergenceEnergy = 0.01;
      std::string convergenceMethod = "relative distance"; //!< Method to determine convergence. Possible values: distance, relative distance, midpoint gradient, midpoint change
      bool log = false;
      std::function<void(Bitss&)> logfn = nullptr;

      State state;
      std::unique_ptr<Minimiser> minimiser;

      Bitss(const State& state1, const State& state2, const std::string& minimiser="Lbfgs");
      Bitss(const State& state1, const State& state2, const Minimiser& minimiser);
      ~Bitss() {};

      Vector run();

      Bitss& setMaxIter(int maxIter);
      Bitss& setDistStep(double distStep);
      Bitss& setConvergenceDist(double convergenceDist);
      Bitss& setConvergenceGrad(double convergenceGrad);
      Bitss& setConvergenceEnergy(double convergenceEnergy);
      Bitss& setConvergenceMethod(std::string convergenceMethod);
      Bitss& setCoefIter(int coefIter);
      Bitss& setAlpha(double alpha);
      Bitss& setBeta(double beta);
      Bitss& setMaxBarrier(double maxBarrier);
      Bitss& setDistFunc(DFunc dist, DGFunc distGrad);
      Bitss& setLog(bool log=true);
      Bitss& setLog(std::function<void(Bitss&)> logfn);

      State getTS();
      Vector getTSCoords();
      std::vector<State> getPair();
      std::vector<Vector> getPairCoords();
      Bitss& setCoords(const Vector& coords);
      Bitss& setCoords(const Vector& coords1, const Vector& coords2);

      int iter() { return _iter; };

    private:
      int _iter;
      Vector _emin;
      Vector _tsOld;
      BitssPotential* _pot;

      static State createState(const State& state1, const State& state2);
      static void adjustState(int iter, State& state);
      static void recomputeCoefficients(State& state);
      static Vector interp(const Vector& coords1, const Vector& coords2, double t);
      bool checkConvergence();
      bool checkFailed();


    public:
      class BitssPotential : public NewPotential<BitssPotential> {
        typedef std::vector<double> Vector;
        public:
          int coefIter = 100;
          double alpha = 10;
          double beta = 0.1;
          double maxBarrier = 0;
          double di;
          double d0;
          double ke;
          double kd;
          State state1;
          State state2;
          DFunc dist = [](const Vector &x1, const Vector &x2) -> double { return vec::norm(x1-x2); };
          DGFunc distGrad = [](const Vector &x1, const Vector &x2) -> Vector { return 1/vec::norm(x1-x2) * (x1-x2); };

          BitssPotential(const State& state1, const State& state2);
          static State newState(const State& state1, const State& state2);

          void energyGradient(const Vector &coords, double* e, Vector* g) const override;
      };

  };

}

#endif
