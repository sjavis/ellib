#include "ellib.h"

int main(int argc, char** argv) {
  ellib::mpiInit(&argc, &argv);
        
  ellib::Lj3d pot;
  ellib::State s1 = pot.newState({0,0,0});
  ellib::State s2 = pot.newState({0,0,0});
  ellib::Bitss bitss(s1, s2);
  ellib::State result = bitss.run();

  return 0;
}
