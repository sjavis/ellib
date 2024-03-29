#include "ellib.h"

using namespace ellib;

int main(int argc, char** argv) {
  mpiInit(&argc, &argv);

  Lj3d pot;
  State s1(pot, {0,0,0});
  State s2(pot, {0,0,0});
  Bitss bitss(s1, s2);
  auto ts = bitss.run();
  print(ts);

  return 0;
}
