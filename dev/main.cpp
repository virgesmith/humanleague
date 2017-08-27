
#include "src/IPF.h"


int main()
{
  std::array<std::vector<double>, 2> m = {std::vector<double>{1,2,3}, 
                                          std::vector<double>{3,3}};

  size_t size[2] = { m[0].size(), m[1].size() };                                        

  NDArray<2, double> s(size);
  s.assign(1.0);

  IPF(s, m);

}
