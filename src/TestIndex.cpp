#include "UnitTester.h"
#include "NDArray.h"
#include "Index.h"

#include <iostream>

void unittest::testIndex()
{
  // Full index tests 
  int64_t values2[] = {0,1,2, 10,11,12};
  NDArray<int64_t> a2({2,3}, values2);
  
  for (Index index2(a2.sizes()); !index2.end(); ++index2)
  {
    CHECK(a2[index2] == index2[0] * 10 + index2[1]);
  }

  int64_t values3[] = {0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24, 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124};
  NDArray<int64_t> a3({2,3,5}, values3);
  
  for (Index index3(a3.sizes()); !index3.end(); ++index3)
  {
    CHECK(a3[index3] == index3[0] * 100 + index3[1] * 10 + index3[2]);
  }

  // Mapped Index tests
  Index index3(a3.sizes());
  MappedIndex mindex0(index3, {0});
  MappedIndex mindex1(index3, {1});
  MappedIndex mindex2(index3, {2});
  MappedIndex mindex01(index3, {0,1});
  MappedIndex mindex02(index3, {0,2});
  MappedIndex mindex10(index3, {1,0});
  MappedIndex mindex12(index3, {1,2});
  MappedIndex mindex20(index3, {2,0});
  MappedIndex mindex21(index3, {2,1});
  for (; !index3.end(); ++index3)
  {
    CHECK(index3[0] == mindex0[0]);    
    CHECK(index3[1] == mindex1[0]);    
    CHECK(index3[2] == mindex2[0]);    

    CHECK(index3[0] == mindex01[0]);    
    CHECK(index3[1] == mindex01[1]);    

    CHECK(index3[0] == mindex02[0]);    
    CHECK(index3[2] == mindex02[1]);    

    CHECK(index3[1] == mindex10[0]);    
    CHECK(index3[0] == mindex10[1]);    

    CHECK(index3[1] == mindex12[0]);    
    CHECK(index3[2] == mindex12[1]);    

    CHECK(index3[2] == mindex20[0]);    
    CHECK(index3[0] == mindex20[1]);    

    CHECK(index3[2] == mindex21[0]);    
    CHECK(index3[1] == mindex21[1]);    
  }

  // Fixed index tests
  std::vector<std::pair<int64_t, int64_t>> d0i0(1,{0,0});
  for (FixedIndex findex(a3.sizes(), d0i0); !findex.end(); ++findex)
  {
    //std:: cout << a3[findex.operator const Index &()] << std::endl;
    //CHECK(a3[findex.operator const Index &()] == a3[findex.fullIndex()]);
  }
}
