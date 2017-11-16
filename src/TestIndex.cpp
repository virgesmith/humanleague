#include "UnitTester.h"
#include "NDArray.h"
#include "NDArrayUtils.h"
#include "Index.h"

#include <iostream>

void unittest::testIndex()
{
  // Full index tests 
  int64_t values2[] = {0,1,2, 10,11,12};
  NDArray<int64_t> a2({2,3}, values2);
  for (Index index2(a2.sizes()); !index2.end(); ++index2)
  {
    CHECK_EQUAL(a2[index2], index2[0] * 10 + index2[1]);
  }

  int64_t values3[] = {0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24, 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124};
  NDArray<int64_t> a3({2,3,5}, values3);
  
  for (Index index3(a3.sizes()); !index3.end(); ++index3)
  {
    CHECK_EQUAL(a3[index3], index3[0] * 100 + index3[1] * 10 + index3[2]);
  }

  // Transposed index tests
  // same values but in a different order
  {
    size_t i = 0;
    int64_t valuest2[6];
    for (TransposedIndex index2(a2.sizes()); !index2.end(); ++index2, ++i)
    {
      valuest2[i] = a2[index2];
    }
    // check 0,1,2,10,11,12 -> 0,10,1,11,2,12
    CHECK_EQUAL(valuest2[0], 0);
    CHECK_EQUAL(valuest2[1], 10);
    CHECK_EQUAL(valuest2[2], 1);
    CHECK_EQUAL(valuest2[3], 11);
    CHECK_EQUAL(valuest2[4], 2);
    CHECK_EQUAL(valuest2[5], 12);
    NDArray<int64_t> at2({3,2}, valuest2);
    TransposedIndex index2(at2.sizes());
    // check 0,10,1,11,2,12 -> 0,1,2,10,11,12
    CHECK_EQUAL(at2[  index2], 0);
    CHECK_EQUAL(at2[++index2], 1);
    CHECK_EQUAL(at2[++index2], 2);
    CHECK_EQUAL(at2[++index2], 10);
    CHECK_EQUAL(at2[++index2], 11);
    CHECK_EQUAL(at2[++index2], 12);
  }
  {
    size_t i = 0;
    int64_t valuest3[30];
    for (TransposedIndex index3(a3.sizes()); !index3.end(); ++index3, ++i)
    {
      valuest3[i] = a3[index3];
    }
    // check 0,1,2,10,11,12 -> 0,10,1,11,2,12
    //{0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24, 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124};
    CHECK_EQUAL(valuest3[0], 0);   CHECK_EQUAL(valuest3[1], 100);    
    CHECK_EQUAL(valuest3[2], 10);  CHECK_EQUAL(valuest3[3], 110);
    CHECK_EQUAL(valuest3[4], 20);  CHECK_EQUAL(valuest3[5], 120);    
    CHECK_EQUAL(valuest3[6], 1);   CHECK_EQUAL(valuest3[7], 101);
    CHECK_EQUAL(valuest3[8], 11);  CHECK_EQUAL(valuest3[9], 111);    
    CHECK_EQUAL(valuest3[10], 21); CHECK_EQUAL(valuest3[11], 121);
    CHECK_EQUAL(valuest3[12], 2);  CHECK_EQUAL(valuest3[13], 102);
    CHECK_EQUAL(valuest3[14], 12); CHECK_EQUAL(valuest3[15], 112);
    CHECK_EQUAL(valuest3[16], 22); CHECK_EQUAL(valuest3[17], 122);
    CHECK_EQUAL(valuest3[18], 3);  CHECK_EQUAL(valuest3[19], 103);
    CHECK_EQUAL(valuest3[20], 13); CHECK_EQUAL(valuest3[21], 113);
    CHECK_EQUAL(valuest3[22], 23); CHECK_EQUAL(valuest3[23], 123);
    CHECK_EQUAL(valuest3[24], 4);  CHECK_EQUAL(valuest3[25], 104);
    CHECK_EQUAL(valuest3[26], 14); CHECK_EQUAL(valuest3[27], 114);
    CHECK_EQUAL(valuest3[28], 24); CHECK_EQUAL(valuest3[29], 124);

    NDArray<int64_t> at3({5,3,2}, valuest3);
    TransposedIndex index3(at3.sizes());
    CHECK_EQUAL(at3[  index3], 0);   CHECK_EQUAL(at3[++index3], 1);   CHECK_EQUAL(at3[++index3], 2);   CHECK_EQUAL(at3[++index3], 3);   CHECK_EQUAL(at3[++index3], 4);
    CHECK_EQUAL(at3[++index3], 10);  CHECK_EQUAL(at3[++index3], 11);  CHECK_EQUAL(at3[++index3], 12);  CHECK_EQUAL(at3[++index3], 13);  CHECK_EQUAL(at3[++index3], 14);
    CHECK_EQUAL(at3[++index3], 20);  CHECK_EQUAL(at3[++index3], 21);  CHECK_EQUAL(at3[++index3], 22);  CHECK_EQUAL(at3[++index3], 23);  CHECK_EQUAL(at3[++index3], 24);
    CHECK_EQUAL(at3[++index3], 100); CHECK_EQUAL(at3[++index3], 101); CHECK_EQUAL(at3[++index3], 102); CHECK_EQUAL(at3[++index3], 103); CHECK_EQUAL(at3[++index3], 104);
    CHECK_EQUAL(at3[++index3], 110); CHECK_EQUAL(at3[++index3], 111); CHECK_EQUAL(at3[++index3], 112); CHECK_EQUAL(at3[++index3], 113); CHECK_EQUAL(at3[++index3], 114);
    CHECK_EQUAL(at3[++index3], 120); CHECK_EQUAL(at3[++index3], 121); CHECK_EQUAL(at3[++index3], 122); CHECK_EQUAL(at3[++index3], 123); CHECK_EQUAL(at3[++index3], 124);
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
    CHECK_EQUAL(index3[0], mindex0[0]);    
    CHECK_EQUAL(index3[1], mindex1[0]);    
    CHECK_EQUAL(index3[2], mindex2[0]);    

    CHECK_EQUAL(index3[0], mindex01[0]);    
    CHECK_EQUAL(index3[1], mindex01[1]);    

    CHECK_EQUAL(index3[0], mindex02[0]);    
    CHECK_EQUAL(index3[2], mindex02[1]);    

    CHECK_EQUAL(index3[1], mindex10[0]);    
    CHECK_EQUAL(index3[0], mindex10[1]);    

    CHECK_EQUAL(index3[1], mindex12[0]);    
    CHECK_EQUAL(index3[2], mindex12[1]);    

    CHECK_EQUAL(index3[2], mindex20[0]);    
    CHECK_EQUAL(index3[0], mindex20[1]);    

    CHECK_EQUAL(index3[2], mindex21[0]);    
    CHECK_EQUAL(index3[1], mindex21[1]);    
  }

  // Fixed index tests (uses slice)
  for (int64_t d = 0; d < a3.dim(); ++d)
  {
    for (int64_t i = 0; i < a3.sizes()[d]; ++i)
      {
        // define slice dimension and index
        std::vector<std::pair<int64_t, int64_t>> s(1,{d,i});
        const NDArray<int64_t>& a3sliced = slice(a3, s[0]);
        for (FixedIndex findex(a3.sizes(), s); !findex.end(); ++findex)
        {
          CHECK_EQUAL(a3[findex.operator const Index &()], a3sliced[findex.free()]);
        }
      }
  }
}
