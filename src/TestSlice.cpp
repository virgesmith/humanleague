#include "UnitTester.h"

#include "NDArray.h"
#include "NDArrayUtils.h"

#include <vector>

void unittest::testSlice()
{
  {
    int64_t values2[] = {0,1,2, 10,11,12};
    NDArray<int64_t> a2({2,3}, values2);

    const NDArray<int64_t>& s00 = slice(a2, {0,0});
    Index i00(s00.sizes());
    CHECK_EQUAL(s00[  i00], 0);
    CHECK_EQUAL(s00[++i00], 1);
    CHECK_EQUAL(s00[++i00], 2);
    
    const NDArray<int64_t>& s01 = slice(a2, {0,1});
    Index i01(s01.sizes());
    CHECK_EQUAL(s01[  i01], 10);
    CHECK_EQUAL(s01[++i01], 11);
    CHECK_EQUAL(s01[++i01], 12);

    const NDArray<int64_t>& s10 = slice(a2, {1,0});
    Index i10(s10.sizes());
    CHECK_EQUAL(s10[  i10], 0);
    CHECK_EQUAL(s10[++i10], 10);

    const NDArray<int64_t>& s11 = slice(a2, {1,1});
    Index i11(s11.sizes());
    CHECK_EQUAL(s11[  i11], 1);
    CHECK_EQUAL(s11[++i11], 11);

    const NDArray<int64_t>& s12 = slice(a2, {1,2});
    Index i12(s12.sizes());
    CHECK_EQUAL(s12[  i12], 2);
    CHECK_EQUAL(s12[++i12], 12);
  }
  {
    int64_t values3[] = {0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24, 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124};
    NDArray<int64_t> a3({2,3,5}, values3);

    const NDArray<int64_t>& s00 = slice(a3, {0,0});
    Index i00(s00.sizes());
    CHECK_EQUAL(s00[  i00],   0); CHECK_EQUAL(s00[++i00],   1); CHECK_EQUAL(s00[++i00],   2); CHECK_EQUAL(s00[++i00],   3); CHECK_EQUAL(s00[++i00],   4);
    CHECK_EQUAL(s00[++i00],  10); CHECK_EQUAL(s00[++i00],  11); CHECK_EQUAL(s00[++i00],  12); CHECK_EQUAL(s00[++i00],  13); CHECK_EQUAL(s00[++i00],  14);
    CHECK_EQUAL(s00[++i00],  20); CHECK_EQUAL(s00[++i00],  21); CHECK_EQUAL(s00[++i00],  22); CHECK_EQUAL(s00[++i00],  23); CHECK_EQUAL(s00[++i00],  24);

    const NDArray<int64_t>& s01 = slice(a3, {0,1});
    Index i01(s01.sizes());
    CHECK_EQUAL(s01[  i01], 100); CHECK_EQUAL(s01[++i01], 101); CHECK_EQUAL(s01[++i01], 102); CHECK_EQUAL(s01[++i01], 103); CHECK_EQUAL(s01[++i01], 104);
    CHECK_EQUAL(s01[++i01], 110); CHECK_EQUAL(s01[++i01], 111); CHECK_EQUAL(s01[++i01], 112); CHECK_EQUAL(s01[++i01], 113); CHECK_EQUAL(s01[++i01], 114);
    CHECK_EQUAL(s01[++i01], 120); CHECK_EQUAL(s01[++i01], 121); CHECK_EQUAL(s01[++i01], 122); CHECK_EQUAL(s01[++i01], 123); CHECK_EQUAL(s01[++i01], 124);

    const NDArray<int64_t>& s10 = slice(a3, {1,0});
    Index i10(s10.sizes());
    CHECK_EQUAL(s10[  i10],   0); CHECK_EQUAL(s10[++i10],   1); CHECK_EQUAL(s10[++i10],   2); CHECK_EQUAL(s10[++i10],   3); CHECK_EQUAL(s10[++i10],   4);
    CHECK_EQUAL(s10[++i10], 100); CHECK_EQUAL(s10[++i10], 101); CHECK_EQUAL(s10[++i10], 102); CHECK_EQUAL(s10[++i10], 103); CHECK_EQUAL(s10[++i10], 104);

    const NDArray<int64_t>& s11 = slice(a3, {1,1});
    Index i11(s11.sizes());
    CHECK_EQUAL(s11[  i11],  10); CHECK_EQUAL(s11[++i11],  11); CHECK_EQUAL(s11[++i11],  12); CHECK_EQUAL(s11[++i11],  13); CHECK_EQUAL(s11[++i11],  14);
    CHECK_EQUAL(s11[++i11], 110); CHECK_EQUAL(s11[++i11], 111); CHECK_EQUAL(s11[++i11], 112); CHECK_EQUAL(s11[++i11], 113); CHECK_EQUAL(s11[++i11], 114);

    const NDArray<int64_t>& s12 = slice(a3, {1,2});
    Index i12(s11.sizes());
    CHECK_EQUAL(s12[  i12],  20); CHECK_EQUAL(s12[++i12],  21); CHECK_EQUAL(s12[++i12],  22); CHECK_EQUAL(s12[++i12],  23); CHECK_EQUAL(s12[++i12],  24);
    CHECK_EQUAL(s12[++i12], 120); CHECK_EQUAL(s12[++i12], 121); CHECK_EQUAL(s12[++i12], 122); CHECK_EQUAL(s12[++i12], 123); CHECK_EQUAL(s12[++i12], 124);

    const NDArray<int64_t>& s20 = slice(a3, {2,0});
    Index i20(s20.sizes());
    CHECK_EQUAL(s20[  i20],   0); CHECK_EQUAL(s20[++i20],  10); CHECK_EQUAL(s20[++i20],  20); 
    CHECK_EQUAL(s20[++i20], 100); CHECK_EQUAL(s20[++i20], 110); CHECK_EQUAL(s20[++i20], 120); 

    const NDArray<int64_t>& s22 = slice(a3, {2,2});
    Index i22(s22.sizes());
    CHECK_EQUAL(s22[  i22],   2); CHECK_EQUAL(s22[++i22],  12); CHECK_EQUAL(s22[++i22],  22); 
    CHECK_EQUAL(s22[++i22], 102); CHECK_EQUAL(s22[++i22], 112); CHECK_EQUAL(s22[++i22], 122); 

    const NDArray<int64_t>& s24 = slice(a3, {2,4});
    Index i24(s24.sizes());
    CHECK_EQUAL(s24[  i24],   4); CHECK_EQUAL(s24[++i24],  14); CHECK_EQUAL(s24[++i24],  24); 
    CHECK_EQUAL(s24[++i24], 104); CHECK_EQUAL(s24[++i24], 114); CHECK_EQUAL(s24[++i24], 124); 
  }

}
