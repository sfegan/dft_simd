/*

   dft_simd/dft_simd.cpp -- Stephen Fegan -- 2018-02-19

   Test drive for FFTW speed tests and for SIMD genfft codelets

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole Polytechnique, CNRS/IN2P3

   This file is part of "dft_simd"

   "dft_simd" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "dft_simd" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#include <immintrin.h>
#include <fftw3.h>
#include <random>
#include <iostream>
#include <gtest/gtest.h>

constexpr int nvec = 8;
constexpr int nsamp = 60;
//constexpr int nsamp_align = (nsamp+nvec-1)/nvec * nvec;
constexpr int nloop = 8192*256;

TEST(TestDFT, FFTW_Aligned_One)
{
  float* xt = fftwf_alloc_real(nsamp);
  float* xf = fftwf_alloc_real(2*(nsamp/2 + 1));
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(nsamp, xt, (fftwf_complex*)xf, FFTW_DESTROY_INPUT|FFTW_MEASURE);
  fftwf_print_plan(plan);
  std::cout << '\n';
  for(int isamp=0;isamp<nsamp;isamp++) {
    xt[isamp] = gen(core);
  }
  for(int iloop=0; iloop<nloop*nvec; iloop++) {
    fftwf_execute(plan);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq] << ' ';
      }
      std::cout << '\n';
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(xf);
  fftwf_free(xt);
}

TEST(TestDFT, FFTW_MisAligned_One)
{
  float* xt = fftwf_alloc_real(nsamp+1) + 1;
  float* xf = fftwf_alloc_real(2*(nsamp/2 + 1)+1) + 1;
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(nsamp, xt, (fftwf_complex*)xf, FFTW_DESTROY_INPUT|FFTW_MEASURE);
  fftwf_print_plan(plan);
  std::cout << '\n';
  for(int isamp=0;isamp<nsamp;isamp++) {
    xt[isamp] = gen(core);
  }
  for(int iloop=0; iloop<nloop*nvec; iloop++) {
    fftwf_execute(plan);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq] << ' ';
      }
      std::cout << '\n';
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(xf-1);
  fftwf_free(xt-1);
}

TEST(TestDFT, FFTW_Aligned_Eight)
{
  float* xt = fftwf_alloc_real(nvec*nsamp);
  float* xf = fftwf_alloc_real(nvec*2*(nsamp/2 + 1));
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  int n = nsamp;
  fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &n, nvec,
                            (float*)xt, nullptr, 1, nsamp,
                            (fftwf_complex*)xf, nullptr, 1, nsamp/2 + 1,
                            FFTW_DESTROY_INPUT|FFTW_MEASURE);
  fftwf_print_plan(plan);
  std::cout << '\n';
  for(int isamp=0;isamp<nsamp*nvec;isamp++) {
    xt[isamp] = gen(core);
  }
  for(int iloop=0; iloop<nloop; iloop++) {
    fftwf_execute(plan);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq] << ' ';
      }
      std::cout << '\n';
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(xf);
  fftwf_free(xt);
}

TEST(TestDFT, FFTW_Aligned_TransposedEight)
{
  float* xt = fftwf_alloc_real(nvec*nsamp);
  float* xf = fftwf_alloc_real(nvec*2*(nsamp/2 + 1));
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  int n = nsamp;
  fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &n, nvec,
                            (float*)xt, nullptr, nvec, 1,
                            (fftwf_complex*)xf, nullptr, nvec, 1,
                            FFTW_DESTROY_INPUT|FFTW_MEASURE);
  fftwf_print_plan(plan);
  std::cout << '\n';
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp * nvec + ivec] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop; iloop++) {
    fftwf_execute(plan);
    if(iloop==0) {
      for(int ifreq=0;ifreq<(nsamp/2 + 1);ifreq++) {
        std::cout << xf[2*ifreq*nvec] << ' ' << xf[2*ifreq*nvec+1] << ' ';
      }
      std::cout << '\n';
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(xf);
  fftwf_free(xt);
}

#if 0 // This seems not to be any different to above
TEST(TestDFT, FFTW_SuperAligned_Eight)
{
  float* xt = fftwf_alloc_real(nvec*nsamp_align);
  float* xf = fftwf_alloc_real(nvec*nsamp_align);
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  int n = nsamp;
  fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &n, nvec,
                            (float*)xt, nullptr, 1, nsamp_align,
                            (fftwf_complex*)xf, nullptr, 1, nsamp_align/2,
                            FFTW_MEASURE);
  fftwf_print_plan(plan);
  std::cout << '\n';
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[ivec*nsamp_align+isamp] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop; iloop++) {
    fftwf_execute(plan);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq] << ' ';
      }
      std::cout << '\n';
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(xf);
  fftwf_free(xt);
}
#endif

#if defined(__AVX__)

using INT = int;
using stride = int;

inline __m256 ADD(const __m256& a, const __m256& b) { return _mm256_add_ps(a,b); }
inline __m256 SUB(const __m256& a, const __m256& b) { return _mm256_sub_ps(a,b); }
inline __m256 MUL(const __m256& a, const __m256& b) { return _mm256_mul_ps(a,b); }

//inline __m256 NEG(const __m256& a) { return _mm256_sub_ps(_mm256_setzero_ps(),a); }
inline __m256 NEG(const __m256& a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0)); }

#if defined(__FMA__)
inline __m256 FMA(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_fmadd_ps(a,b,c); }
inline __m256 FMS(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_fmsub_ps(a,b,c); }
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline __m256 FNMA(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_fnmsub_ps(a,b,c); }
inline __m256 FNMS(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_fnmadd_ps(a,b,c); }
#else
inline __m256 FMA(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_add_ps(_mm256_mul_ps(a,b),c); }
inline __m256 FMS(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_sub_ps(_mm256_mul_ps(a,b),c); }
inline __m256 FNMA(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_sub_ps(NEG(_mm256_mul_ps(a,b)),c); }
inline __m256 FNMS(const __m256& a, const __m256& b, const __m256& c) {
  return _mm256_add_ps(NEG(_mm256_mul_ps(a,b)),c); }
#endif

inline std::pair<__m256,__m256> ADD(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b) {
  return { ADD(a.first,b.first), ADD(a.second,b.second) }; }
inline std::pair<__m256,__m256> SUB(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b) {
  return { SUB(a.first,b.first), SUB(a.second,b.second) }; }
inline std::pair<__m256,__m256> MUL(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b) {
  return { MUL(a.first,b.first), MUL(a.second,b.second) }; }

inline std::pair<__m256,__m256> NEG(const std::pair<__m256,__m256>& a) {
  return { NEG(a.first), NEG(a.second) }; }

inline std::pair<__m256,__m256> FMA(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FMA(a.first,b.first,c.first), FMA(a.second,b.second,c.second) }; }
inline std::pair<__m256,__m256> FMS(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FMS(a.first,b.first,c.first), FMS(a.second,b.second,c.second) }; }
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline std::pair<__m256,__m256> FNMA(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FNMA(a.first,b.first,c.first), FNMA(a.second,b.second,c.second) }; }
inline std::pair<__m256,__m256> FNMS(const std::pair<__m256,__m256>& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FNMS(a.first,b.first,c.first), FNMS(a.second,b.second,c.second) }; }

inline std::pair<__m256,__m256> MUL(const __m256& a, const std::pair<__m256,__m256>& b) {
  return { MUL(a,b.first), MUL(a,b.second) }; }
inline std::pair<__m256,__m256> FMA(const __m256& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FMA(a,b.first,c.first), FMA(a,b.second,c.second) }; }
inline std::pair<__m256,__m256> FMS(const __m256& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FMS(a,b.first,c.first), FMS(a,b.second,c.second) }; }
inline std::pair<__m256,__m256> FNMA(const __m256& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FNMA(a,b.first,c.first), FNMA(a,b.second,c.second) }; }
inline std::pair<__m256,__m256> FNMS(const __m256& a, const std::pair<__m256,__m256>& b, const std::pair<__m256,__m256>& c) {
  return { FNMS(a,b.first,c.first), FNMS(a,b.second,c.second) }; }

#define DK(name, val) \
  static const __m256 name = { (val),(val),(val),(val),(val),(val),(val),(val) }

inline void MAKE_VOLATILE_STRIDE(int a, int b) { }

namespace m256 {

using E = __m256;
using R = __m256;

inline int WS(const stride s, const stride i) { return s*i; }

#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

} // namespace m256

TEST(TestDFT, AVX2)
{
  __m256* xt = nullptr;
  __m256* xf;
  ::posix_memalign((void**)&xt, 32, nvec*nsamp*sizeof(float));
  ::posix_memalign((void**)&xf, 32, nvec*2*(nsamp/2+1)*sizeof(float));
  for(unsigned i=0;i<2*(nsamp/2+1);i++)xf[i] = _mm256_setzero_ps();
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp][ivec] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop; iloop++) {
    m256::dft_codelet_r2cf_60(xt, xt+1, xf, xf+1, 2, 2, 2, 1, 0, 0);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq][0] << ' ';
      }
      std::cout << '\n';
    }
  }
  ::free(xf);
  ::free(xt);
}

namespace m256_FixedStride {

using E = __m256;
using R = __m256;

inline int WS(const stride s, const stride i) { return 2*i; }

#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

} // namespace m256

TEST(TestDFT, AVX2_FixedStride)
{
  __m256* xt = nullptr;
  __m256* xf;
  ::posix_memalign((void**)&xt, 32, nvec*nsamp*sizeof(float));
  ::posix_memalign((void**)&xf, 32, nvec*2*(nsamp/2+1)*sizeof(float));
  for(unsigned i=0;i<2*(nsamp/2+1);i++)xf[i] = _mm256_setzero_ps();
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp][ivec] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop; iloop++) {
    m256_FixedStride::dft_codelet_r2cf_60(xt, xt+1, xf, xf+1, 2, 2, 2, 1, 0, 0);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq][0] << ' ';
      }
      std::cout << '\n';
    }
  }
  ::free(xf);
  ::free(xt);
}

namespace m256_Unroll2 {

using E = std::pair<__m256,__m256>;
using R = std::pair<__m256,__m256>;
using INT = int;
using stride = int;

inline int WS(const stride s, const stride i) { return s*i; }

#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

} // namespace m256

TEST(TestDFT, AVX2_Unroll2)
{
  std::pair<__m256,__m256>* xt = nullptr;
  std::pair<__m256,__m256>* xf;
  ::posix_memalign((void**)&xt, 32, 2*nvec*nsamp*sizeof(float));
  ::posix_memalign((void**)&xf, 32, 2*nvec*2*(nsamp/2+1)*sizeof(float));
  for(unsigned i=0;i<2*(nsamp/2+1);i++)
    xf[i].first = xf[i].second = _mm256_setzero_ps();
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp].first[ivec] = gen(core);
    }
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp].second[ivec] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop/2; iloop++) {
    m256_Unroll2::dft_codelet_r2cf_60(xt, xt+1, xf, xf+1, 2, 2, 2, 1, 0, 0);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq].first[0] << ' ';
      }
      std::cout << '\n';
    }
  }
  ::free(xf);
  ::free(xt);
}

namespace m256_Unroll2_FixedStride {

using E = std::pair<__m256,__m256>;
using R = std::pair<__m256,__m256>;

inline int WS(const stride s, const stride i) { return 2*i; }

#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

} // namespace m256

TEST(TestDFT, AVX2_Unroll2_FixedStride)
{
  std::pair<__m256,__m256>* xt = nullptr;
  std::pair<__m256,__m256>* xf;
  ::posix_memalign((void**)&xt, 32, 2*nvec*nsamp*sizeof(float));
  ::posix_memalign((void**)&xf, 32, 2*nvec*2*(nsamp/2+1)*sizeof(float));
  for(unsigned i=0;i<2*(nsamp/2+1);i++)
    xf[i].first = xf[i].second = _mm256_setzero_ps();
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp].first[ivec] = gen(core);
    }
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp].second[ivec] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop/2; iloop++) {
    m256_Unroll2_FixedStride::dft_codelet_r2cf_60(xt, xt+1, xf, xf+1, 0, 0, 0, 1, 0, 0);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq].first[0] << ' ';
      }
      std::cout << '\n';
    }
  }
  ::free(xf);
  ::free(xt);
}

#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
