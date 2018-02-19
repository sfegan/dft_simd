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

using E = __m256;
using R = __m256;
using INT = int;
using stride = int;

inline int WS(const stride s, const stride i) { return s*i; }

inline E ADD(const E a, const E b) { return _mm256_add_ps(a,b); }
inline E SUB(const E a, const E b) { return _mm256_sub_ps(a,b); }
inline E MUL(const E a, const E b) { return _mm256_mul_ps(a,b); }

//inline E NEG(const E a) { return _mm256_sub_ps(_mm256_setzero_ps(),a); }
inline E NEG(const E a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.0)); }

inline E FMA(const E a, const E b, const E c) { return _mm256_fmadd_ps(a,b,c); }
inline E FMS(const E a, const E b, const E c) { return _mm256_fmsub_ps(a,b,c); }
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline E FNMA(const E a, const E b, const E c) { return _mm256_fnmsub_ps(a,b,c); }
inline E FNMS(const E a, const E b, const E c) { return _mm256_fnmadd_ps(a,b,c); }

#define DK(name, val) \
  static const E name = { (val),(val),(val),(val),(val),(val),(val),(val) };

#define MAKE_VOLATILE_STRIDE(a,b) 0

#include "dft_c2c_60.c"
#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

constexpr int nvec = 8;
constexpr int nsamp = 60;
constexpr int nsamp_align = (nsamp+nvec-1)/nvec * nvec;
constexpr int nloop = 8192*256;

TEST(TestDFT, FFTW_Aligned_One)
{
  float* xt = fftwf_alloc_real(nsamp);
  float* xf = fftwf_alloc_real(2*(nsamp/2 + 1));
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(nsamp, xt, (fftwf_complex*)xf, FFTW_MEASURE);
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
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(nsamp, xt, (fftwf_complex*)xf, FFTW_MEASURE);
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
                            FFTW_MEASURE);
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

TEST(TestDFT, AVX2)
{
  __m256* xt = (__m256*)fftwf_malloc(nvec*nsamp*4);
  __m256* xf = (__m256*)fftwf_malloc(nvec*2*(nsamp/2+1)*4);
  for(unsigned i=0;i<2*(nsamp/2+1);i++)xf[i] = _mm256_setzero_ps();
  std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0,1.0);
  for(int ivec=0;ivec<nvec;ivec++) {
    for(int isamp=0;isamp<nsamp;isamp++) {
      xt[isamp][ivec] = gen(core);
    }
  }
  for(int iloop=0; iloop<nloop; iloop++) {
    dft_codelet_r2cf_60(xt, xt+1, xf, xf+1, 2, 2, 2, 1, 30, 31);
    if(iloop==0) {
      for(int ifreq=0;ifreq<2*(nsamp/2 + 1);ifreq++) {
        std::cout << xf[ifreq][0] << ' ';
      }
      std::cout << '\n';
    }
  }
  fftwf_free(xf);
  fftwf_free(xt);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
