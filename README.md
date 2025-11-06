## Efficient bulk transformation of short data samples using DFTs ##

__Adapting FFTW scalar codelets to CPU SIMD data types and instructions or to
GPUs uing OpenCL to achieve significantly improved DFT performance.__

_Stephen Fegan, LLR/Ecole Polytechnique, 2018-02-19_

## Update 2025

Added test of 1024 point codelet with AVX2 and AVX512. The
codelet is much more complex than the 60 point case tested originally, taking 
significantly longer to generate using the CAML codelet generator (which must 
only be done once), and to compile. In my test it runs more than twice as fast 
as using the FFTW API.

Run times on AMD EPYC 9474F 48-Core processor

| Test case                       | 60 sample AVX2  | 1024 sample AVX2 | 1024 sample AVX512 |
| ------------------------------- | --------------- | ---------------- | ------------------ |
| FFTW_Aligned_One                |          858 ms |        12,136 ms |          24,657 ms |
| FFTW_MisAligned_One             |        1,482 ms |        36,244 ms |          68,664 ms |
| FFTW_Aligned_Eight              |          859 ms |        13,637 ms |          27,441 ms |
| FFTW_Aligned_TransposedEight    |        1,083 ms |        24,113 ms |          59,545 ms |
| CODELET_AVX                     |          149 ms |         9,124 ms |          12,628 ms |
| CODELET_AVX_FixedStride         |           79 ms |         7,270 ms |          10,610 ms |
| CODELET_AVX_Unroll2             |          145 ms |         9,400 ms |          14,553 ms |
| CODELET_AVX_Unroll2_FixedStride |           93 ms |         8,167 ms |          13,044 ms |

## Original description

In many high-energy astronomy and particle physics experiments data consists of
relatively short digitised waveforms that need to be treated, i.e. filtered and
analysed. Often this calls for transformation of the signal into frequency
domain, and hence starts with a discrete Fourier transform (DFT), possibly
followed by filtering and an inverse transform. Or perhaps the DFT is used to
generate the power-spectrum of the signal for diagnostic purposes.

In applications in imaging atmospheric Cherenkov astronomy a dataset may consist
of thousands of events recorded per second, each with as many as a few thousand
channels recording 60ns of data at a sampling rate of 1GHz. In a half-hour run
this could result in 18 billion 60-sample datasets to transform.

There is clearly therefore a significant desire that the DFT implementation be
as efficient as possible, i.e. that it fully exploit the resources of the CPU.

For general purposes the [FFTW library](http://fftw.org/) provides an
world-class DFT implementation; in fact the authors somewhat jokingly claim that
FFTW is an acronym for the _Fastest Fourier Transform in the West_. FFTW uses a
two-step process to implement DFTs. Transforms must be __planned__ before they
can be __executed__. Internally to FFTW transforms are implemented by
__codelets__ which operate on small data arrays. During planning these are
combined by the FFTW library in an optimal divide-and-conquer approach to
achieve the full transform. For example the library has custom codelets that
transform 5 and 12 elements and these can be combined by the library to
implement a transform of 60 samples. Once the transform is planned it can be
executed any number of times. FFTW provides an advanced interface to implement
bulk transforms without need to call the library each time, presumably speeding
up execution. It also supports SIMD (vector) processing instruction sets such as
SSE and AVX supported by all modern Intel CPUs. There is some hope that
therefore that this library could be used to efficiently bulk-transform
waveforms from IACT applications.

This repository consists of a simple set of test cases the use FFTW to transform
a waveform of 60 single-precision floats to Fourier space. They can be accessed
by building and running the __dft_simd__ test using CMake, e.g.

````sh
mkdir mybuild
cd mybuild
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
./dft_simd
````

This will run five test cases, each of which transforms 2**24 (16.8 million)
waveforms. The first four test cases use FFTW, the final one uses a custom SIMD
transform described below. The four FFTW test cases, along with their running
times on my laptop, a MacBookPro running High Sierra on [``Intel(R) Core(TM)
i5-5287U CPU @
2.90GHz``](https://ark.intel.com/products/84988/Intel-Core-i5-5287U-Processor-3M-Cache-up-to-3_30-GHz),
__which supports AVX2 and FMA__, with FFTW installed through MacPorts, are:

- Single DFT per _execute_, aligned datasets : __1925 ms__. Here the FFTW planner
  chose SSE-aware (SIMD) versions of the 5 and 12 sample codelets.

- Single DFT per _execute_, deliberately misaligned datasets : __3135 ms__.
  Here the planner could not choose the SSE codelets as the data were not
  properly aligned.

- Bulk DFT of 8 transforms per _execute_, aligned datasets : __1912 ms__. The
  planner used the SIMD codelets.

- Bulk DFT of 8 transforms per _execute_, aligned datasets, with the order of
  the axes transposed : __2563 ms__. Here the planner combined an SIMD
  version of the 12 sample codelet but a scalar version of the 5 sample codelet.

It can be seen that the SIMD plans provide a relatively large improvement in
execution speed, as expected, but that they do not come close to providing the
factor of 8 improvement that one might expect given that the AVX vector
registers are 256 bits wide, i.e. that they can process 8 floats at once. Also,
note that the SIMD improves the DFT speed even in the simple
single-transform-per-execution case. No additional improvement was seen in the
8-transforms-at-once case, even though FFTW could have theoretically achieved
this.

I think that in essence this shows is that FFTW uses SIMD instructions to
manipulate the data __horizontally__, combining elements from a single transform
dataset into vectors where possible to execute SIMD instruction, reducing the
total number of instructions to execute in the transform. It does not however
seem to combine data __vertically__, i.e. it does not combine samples from
different datasets into a vector and manipulate them, although that is the
__classic__ SIMD approach. Horizontal SIMD configurations tend not to be as
efficient on small data sets, since there may not be enough elements in the
dataset to efficiently fill the vectors every time and the SIMD vectors will
inevitably need to be combined, which leads to a significant number of scalar
operations. A vertical implementation can often use 100% pure SIMD instructions,
since the data from the individual datasets never need to be mixed. In addition
the algorithm is substantially easier, since it is exactly the same as a scalar
algorithm, but with a vector data type, i.e. it is almost as easy as replacing
``float`` with ``__m256`` to achieve an 8-fold increase in speed.

To evaluate the vertical approach we would need an FFT library that accepts and
operates on SIMD data types. One such possibility seems to be [KISS
FFT](https://github.com/mborgerding/kissfft), although I did not test it. Here I
exploit the good design of the FFTW algorithm. As discussed above this library
is built upon codelets that implement the small transforms. The codelets are
themselves machine-generated, using a package, ``genfft``, that is [distributed
with FFTW3](https://github.com/FFTW/fftw3/tree/master/genfft). ``genfft``
consists of a series of code generators written in [CAML](http://ocaml.org/)
that produce codelets of various types for any desired number of samples. The
are generators for scalar and SIMD transforms of various types (complex DFT,
real DFT, DCT etc..), generators that produce "twiddle factors" to allow the
output of codelets to be combined into larger transforms, and others. Of these
the three generators of primary interest here are:

- ``gen_r2cf.ml`` : produce scalar REAL to COMPLEX (_forward_) transform
- ``gen_r2cb.ml`` : produce scalar COMPLEX to REAL (_backward_) transform
- ``gen_notw.ml`` : produce scalar COMPLEX to COMPLEX transform

The generators can be compiled using the ``ocmlbuild`` compiler that is
available on MacPorts (package
[ocaml-ocamlbuild](https://www.macports.org/ports.php?by=library&substr=ocaml-ocamlbuild)
and ocaml-num).
To build the code generators try:

````sh
ocamlbuild -classic-display -libs unix,nums gen_r2cf.native gen_r2cb.native gen_notw.ml
````
Revisiting this in 2025 I need to do:

````sh
ocamlbuild -use-ocamlfind -pkg unix -pkg num gen_r2cf.native gen_r2cb.native gen_twiddle.native gen_notw.native gen_hc2hc.native
````


As a first example of using the generators we can build a codelet for a 3-sample
real-to-complex (forward) DFT by running:

````sh
N=3; ./gen_r2cf.native -n ${N} -standalone -fma -generic-arith \
  -compact -name dft_codelet_r2cf_${N} > dft_r2cf_${N}.c
````

The key options are:

- ``-n`` : specifies the DFT size,
- ``-standalone`` : instructs the generator to produce only the codelet function,
  and not the support functionality to allow the codelet be registered with FFTW,
- ``-fma`` : allows the generator to use fused multiply and add instructions,
- ``-generic-arith`` : instructs the generator to use function-style arithmetic
  rather than operators, for example ``a=b+c`` will be generated as ``a=MUL(b,c)``,
- ``-name`` : specifies the name of the generated codelet function.

This will produce the following C code (which I have run through indent) :

````c
/*
 * This function contains 4 FP additions, 2 FP multiplications,
 * (or, 3 additions, 1 multiplications, 1 fused multiply/add),
 * 7 stack variables, 2 constants, and 6 memory accesses
 */
void dft_codelet_r2cf_3(R * R0, R * R1, R * Cr, R * Ci,
  stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
  DK(KP866025403, +0.866025403784438646763723170752936183471402627);
  DK(KP500000000, +0.500000000000000000000000000000000000000000000);
  {
    INT i;
    for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs,
        Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(12, rs),
        MAKE_VOLATILE_STRIDE(12, csr), MAKE_VOLATILE_STRIDE(12, csi)) {
      E T1, T2, T3, T4;
      T1 = R0[0];
      T2 = R1[0];
      T3 = R0[WS(rs, 1)];
      T4 = ADD(T2, T3);
      Cr[WS(csr, 1)] = FNMS(KP500000000, T4, T1);
      Ci[WS(csi, 1)] = MUL(KP866025403, SUB(T3, T2));
      Cr[0] = ADD(T1, T4);
    }
  }
}
````

As can be seen the algorithm of the codelet is expressed in terms of abstract
types ``R`` and ``E``, abstract mathematical operations ``MUL``, ``SUB``,
``ADD``, ``NEG``, ``FMA``, ``FMS``, ``FNMA`` and ``FNMS``, and some helper
functions & macros ``DK``, ``WS``, and ``MAKE_VOLATILE_STRIDE``. Within FFTW
this allows the codelets to be used with single-precision, double-precision and
long-double floats; hence three versions of the library is usually available to
developers as ``-fftw3``, ``-fftw3f``, and ``-fftw3ld``. Each of these libraries
is compiled by setting ``E`` and ``R`` appropriately during the build.

It is this flexibility that allows the codelets to be trivially used with SIMD
vector types. For example the following macros or inlines are sufficient to
allow the code to be used to transform eight datasets packed into the AVX
single-precision vector type ``__m256`` available as part of the [Intel
intrinsics API](https://software.intel.com/sites/landingpage/IntrinsicsGuide/).

````cpp
#if defined(__AVX__) and defined(__FMA__)

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
  static const E name = { (val),(val),(val),(val),(val),(val),(val),(val) }

inline void MAKE_VOLATILE_STRIDE(int a, int b) { }

#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

#endif // defined(__AVX__) and defined(__FMA__)
````

The types and functions could be changed to operate on double-precision AVX
vectors (``__m256d``) and presumably the single-precision and double-precision
AVX-512 vector types also. Note the somewhat confusing difference between the
FFTW and Intel definitions of the fused negative multiply and add/subtract
instructions that requires ``FNMA`` be mapped to ``_mm256_fnmsub_ps`` and
``FNMS`` be mapped to ``_mm256_fnmadd_ps``.

The inputs to the ``r2cf`` codelet are:

- Two input ``R`` arrays with the even (``R0``) and odd elements (``R1``) of the
  real (vector) datasets; see for example Numerical Recipes in C++ section
  12.3.2 for discussion. If N is even then each of these should have ``N/2``
  accessible elements, each starting at index 0 (but see discussion of stride
  below). If N is odd then the even array should have ``(N+1)/2`` accessible
  elements, and the odd ``(N-1)/2``, each starting at index 0.

- Two output ``R`` arrays into which the codelet will write the real (``Cr``)
  and imaginary (``Ci``) components of the (vector) transform. If N is even
  the real array should have ``N/2+1`` addressable elements, starting at zero
  (i.e. ``[0,1,...,N/2]``) while the imaginary array should have ``N/2-1``
  elements starting at one (i.e. ``[1,2,...,N/2-1]``). For N odd the real
  array should have ``(N+1)/2`` addressable elements starting at zero
  (i.e. ``[0,1,...,(N-1)/2]``), while the imaginary array should have
  ``(N-1)/2`` addressable elements, starting at one (i.e. ``[1,2,...,(N-1)/2]``).
  In all cases ``Ci[0]`` is untouched by the code, and the user should set it
  to zero if desired. The same is true for ``Ci[N/2]`` in the case when N is
  even.

- Three input array stride specifiers, ``rs``, ``csr``, and ``csi``. These give
  considerable flexibility to the user to lay the arrays out as desired. As
  an example it may be desirable to simply have all the samples in order in one
  array of size ``N``, and return the DFT in an array of size ``2*(N/2+1)``
  with the [_standard
  layout_](http://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format)
  of complex numbers where each real element is followed by each complex
  element. This can be achieved with something like this:

    ````cpp
    constexpr int N = 60;
    __m256 xt[N]; // input data
    __m256 xf[2*(N/2+1)]; // output array
    __m256* R0 = xt;
    __m256* R1 = xt+1;
    __m256* Cr = xf;
    __m256* Ci = xf+1;
    int rs = 2;
    int csr = 2;
    int csi = 2;
    // call the codelet and zero unwritten imaginary components if desired
    dft_codelet_r2cf_60(R0, R1, Cr, Ci, rs, csr, csi, ... see below ... );
    Ci[0] = Ci[N/2 * 2 /* must remember stride ! */] = _mm256_setzero_ps();
    ````

  The FFTW [_half-complex_
  format](http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html#The-Halfcomplex_002dformat-DFT)
  (where all the real DFT amplitudes come first ordered by increasing frequency
  starting with DC, followed by the imaginary amplitudes in order of decreasing
  frequency) can also be achieved using a negative stride on the imaginary
  array, with something like this:

    ````cpp
    constexpr int N = 60;
    __m256 xt[N]; // input data
    __m256 xf[N]; // output array
    __m256* R0 = xt;
    __m256* R1 = xt+1;
    __m256* Cr = xf;
    __m256* Ci = xf+N;
    int rs = 2;
    int csr = 1;
    int csi = -1;
    // call the codelet
    dft_codelet_r2cf_60(R0, R1, Cr, Ci, rs, csr, csi, ... see below ... );
    ````

- The final three parameters allow the codelet to perform multiple transforms
  in a loop in one single call. The ``v`` option specifies how many transforms
  to perform, it should be at least one if the code is to do anything at all.
  The ``ivs`` and ``ovs`` parameters specify the _stride_ of the arrays of
  input and output data if multiple data sets are to be transformed (the "v"
  stands for vector in the parlance of FFTW, but that is a confusing term here
  as we talk about SIMD vectors, so I try to avoid it). The simplest is to set
  ``v=1`` in which case ``ivs`` and ``ovs`` are irrelevant.

To generate the codelet for the inverse DFT the generator ``gen_r2cb.native``
should be used, with the option ``-sign 1`` (otherwise the order of the
output will be reversed). For example to generate the N=3 case use:

````sh
N=3; ./gen_r2cb.native -n ${N} -sign 1 -standalone -fma -generic-arith \
  -compact -name dft_codelet_r2cb_${N} > dft_r2cb_${N}.c
````

The ``dft_simd`` test code in this repository uses the 60-element forward
real-to-complex DFT codelet to compare with the calls to FFTW in the
configurations described above. It performs the same number of transforms
(16.7 million) in batches of eight by packing the data into an array of
SIMD vectors of type ``__m256``. The results are:

- SIMD DFT of 8 transforms per call to 60-sample codelet : __379 ms__.

This code is therefore approximately __five__ times faster than the fastest
FFTW option, and more than __eight__ times faster than the slowest case. This
is probably not too surprising given that the slow FFTW plan used no SIMD
enhancements (and so a factor of 8 would be expected given the size of the
AVX vectors), and the fastest plan used some SSE SIMD enhancements, but could
not used 100% AVX SIMD instructions.

### Fixed Stride ###

Examining the assembly version of the test code suite (produced ``Apple LLVM
version 9.0.0 (clang-900.0.39.2)`` on MacOS High Sierra) shows that a
significant number of instructions are generated at the start of the codelet
functions to handle the variable real and complex strides that can be specified.
Since any value of these strides can be specified at runtime the memory
locations for the elements in input and output data arrays must be calculated
through a series of scalar integer multiplies, adds and shifts. If the strides
are known at compile time then these runtime calculations are not needed, and
represent a relatively large waste of computation time: the vector pipelines
cannot start to process the DFTs until the scalar pipelines complete the
computation of the array access memory locations. In the first example we showed
above the real and complex strides are fixed at ``rs=csr=csi=2``, and we
can modify the ``WS`` helper function to assume this at compile time by
changing:

````cpp
inline int WS(const stride s, const stride i) { return 2*i; }
````

This change hardcodes the stride and moots the values  ``rs``, ``csr`` and
``csi``. This single one-line change to the code approximately doubles the speed
of the test suite on my laptop:

- SIMD DFT of 8 transforms per call to 60-sample codelet with fixed stride: __207 ms__.

Of course this comes at the cost of reduced runtime flexibility. However if the
flexibility of changing strides is not needed, then the tradeoff would seem
worth making. Of course, fixed and variable stride versions could easily be
compiled into the same code a selected at runtime, giving the best of both
worlds.

### Loop unrolling ###

Further improvement in performance can potentially be achieved by using the
standard approach of loop unrolling. Generally loop unrolling techniques reduce
the number of iterations in a loop by calculating multiple quantities together
in the loop body. This can improve performance by allowing more registers to be
used (if a single loop calculation does not use the fill set of registers) and
increasing the usage of the CPU pipelines (if a single loop calculation does
not completely fill the pipelines).

Loop unrolling can be applied to the short DFTs codelets by processing multiple
vector datasets at once. One approach, outlined here, is to change the data type
from a single vector ``__m256`` to a structure containing multiple vectors, e.g.
``std::pair<__m256,__m256>``, changing the abstract mathematical operations
to match. The following method is used in the test code in this repository,

````cpp
#if defined(__AVX__) and defined(__FMA__)
using E = std::pair<__m256,__m256>;
using R = std::pair<__m256,__m256>;
using INT = int;
using stride = int;

inline int WS(const stride s, const stride i) { return s*i; }

inline E ADD(const E a, const E b) {  
  return { _mm256_add_ps(a.first,b.first), _mm256_add_ps(a.second,b.second) }; }
inline E SUB(const E a, const E b) {
  return { _mm256_sub_ps(a.first,b.first), _mm256_sub_ps(a.second,b.second) }; }
inline E MUL(const E a, const E b) {
  return { _mm256_mul_ps(a.first,b.first), _mm256_mul_ps(a.second,b.second) }; }

inline E NEG(const E a) { return { _mm256_xor_ps(a.first, _mm256_set1_ps(-0.0)),
    _mm256_xor_ps(a.second, _mm256_set1_ps(-0.0)) }; }

inline E FMA(const E a, const E b, const E c) {
  return { _mm256_fmadd_ps(a.first,b.first,c.first), _mm256_fmadd_ps(a.second,b.second,c.second) }; }
inline E FMS(const E a, const E b, const E c) {
  return { _mm256_fmsub_ps(a.first,b.first,c.first), _mm256_fmsub_ps(a.second,b.second,c.second) }; }
inline E FNMA(const E a, const E b, const E c) {
  return { _mm256_fnmsub_ps(a.first,b.first,c.first),
    _mm256_fnmsub_ps(a.second,b.second,c.second) }; }
inline E FNMS(const E a, const E b, const E c) {
  return { _mm256_fnmadd_ps(a.first,b.first,c.first),
    _mm256_fnmadd_ps(a.second,b.second,c.second) }; }

inline E MUL(const __m256 a, const E b) {
  return { _mm256_mul_ps(a,b.first), _mm256_mul_ps(a,b.second) }; }
inline E FMA(const __m256 a, const E b, const E c) {
  return { _mm256_fmadd_ps(a,b.first,c.first),
    _mm256_fmadd_ps(a,b.second,c.second) }; }
inline E FMS(const __m256 a, const E b, const E c) {
  return { _mm256_fmsub_ps(a,b.first,c.first),
    _mm256_fmsub_ps(a,b.second,c.second) }; }
inline E FNMA(const __m256 a, const E b, const E c) {
  return { _mm256_fnmsub_ps(a,b.first,c.first),
    _mm256_fnmsub_ps(a,b.second,c.second) }; }
inline E FNMS(const __m256 a, const E b, const E c) {
  return { _mm256_fnmadd_ps(a,b.first,c.first),
    _mm256_fnmadd_ps(a,b.second,c.second) }; }

#define DK(name, val) \
  static const __m256 name = { (val),(val),(val),(val),(val),(val),(val),(val) }

inline void MAKE_VOLATILE_STRIDE(int a, int b) { }

#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"
#endif // defined(__AVX__) and defined(__FMA__)
````

The main difference to the non-unrolled code is that two sets of multiply and
FMA functions are needed, one to handle cases where pairs of vectors are
multiplied, and a second to handle multiplication by a (vector) constant.

Applying this code to the same test case in which 16.8 million DFTs are
calculated in 1.05 million calls to the codelet results in the running time:

- SIMD DFT of 16 transforms in 2 AVX vectors per call to 60-sample codelet : __301 ms__.

giving a roughly 20% improvement over the non-unrolled variable-stride version.
A fixed-stride version of this unrolled code runs in:

- SIMD DFT of 16 transforms in 2 AVX vectors per call to 60-sample codelet with fixed stride : __227 ms__.

which is slower than the non-unrolled version with fixed stride above. This
shows that unrolling helps keep the vector pipelines filled while the CPU waits
for the array addresses to be calculated, but that it is not advantageous in the
case where array addresses are know at compile time and where the vector
pipelines are already basically full. In this case in fact there is a penalty,
which may be related to register over-use.

### Results on Intel Xeon (Broadwell) ###

All the results given thus far were from running the tests on my laptop, an
Intel Core (Broadwell) as described at the top of the page. Here we present the
results from the eight test cases transforming 16.8 million 60-sample datasets
on an [``Intel(R) Xeon(R) CPU E5-2650
v4``](https://ark.intel.com/products/91767/Intel-Xeon-Processor-E5-2650-v4-30M-Cache-2_20-GHz)
which supports ``AVX2`` and ``FMA``. The system runs ``Linux 3.10.0-693.11.6.el7.x86_64``
with compiler ``gcc version 4.8.5 20150623 (Red Hat 4.8.5-16)``.

- FFTW3 1 waveform/call, aligned : __2194 ms__
- FFTW3 1 waveform/call, un-aligned : __3643 ms__
- FFTW3 8 waveforms/call, aligned : __2264 ms__
- FFTW3 8 waveforms/call, aligned and transposed : __2961 ms__
- AVX codelet, 8 waveform SIMD vector : __220 ms__
- AVX codelet, 8 waveform SIMD vector, fixed stride : __229 ms__
- AVX codelet, 8 waveform SIMD vector, unrolled 16 waveforms/call : __296 ms__
- AVX codelet, 8 waveform SIMD vector, unrolled 16 waveforms/call, fixed stride : __297 ms__

On this system the AVX codelet improves performance by a factor of 10 over the
best case of FFTW3. Here there is no improvement in speed from using a fixed
stride code, nor from loop unrolling.

### Running DFT codelets on GPUs with OpenCL ###

Since the FFTW codelet is just a simple C function it can also be relatively
easily adapted to GPUs by integrating into an OpenCL kernel. The snippet below
shows an extremely naive OpenCL test program that is used here to evaluate the
performance of the codelet under OpenCL. The program is very little more than a
few defines and a trivial kernel to call the codelet. The driver code that runs
this on OpenCL under MacOS can be seen in the test case of ``dft_simd.cpp``.

````c
#define E float
#define R __global float
#define INT int
#define stride int
#define ADD(a,b) ((a)+(b))
#define SUB(a,b) ((a)-(b))
#define MUL(a,b) ((a)*(b))
#define NEG(a) (-(a))
#define FMA(a,b,c) fma((a),(b),(c))
#define FMS(a,b,c) fma((a),(b),-(c))
#define FNMA(a,b,c) (-fma((a),(b),(c)))
#define FNMS(a,b,c) fma(-(a),(b),(c))
#define WS(s,i) (2*(i))
#define DK(name, val) const float name = val
#define MAKE_VOLATILE_STRIDE(a, b) 1
#include "dft_r2cf_60.c"
void __kernel simple_dft(__global float* xt, __global float* xf) {
  dft_codelet_r2cf_60(xt, xt+1, xf, xf+1, 0, 0, 0, 1, 0, 0);
}
````

Running this test code on an Early 2015-era MacBook Pro equipped with an on-CPU
``Intel(R) Iris(TM) Graphics 6100`` GPU, which supports 48 compute units results
in the following execution time:

- Enqueuing of 16.8M calls to 60-sample DFT codelet kernel with fixed
stride : __835 ms__ (of which 130 ms is run-time compilation of the GPU code
and would likely be amortised in bulk use of the program).

This is faster than the native FFTW implementation, but slower than the SIMD
codelets running on the CPU. It shows that running the codelets on GPUs may
provide a useful tool to perform bulk DFTs. On a HPC-grade GPU with thousands of
compute cores there is reason to believe this approach may significantly
out-perform the generic CPU. I hope to have access to such a machine in the
future.

### Conclusion ###

For cases where a large number of small DFTs must be performed the approach used
by FFTW does not seem to be optimal. This may be because FFTW is combining data
elements from the same dataset into SIMD vectors (horizontal approach), rather
than composing SIMD vectors from elements from different datasets and
calculating multiple transforms at once (vertical approach).

However the flexible design of the FFTW codelets, in particular their use of
abstract data types and mathematical operations, allows them to be used directly
with SIMD data vector types. Therefore a _scalar_ codelet can be trivially
transformed into a _vector_ codelet to allow efficient SIMD DFTs to be
calculated.

This approach is only really feasible for small transformation sizes that can be
anticipated at compile time, as it requires the codelets be generated and
compiled in advance. Codelets for long transformation sizes are impractical as
they grow in size by Nlog(N), and will relatively quickly exceed the cache size
of the CPU, after which their execution will slow. Long transforms should be
broken down into loops over the prime factors, meaning that something like the
full machinery of FFTW to plan and combine results from different codelets would
need to be reimplemented for the vector types.

In principle FFTW itself could provide a full library that works with SIMD
vector types (like they provide for float, double, long double), but in practice
it would probably be more useful to the user for them to handle the case of
multiple DFTs requested through the _advanced_ interface differently, using
vertical codelets internally.

The conclusions on the tradeoffs of using fixed stride and/or loop unrolling may
depend strongly on the type of system used, for example an AVX-512 system which
has more vector registers, or on a system with more scalar or vector pipelines
than my laptop. It may also strong depend on the compiler, and also also on the
size of the DFT (I only tested N=60). For example, shorter DFT sizes, such as
N=16, may not use the  pipelines/registers as efficiently as N=60 and hence
unrolling may be beneficial. The particular case of interest to the user should
be studied. The primary conclusion that the SIMD codelets outperforms FFTW by
a relatively large factor for short DFTs seems to be robust.

### License ###

__dft_simd__ is distributed under the [GPLv2](LICENSE) license, although
copyright on the machine-generated portions of the code seems [problematic to
assert](http://www.wipo.int/wipo_magazine/en/2017/05/article_0003.html). This
readme is distributed under the terms of the
[CC-BY-SA-4.0](LICENSE_CC_BY_SA.md) license.
