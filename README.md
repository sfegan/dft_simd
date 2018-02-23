## Efficient bulk transformation of short data samples using DFTs ##

__Adapting FFTW scalar codelets to SIMD data types and instructions to achieve
signifcantly improved DFT performance.__

_Stephen Fegan, 2018-02-19_

In many high-energy astronomy and particle physics experiments data consists of
relatively short digitized waveforms that need to be treated, i.e. filtered and
analyzed. Often this calls for transformation of the signal into frequency
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
times on my laptop, a MacBookPro running High Sierra on Intel(R) Core(TM)
i5-5287U CPU @ 2.90GHz, __which supports AVX2__, with FFTW installed through
MacPorts, are:

- Single DFT per _execute_, aligned datasets : __1968 ms__. Here the FFTW planner
  chose SSE-aware (SIMD) versions of the 5 and 12 sample codelets.

- Single DFT per _execute_, deliberately misaligned datasets : __3260 ms__.
  Here the planner could not choose the SSE codelets as the data were not
  properly aligned.

- Bulk DFT of 8 transforms per _execute_, aligned datasets : __2013 ms__. The
  planner used the SIMD codelets.

- Bulk DFT of 8 transforms per _execute_, aligned datasets, with the order of
  the axes transposed : __2688 ms__. Here the planner combined an SIMD
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
[ocaml-ocamlbuild](https://www.macports.org/ports.php?by=library&substr=ocaml-ocamlbuild)).
To build the code generators try:

````sh
ocamlbuild -classic-display -libs unix,nums gen_r2cf.native gen_r2cb.native gen_notw.ml
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
intrinics API](https://software.intel.com/sites/landingpage/IntrinsicsGuide/).

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

#include "dft_c2c_60.c"
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

- SIMD DFT of 8 transforms per call to 60-sample codelet : __400 ms__.

This code is therefore approximately __five__ times faster than the fastest
FFTW option, and more than __eight__ times faster than the slowest case. This
is probably not too surprising given that the slow FFTW plan used no SIMD
enhancements (and so a factor of 8 would be expected given the size of the
AVX vectors), and the fastest plan used some SSE SIMD enhancements, but could
not used 100% AVX SIMD instructions.

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
anticpated at compile time, as it requires the codelets be generated and
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

### License ###

__dft_simd__ is distributed under the [GPLv2](LICENSE.md) license, although
copyright on the machine-generated portions of the code seems [problematic to
assert](http://www.wipo.int/wipo_magazine/en/2017/05/article_0003.html). This
readme is distributed under the terms of the
[CC-BY-SA-4.0](data/LICENSE_CC_BY_SA.md) license.
