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
this could result in 18 billion 60-sample traces to transform.

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
efficient on small data sets, since the elements in the vector must inevitably
be combined, which leads to a significant number of scalar operations. A
vertical implementation can often use 100% pure SIMD instructions, since
the data from the individual datasets do not need to be mixed. In addition the
algorithm is substantially easier, since it is exactly the same as a scalar
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
available on MacPorts. To build the code generators try:

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
types ``R`` and ``E``, and abstract mathematical operations ``MUL``, ``SUB``,
``ADD``, ``NEG``, ``FMA``, ``FMS``, ``FNMA`` and ``FNMS``. Within FFTW this
allows the codelets to be used with single-precision, double-precision and
long-double floats; hence three versions of the library is usually available to
developers as ``-fftw3``, ``-fftw3f``, and ``-fftw3ld``. Each of these libraries
is compiled by setting ``E`` and ``R`` appropriately during the build.

It is this flexibility that allows the codelets to be trivially used with SIMD
vector types. For example the following macros or inlines are sufficient to
allow the code to be used with the AVX single-precision vector type ``__m256``
available as part of the [Intel intrinics API](https://software.intel.com/sites/landingpage/IntrinsicsGuide/).

````c++
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
  static const E name = { (val),(val),(val),(val),(val),(val),(val),(val) };

#define MAKE_VOLATILE_STRIDE(a,b) 0

#include "dft_c2c_60.c"
#include "dft_r2cf_60.c"
#include "dft_r2cb_60.c"

#endif // defined(__AVX__) and defined(__FMA__)
````

### License ###

__dft_simd__ is distributed under the [GPLv2](LICENSE.md) license, although
copyright on the machine-generated portions of the code seems [problematic
to assert](http://www.wipo.int/wipo_magazine/en/2017/05/article_0003.html).
