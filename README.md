# dft_simd #

### Efficient bulk transformation of short data samples using DFTs ###

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

This will run four test cases, each of which transforms 2**24 (16.8 million)
waveforms. The first three test cases use FFTW, the final one uses a custom SIMD
transform described below. The four FFTW test cases with their running times on
my laptop, a MacBookPro running High Sierra on Intel(R) Core(TM) i5-5287U CPU @
2.90GHz, __which supports AVX2__, with FFTW installed through MacPorts, are:

- Single DFT per _execute_, aligned datasets : __1968 ms__. Here the FFTW planner
  chose SSE-aware (SIMD) versions of the 5 and 12 sample codelets.

- Single DFT per _execute_, deliberately misaligned datasets : __3327 ms__.
  Here the planner could not choose the SSE codelets as the data were unaligned.

- Bulk DFT of 8 transforms per _execute_, aligned datasets : __2124 ms__. The
  planner used the SIMD codelets.

It can be seen that the SIMD plans provide a relatively large improvement in
execution speed, as expected, but that they do not come close to providing the
factor of 8 improvement that one might expect given that the AVX vector
registers are 256 bits wide, i.e. that they can process 8 floats at once. Also,
note that the SIMD improves even DFT speed even in the simple
single-transform-per-execution case. No additional improvement was seen in the
8-transforms-at-once case, even though FFTW could have theoretically achieved
this.

In essence this shows is that FFTW uses SIMD instructions to manipulate the data
__horizontally__, combining data from a single transform into vectors  where
possible to execute SIMD instruction. It does not however combine data
__vertically__, i.e. it does not combine samples from different transforms into
a vector and manipulate them, although that is the __classic__ SIMD approach,
used since the days of the old CDC and Cray mainframe vector processors.





### License ###

__dft_simd__ is distributed under the [GPLv2](LICENSE.md) license.
