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
consists of a series of code generators written in CAML that produce codelets
of various types for any desired number of samples. The three generators of
primary interest here are:

- ``gen_r2cf.ml`` : produce REAL to COMPLEX, _forward_, transform
- ``gen_r2cb.ml`` : produce COMPLEX to REAL, _backward_, transform
- ``gen_notw.ml`` : produce COMPLEX to COMPLEX transform




### License ###

__dft_simd__ is distributed under the [GPLv2](LICENSE.md) license, although
copyright on the machine-generated portions of the code seems [problematic
to assert](http://www.wipo.int/wipo_magazine/en/2017/05/article_0003.html).
