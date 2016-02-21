AMRNG: C++ AMP RNG Library

This is a library of Random Number Generarators that C++ AMP developers can freely use in their projects.

Adding new random generators:

    - amp_rand_collection is the base class for making distinct instances of an RNG available for use in parallel_for_each. This means each thread get a distinct RNG object. 
      The instances can be initialized such that different threads do not receive the same random numbers.

    - Add a new RNG class in its own header file, and make sure it #include's the "amp_rand_collection.h" header. It is suggested to follow the pattern exemplified by the tinymt RNG.

    - Add a new collection that inherits from amp_rand_collection. Implement the constructor to initialize the thread specific instances of your random number generator.

    - inc\amp_tinymt_rng.h is a random number generator that implements 'TinyMT', a variant of Mersenne twister tailored to work on accelerators

    - inc\amp_sobol_rng.h is another random number generator that implements 'Quasi Sobol Sequence Generator' up to 21201 dimensions.


Using the library in your app:

    - #include the desired RNG header file:  "inc\amp_tinymt_rng.h" or "inc\amp_sobol_rng.h".

    - Create a generator collection (pick the amp_rand_collection type corresponding to the random number generator of your choice)

    - Initialize the generator collection. This can be done either at the time of creation (provide a seed) or inside the parallel_for_each (by calling initialize)

    - Use the generator in parallel_for_each. Call next_xxx method on the thread specific instance to get the next random number.

    - test\main.cpp shows different ways of using the tinymt random number generator and Sobol sequence generator in a parallel_for_each
	  test\{tinymt_data.txt, sobol_data.txt} contain the output of running the sample with 10K random number sequences.
	  test\amprng.xlsm is the spreedsheet template displaying the Sobol sequence and tinyMT random number sequence. 


Hardware requirement:

    - This sample requires DirectX 11 capable card, if none detected sample will use DirectX 11 Emulator.

Software requirement:

    - Install Visual Studio 2012 from http://msdn.microsoft.com

