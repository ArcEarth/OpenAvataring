AMPFFT: C++ AMP FFT Library

This library is for the Visual Studio 2012.  It is built around the Direct3D FFT API's and can
perform forward and backwards transforms.

This drop is organized as follows:

   inc                       Headers for C++ FFT
   x64\lib                   64-bit dynamic libs, release and debug
   x64\bin                   64-bit DLL and PDB, release and debug
   x86\lib                   32-bit dynamic libs, release and debug
   x86\bin                   32-bit DLL and PDB, release and debug

To compile:
   1. Add "inc" to your INCLUDE path.
   2. Add "x64\lib" or "x86\lib" to your LIB path.
   3. Add "x64\bin" or "x86\bin" to your PATH.
   4. Choose which lib to link against (see below).
   4. cl -EHs -c foo.cpp  <library>.lib

You can link against a static library or an import library.  There are release and debug
versions of both libraries included:

   amp_fft.lib           Release dynamic import library to amp_fft.dll
   amp_fftd.lib          Debug dynamic import library to amp_fftd.dll    

Enjoy.