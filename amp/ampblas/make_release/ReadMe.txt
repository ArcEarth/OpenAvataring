AMPBLAS: C++ AMP BLAS Library

This library is for the Visual Studio 2012 RC.  A new version will be released when the 
final product is generally available.

This library contains an adaptation of the legacy cblas interface to BLAS for 
C++ AMP. The library also contains templated C++ interface for the cblas routines which 
in certain degree can simplify using BLAS in C++ AMP for C++ programmers. 

This drop is organized as follows:

   ampblas\inc               Headers for C++ BLAS
   ambcblas\inc              Headers for C BLAS
   x64\lib                   64-bit static and dynamic libs, release and debug
   x64\bin                   64-bit DLL and PDB, release and debug
   x86\lib                   32-bit static and dynamic libs, release and debug
   x86\bin                   32-bit DLL and PDB, release and debug

The interface for the legacy cblas is defined in ampcblas\inc\ampcblas.h file. The templated 
C++ interface for the cblas routines is defined in ampblas\inc\ampblas.h.

To compile:
   1. Add "ampblas\inc" and/or "ampcblas\inc" to your INCLUDE path.
   2. Add "x64\lib" or "x86\lib" to your LIB path.
   3. Add "x64\bin" or "x86\bin" to your PATH.
   4. Choose which lib to link against (see below).
   4. cl -EHs -c foo.cpp  <library>.lib

You can link against a static library or an import library.  There are release and debug
versions of both libraries included:

   ampblas_static.lib      Release static library
   ampblas_staticd.lib     Debug static library
   ampcblas.lib            Release dynamic import library to ampcblas.dll
   ampcblasd.lib           Debug dynamic import library to ampcblasd.dll    

There is a small sample "BlasTest.cpp" in the root of the directory.  You can compile and run it
to verify that you've properly followed the above directions.

Enjoy.