AMP LAPACK: C++ AMP LAPACK Library

This library contains a subset of the LAPACK library for C++ AMP.  This library
depends on the C++ AMP BLAS library (http://ampblas.codeplex.com), which is defined
as an svn:external to the AMP LAPACK source tree.

The library is in the form of C++ header files that are available in "inc" directory.

To use the library:

1) The AMP LAPACK library uses a hybrid approach to maximize performance. As such, a 
   CPU LAPACK library must be included for host calculations. The AMP LAPACK solution has 
   been configured to help with this process through the use of a few important
   environment variables. These include:
   
   LAPACK_LIB_PATH_32   Location the linker should look for a host 32-bit LAPACK library
   LAPACK_LIB_PATH_64   Location the linker should look for a host 64-bit LAPACK library
   LAPACK_LIB_FILES_32  Library (or libraries) the linker should search for 32-bit LAPACK routines
   LAPACK_LIB_FILES_64  Library (or libraries) the linker should search for 64-bit LAPACK routines
   
   Additionally, a number of preprocessor options can be supplied to specify the signature of the 
   intended LAPACK routines:
   
   -D_LAPACK_ILP64    64-bit integers are used in your LAPACK library
   
   -D_LAPACK_UPPER              LAPACK functions are in uppercase (i.e. SGETRF)
   -D_LAPACK_UPPER_UNDERSCORE   LAPACK functions are in uppercase with an underscore (i.e. SGETRF_)
   -D_LAPACK_LOWER              LAPACK functions are in lowercase (i.e. sgetrf)
   -D_LAPACK_LOWER_UNDERSCORE   LAPACK functions are in lowercase with an underscore (i.e. sgetrf_)
	
   If none of this signatures match your link library, simply edit the lapack_host.h header.
      
2) Add "include <amp_lapack.h>" in cpp source file
