AMPBLAS: C++ AMP BLAS Library

This library contains an adaptation of the legacy cblas interface to BLAS for 
C++ AMP. At this point almost all interfaces are not implemented. One 
exception is the ampblas_saxpy and ampblas_daxpy which serve as a template for the 
implementation of other routines.

The library also contains templated C++ interface for the cblas routines which 
in certain degree can simplify using BLAS in C++ AMP for C++ programmers. In the 
future, we might expand the project to provide an abstract layer on top of the C++ 
interface, such as expression template based matrix class, to enrich programmability. 

The interface for the legacy cblas is defined in inc\ampcblas.h file. The templated 
C++ interface for the cblas routines is defined in inc\ampblas.h

In order to use AMPBLAS you need first build the library. The library can be built 
using the Visual Studio project file provided, which is created using Visual Studio
11 Beta (You can download it from: http://www.microsoft.com/visualstudio/11/en-us/downloads).
The library is built in the .dll format and installed at:

  debug\ampblasd.dll      ---- // win32 debug version
  release\ampblas.dll     ---- // win32 release version
  x64\debug\ampblasd.dll   --- // amd64 debug version
  x64\release\ampblas.dll   -- // amd64 release version

Once the library is built, to use the library, you need to:

1) Include inc\ampcblas.h in your source if you want to use the legacy cblas interface
   or include inc\ampblas.h in your source if you want to use the C++ interface
2) Compile your .cpp source files
3) Link the resulting object files with the ampblas.lib or ampblasd.lib which is in 
   the corresponding install directory.
   
To run your application, you need to add to the path where the library is installed. 
You also need to have DirectX 11 capable cards, or you can run your application on
DirectX 11 Emulator.

Enjoy!
