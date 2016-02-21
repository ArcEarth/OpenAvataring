AMP_FFT: C++ AMP FFT Library

The C++ AMP FFT library is built around the Direct3D FFT API’s. It can perform 
forward and backwards transforms. 

In order to use C++ AMP FFT library you need first build the library. The library 
can be built using the Visual Studio project file provided, which is created 
using Visual Studio 2012 (You can download it from: 
http://www.microsoft.com/visualstudio/eng/downloads). The library is built 
in the .dll format and installed at:

    bin\x86\debug\amp_fftd.dll      --- win32 debug version
    bin\x86\release\amp_fft.dll     --- win32 release version
    bin\x64\debug\amp_fftd.dll      --- amd64 debug version
    bin\x64\release\amp_fft.dll     --- amd64 release version
  
Once the library is built, to use the library, you need to:

1) Include inc\amp_fft.h in your source 
2) Compile your .cpp source files
3) Link the resulting object files with the amp_fft.lib or amp_fftd.lib which is in 
   the corresponding install directory.
   
The sample directory has an example shows how to use C++ AMP FFT library to perform 
forward and backwards transforms. 
