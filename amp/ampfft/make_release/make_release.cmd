rmdir /s drop
mkdir drop
mkdir drop\inc
mkdir drop\x64
mkdir drop\x64\bin
mkdir drop\x64\lib
mkdir drop\x86
mkdir drop\x86\bin
mkdir drop\x86\lib

copy ReadMe.txt drop
copy License.txt drop
xcopy ..\inc drop\inc /S/I

copy ..\bin\x64\debug\amp_fftd.dll drop\x64\bin
copy ..\bin\x64\debug\amp_fftd.pdb drop\x64\bin
copy ..\bin\x64\release\amp_fft.dll drop\x64\bin
copy ..\bin\x64\release\amp_fft.pdb drop\x64\bin
copy ..\bin\x64\debug\amp_fftd.lib drop\x64\lib
copy ..\bin\x64\release\amp_fft.lib drop\x64\lib

copy ..\bin\x86\debug\amp_fftd.dll drop\x86\bin
copy ..\bin\x86\debug\amp_fftd.pdb drop\x86\bin
copy ..\bin\x86\release\amp_fft.dll drop\x86\bin
copy ..\bin\x86\release\amp_fft.pdb drop\x86\bin
copy ..\bin\x86\debug\amp_fftd.lib drop\x86\lib
copy ..\bin\x86\release\amp_fft.lib drop\x86\lib
