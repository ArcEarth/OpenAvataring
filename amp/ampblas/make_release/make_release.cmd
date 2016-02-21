rmdir /s drop
mkdir drop
mkdir drop\ampblas
mkdir drop\ampcblas
mkdir drop\x64
mkdir drop\x64\bin
mkdir drop\x64\lib
mkdir drop\x86
mkdir drop\x86\bin
mkdir drop\x86\lib

copy ReadMe.txt drop
copy BlasTest.cpp drop
copy License.txt drop
xcopy ..\ampblas\inc drop\ampblas\inc /S/I
xcopy ..\ampcblas\inc drop\ampcblas\inc /S/I

copy ..\bin\x64\debug\ampcblasd.dll drop\x64\bin
copy ..\bin\x64\debug\ampcblasd.pdb drop\x64\bin
copy ..\bin\x64\release\ampcblas.dll drop\x64\bin
copy ..\bin\x64\release\ampcblas.pdb drop\x64\bin
copy ..\bin\x64\debug\ampblas_staticd.lib drop\x64\lib
copy ..\bin\x64\debug\ampcblasd.lib drop\x64\lib
copy ..\bin\x64\release\ampblas_static.lib drop\x64\lib
copy ..\bin\x64\release\ampcblas.lib drop\x64\lib

copy ..\bin\x86\debug\ampcblasd.dll drop\x86\bin
copy ..\bin\x86\debug\ampcblasd.pdb drop\x86\bin
copy ..\bin\x86\release\ampcblas.dll drop\x86\bin
copy ..\bin\x86\release\ampcblas.pdb drop\x86\bin
copy ..\bin\x86\debug\ampblas_staticd.lib drop\x86\lib
copy ..\bin\x86\debug\ampcblasd.lib drop\x86\lib
copy ..\bin\x86\release\ampblas_static.lib drop\x86\lib
copy ..\bin\x86\release\ampcblas.lib drop\x86\lib
