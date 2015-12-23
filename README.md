#Tangible Holographic Sketch
This is a WIP project <br/>
if you encounter a problem, email Arc.
#Build and Dependencies
##Hardware
* Vicon Mocap
* Leap Motion
* Kinect 2 for Windows/XBox One

##Platform/Compiler
* Windows 8.1+
* Visual Studio 2015 (VC++ 14)

##External SDK and Library
* boost 1.5.9+ [download](http://sourceforge.net/projects/boost/files/boost/1.59.0/)
* FBX SDK 2015+ [download](http://download.autodesk.com/us/fbx_release_older/2016.1.2/fbx20161_2_fbxsdk_vs2015_win.exe)
* Kinect SDK 2.0+ [download](http://www.microsoft.com/en-us/download/details.aspx?id=44561)
* Leap SDK 2.31+ [download](http://1drv.ms/1NYFRGk)
+ Eigen 3.2.6+ (included)
+ DirectXTK (included) 
+ CGAL (included)
+ tinyXML2 (included)
+ tinyObjLoader (included)
+ GSL (included)

##Define following MARCOS in [Enviroment Variable](http://superuser.com/questions/949560/how-do-i-set-system-environment-variables-in-windows-10) $(PATH) 
* $(LeapSDK_Root) D:\SDKs\LeapSDK
* $(BOOST_ROOT) D:\SDKs\boost_1_59_0
* $(KINECTSDK20_DIR) C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\
* $(FBX_SDK_ROOT) C:\Program Files\Autodesk\FBX\FBX SDK\2016.1.2

##Compiled Boost Binary [Download](http://1drv.ms/1Qj7Mkt) 
### Unzip the binary into BOOST_ROOT folder, so that the boost root folder look like this:
* $(BOOST_ROOT)\stage\v140\x64
* $(BOOST_ROOT)\stage\v140\x86

### Dependent Lib
* Boost::Signal2 (this is evil)
* Boost::Any (header only)
* Boost::Range (header only)
* Boost::Regex (secondary dependency, even more evil)