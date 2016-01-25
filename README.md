#Open Avataring Toolkit
This is WIP project, which subject to control arbitary skeleton-mesh with arbitary input skeleton stream. <br/>
WIP means:
* No support from Author absolutely. No associate docutment yet.
* The code are also not ensured to compile or run at this stage.
## License : GPL-V3 for WIP stage.

#Build and Dependencies
## Supported Hardware
* Vicon Mocap
* Leap Motion
* Kinect 2 for Windows/XBox One

## Planned Supported Game Engine
* Customized Engein
* Unreal 4.10 + (Todo)
* Unity 5 + (Todo)

##Platform/Compiler
* Windows 8.1+
* Visual Studio 2015 (VC++ 14)

##External SDK and Library
* FBX SDK 2016+ [download](http://download.autodesk.com/us/fbx_release_older/2016.1.2/fbx20161_2_fbxsdk_vs2015_win.exe)
* Kinect SDK 2.0+ [download](http://www.microsoft.com/en-us/download/details.aspx?id=44561)
* Leap SDK 2.31+ [download](http://1drv.ms/1NYFRGk)
+ Eigen 3.2.6+ (included)
+ DirectXTK (included) 
+ tinyXML2 (included)
+ tinyObjLoader (included)
+ GSL (included)
+ CGAL (included)

##Makesure / Define following MARCOS in [Enviroment Variable](http://superuser.com/questions/949560/how-do-i-set-system-environment-variables-in-windows-10) $(PATH) 
* $(LeapSDK_Root) D:\SDKs\LeapSDK
* $(KINECTSDK20_DIR) C:\Program Files\Microsoft SDKs\Kinect\v2.0_1409\
* $(FBX_SDK_ROOT) C:\Program Files\Autodesk\FBX\FBX SDK\2016.1.2
