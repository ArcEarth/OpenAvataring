#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <locale>  
#include <iomanip>
#include "ViconRTClient.h"

ViconRTClient::Ref gViconRTClient;
std::string gViconClientIp =std::string("172.21.12.10");//std::string("localhost");

int main(int argc,char *argv[]){
	gViconRTClient=ViconRTClient::create(gViconClientIp,"./test/test");
	gViconRTClient->start();
	while(true){
		gViconRTClient->update();	
		gViconRTClient->printData();
	}
	return 0;
}