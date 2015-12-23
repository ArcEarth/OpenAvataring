//-----------------------------------------------------------------------------
//	ClientCodes
//-----------------------------------------------------------------------------

#include "ClientCodes.h"

const std::vector< std::string > ClientCodes::MarkerTokens = MakeMarkerTokens();
const std::vector< std::string > ClientCodes::BodyTokens = MakeBodyTokens();



BoneData::BoneData(BodyChannel *_channel,BodyData *_body):channel(_channel),body(_body),visit(false){
}

MarkerData::MarkerData():X(0.0f),Y(0.0f),Z(0.0f),Visible(false){
}

/*void BoneData::init(){
	toWorld=Eigen::Translation3d(body->TX,body->TY,body->TZ)*Eigen::Quaterniond(body->QW,body->QX,body->QY,body->QZ);
	toLocal=Eigen::Translation3d(body->btX,body->btY,body->btZ)*Eigen::Quaterniond(body->bqW,body->bqX,body->bqY,body->bqZ);
}*/

