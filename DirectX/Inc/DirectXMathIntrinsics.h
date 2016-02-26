#pragma once
#if defined(__SSE3__) || defined(__SSE4__) || defined(__AVX__) || defined(__AVX2__)
#ifdef __SSE3__
#include "DirectXMathSSE3.h"
#endif
#ifdef __SSE4__
#include "DirectXMathSSE4.h"
#include "DirectXMathSSE3.h"
#endif
#ifdef __AVX__
#ifdef __FMA4__
#include "DirectXFMA4.h"
#else
#ifdef __FMA3__
#include "DirectXFMA3.h"
#endif
#endif
#include "DirectXMathAVX.h"
#include "DirectXMathSSE4.h"
#include "DirectXMathSSE3.h"
#endif
#ifdef __AVX2__
#include "DirectXMathAVX2.h"
#include "DirectXMathSSE4.h"
#include "DirectXMathSSE3.h"
#endif

#define XM_NAMES ::DirectX::

#if defined(__AVX2__)
// AVX2 Header inlcude a copy of AVX header's swizzle and permute functions
#define XM_PERMUTE	XM_NAMES AVX2::
#define XM_FMA		XM_NAMES AVX2::
#define XM_GET_SET	XM_NAMES SSE4::
#define XM_DOT		XM_NAMES SSE4::
#define XM_ROUND	XM_NAMES SSE4::
#elif defined(__AVX__)
#define XM_PERMUTE XM_NAMES AVX::
#if define(__FMA4__)
#define XM_FMA	   XM_NAMES FMA4::
#elif define(__FMA3__)
#define XM_FMA	   XM_NAMES FMA3::
#define XM_FMA
#endif
#define XM_GET_SET	XM_NAMES SSE4::
#define XM_DOT		XM_NAMES SSE4::
#define XM_ROUND	XM_NAMES SSE4::
#elif defined(__SSE4__)
#define XM_PERMUTE	XM_NAMES
#define XM_FMA		XM_NAMES
#define XM_GET_SET	XM_NAMES SSE4::
#define XM_DOT		XM_NAMES SSE4::
#define XM_ROUND	XM_NAMES SSE4::
#else //defined(__SSE3__)
#define XM_PERMUTE	XM_NAMES
#define XM_FMA		XM_NAMES
#define XM_GET_SET	XM_NAMES
#define XM_DOT		XM_NAMES
#define XM_ROUND	XM_NAMES
#endif

// Name space for the intrinsic override functions
#ifndef _DXMEXT
#define _DXMEXT ::DirectX::Extension::
#endif

namespace DirectX
{
	namespace Extension
	{
		using XM_PERMUTE XMVectorPermute;
		using XM_PERMUTE XMVectorSwizzle;
		using XM_PERMUTE XMVectorShiftLeft;
		using XM_PERMUTE XMVectorRotateLeft;
		using XM_PERMUTE XMVectorRotateRight;
		using XM_PERMUTE XMVectorSplatX;
		using XM_PERMUTE XMVectorSplatY;
		using XM_PERMUTE XMVectorSplatZ;
		using XM_PERMUTE XMVectorSplatW;

		using XM_FMA	 XMVector2Transform;
		using XM_FMA	 XMVector2TransformNormal;
		using XM_FMA	 XMVector2TransformCoord;

		using XM_FMA	 XMVector3Transform;
		using XM_FMA	 XMVector3TransformNormal;
		using XM_FMA	 XMVector3TransformCoord;
		using XM_FMA	 XMVector3Project;
		using XM_FMA	 XMVector3Unproject;

		using XM_FMA	 XMVector4Transform;

		using XM_FMA	 XMMatrixMultiply;
		using XM_FMA	 XMMatrixMultiplyTranspose;

		using XM_NAMES   XMVectorGetXPtr;
		using XM_GET_SET XMVectorGetYPtr;
		using XM_GET_SET XMVectorGetZPtr;
		using XM_GET_SET XMVectorGetWPtr;

		using XM_NAMES   XMVectorGetIntX;
		using XM_GET_SET XMVectorGetIntY;
		using XM_GET_SET XMVectorGetIntZ;
		using XM_GET_SET XMVectorGetIntW;

		using XM_NAMES   XMVectorGetIntXPtr;
		using XM_GET_SET XMVectorGetIntYPtr;
		using XM_GET_SET XMVectorGetIntZPtr;
		using XM_GET_SET XMVectorGetIntWPtr;

		using XM_NAMES	 XMVectorGetX;
		using XM_NAMES	 XMVectorGetY;
		using XM_NAMES	 XMVectorGetZ;
		using XM_NAMES	 XMVectorGetW;

		using XM_NAMES	 XMVectorSetX;
		using XM_GET_SET XMVectorSetY;
		using XM_GET_SET XMVectorSetZ;
		using XM_GET_SET XMVectorSetW;

		using XM_NAMES	 XMVectorSetIntX;
		using XM_GET_SET XMVectorSetIntY;
		using XM_GET_SET XMVectorSetIntZ;
		using XM_GET_SET XMVectorSetIntW;

		using XM_NAMES	 XMVectorSetXPtr;
		using XM_NAMES   XMVectorSetYPtr;
		using XM_NAMES   XMVectorSetZPtr;
		using XM_NAMES   XMVectorSetWPtr;

		using XM_NAMES	 XMVectorSetIntXPtr;
		using XM_NAMES	 XMVectorSetIntYPtr;
		using XM_NAMES	 XMVectorSetIntZPtr;
		using XM_NAMES   XMVectorSetIntWPtr;


		using XM_ROUND	 XMVectorRound;
		using XM_ROUND	 XMVectorTruncate;
		using XM_ROUND	 XMVectorFloor;
		using XM_ROUND	 XMVectorCeiling;

		// Dot intrinsic related
		using XM_DOT	 XMVector2Dot;
		using XM_DOT	 XMVector2LengthSq;
		using XM_DOT	 XMVector2ReciprocalLengthEst;
		using XM_DOT	 XMVector2ReciprocalLength;
		using XM_DOT	 XMVector2LengthEst;
		using XM_DOT	 XMVector2Length;
		using XM_DOT	 XMVector2NormalizeEst;
		using XM_DOT	 XMVector2Normalize;

		using XM_DOT	 XMVector3Dot;
		using XM_DOT	 XMVector3LengthSq;
		using XM_DOT	 XMVector3ReciprocalLengthEst;
		using XM_DOT	 XMVector3ReciprocalLength;
		using XM_DOT	 XMVector3LengthEst;
		using XM_DOT	 XMVector3Length;
		using XM_DOT	 XMVector3NormalizeEst;
		using XM_DOT	 XMVector3Normalize;

		using XM_DOT	 XMVector4Dot;
		using XM_DOT	 XMVector4LengthSq;
		using XM_DOT	 XMVector4ReciprocalLengthEst;
		using XM_DOT	 XMVector4ReciprocalLength;
		using XM_DOT	 XMVector4LengthEst;
		using XM_DOT	 XMVector4Length;
		using XM_DOT	 XMVector4NormalizeEst;
		using XM_DOT	 XMVector4Normalize;

		using XM_DOT	 XMPlaneNormalizeEst;
		using XM_DOT	 XMPlaneNormalize;
	}

#if defined(__SSE3__) && !defined(__SSE4__)
template<> inline XMVECTOR XM_CALLCONV XMVectorSwizzle<0,0,2,2>(FXMVECTOR V) { return _mm_moveldup_ps(V); }
template<> inline XMVECTOR XM_CALLCONV XMVectorSwizzle<1,1,3,3>(FXMVECTOR V) { return _mm_movehdup_ps(V); }	
#endif
}
#else
#define _DXMEXT
#endif