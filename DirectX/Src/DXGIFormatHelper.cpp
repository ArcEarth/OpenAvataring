#include "DXGIFormatHelper.h"
#include <stdexcept>
#include <algorithm>

using namespace DirectX;
using namespace DirectX::PackedVector;
using namespace DXGIFormatTraits;

#define LOAD_ELEMENT4( type, func )\
{\
	const type * __restrict sPtr = reinterpret_cast<const type*>(pSource);\
	return func( sPtr );\
}

#define LOAD_ELEMENT3( type, func, defvec )\
{\
	const type * __restrict sPtr = reinterpret_cast<const type*>(pSource);\
	XMVECTOR v = func( sPtr );\
	return XMVectorSelect( defvec, v, g_XMSelect1110 );\
}

#define LOAD_ELEMENT2( type, func, defvec )\
{\
	const type * __restrict sPtr = reinterpret_cast<const type*>(pSource);\
	XMVECTOR v = func( sPtr );\
	return XMVectorSelect( defvec, v, g_XMSelect1100 );\
}

#define LOAD_ELEMENT1( type, func, defvec )\
{\
	const type * __restrict sPtr = reinterpret_cast<const type*>(pSource);\
	XMVECTOR v = func( sPtr );\
	return XMVectorSelect( defvec, v, g_XMSelect1000 );\
}

#define SAVE_ELEMENT( type, func ) \
	func( reinterpret_cast<type*>(pDestination) , V);\
	break;

XMVECTOR DirectX::XMLoadDXGIFormat(const void *pSource, DXGI_FORMAT format)
{
	assert(pSource);
	assert(IsValid(format) && !IsVideo(format) && !IsTypeless(format) && !IsCompressed(format) && !IsPacked(format));

	switch (format)
	{
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		LOAD_ELEMENT4(XMFLOAT4, XMLoadFloat4)

	case DXGI_FORMAT_R32G32B32A32_UINT:
		LOAD_ELEMENT4(XMUINT4, XMLoadUInt4)

	case DXGI_FORMAT_R32G32B32A32_SINT:
		LOAD_ELEMENT4(XMINT4, XMLoadSInt4)

	case DXGI_FORMAT_R32G32B32_FLOAT:
		LOAD_ELEMENT3(XMFLOAT3, XMLoadFloat3, g_XMIdentityR3)

	case DXGI_FORMAT_R32G32B32_UINT:
		LOAD_ELEMENT3(XMUINT3, XMLoadUInt3, g_XMIdentityR3)

	case DXGI_FORMAT_R32G32B32_SINT:
		LOAD_ELEMENT3(XMINT3, XMLoadSInt3, g_XMIdentityR3)

	case DXGI_FORMAT_R16G16B16A16_FLOAT:
		LOAD_ELEMENT4(XMHALF4, XMLoadHalf4)

	case DXGI_FORMAT_R16G16B16A16_UNORM:
		LOAD_ELEMENT4(XMUSHORTN4, XMLoadUShortN4)

	case DXGI_FORMAT_R16G16B16A16_UINT:
		LOAD_ELEMENT4(XMUSHORT4, XMLoadUShort4)

	case DXGI_FORMAT_R16G16B16A16_SNORM:
		LOAD_ELEMENT4(XMSHORTN4, XMLoadShortN4)

	case DXGI_FORMAT_R16G16B16A16_SINT:
		LOAD_ELEMENT4(XMSHORT4, XMLoadShort4)

	case DXGI_FORMAT_R32G32_FLOAT:
		LOAD_ELEMENT2(XMFLOAT2, XMLoadFloat2, g_XMIdentityR3)

	case DXGI_FORMAT_R32G32_UINT:
		LOAD_ELEMENT2(XMUINT2, XMLoadUInt2, g_XMIdentityR3)

	case DXGI_FORMAT_R32G32_SINT:
		LOAD_ELEMENT2(XMINT2, XMLoadSInt2, g_XMIdentityR3)

	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		{
			const float * sPtr = reinterpret_cast<const float*>(pSource);
			const uint8_t* ps8 = reinterpret_cast<const uint8_t*>(&sPtr[1]);
			return XMVectorSet(sPtr[0], static_cast<float>(*ps8), 0.f, 1.f);
		}


	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
		LOAD_ELEMENT4(XMUDECN4, XMLoadUDecN4);

	case DXGI_FORMAT_R10G10B10A2_UINT:
		LOAD_ELEMENT4(XMUDEC4, XMLoadUDec4);

	case DXGI_FORMAT_R11G11B10_FLOAT:
		LOAD_ELEMENT3(XMFLOAT3PK, XMLoadFloat3PK, g_XMIdentityR3);

	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		LOAD_ELEMENT4(XMUBYTEN4, XMLoadUByteN4)

	case DXGI_FORMAT_R8G8B8A8_UINT:
		LOAD_ELEMENT4(XMUBYTE4, XMLoadUByte4)

	case DXGI_FORMAT_R8G8B8A8_SNORM:
		LOAD_ELEMENT4(XMBYTEN4, XMLoadByteN4)

	case DXGI_FORMAT_R8G8B8A8_SINT:
		LOAD_ELEMENT4(XMBYTE4, XMLoadByte4)

	case DXGI_FORMAT_R16G16_FLOAT:
		LOAD_ELEMENT2(XMHALF2, XMLoadHalf2, g_XMIdentityR3)

	case DXGI_FORMAT_R16G16_UNORM:
		LOAD_ELEMENT2(XMUSHORTN2, XMLoadUShortN2, g_XMIdentityR3)

	case DXGI_FORMAT_R16G16_UINT:
		LOAD_ELEMENT2(XMUSHORT2, XMLoadUShort2, g_XMIdentityR3)

	case DXGI_FORMAT_R16G16_SNORM:
		LOAD_ELEMENT2(XMSHORTN2, XMLoadShortN2, g_XMIdentityR3)

	case DXGI_FORMAT_R16G16_SINT:
		LOAD_ELEMENT2(XMSHORT2, XMLoadShort2, g_XMIdentityR3)

	case DXGI_FORMAT_D32_FLOAT:
	case DXGI_FORMAT_R32_FLOAT:
		LOAD_ELEMENT1(float, XMLoadFloat, g_XMIdentityR3);

	case DXGI_FORMAT_R32_UINT:
	{
		const uint32_t* __restrict sPtr = reinterpret_cast<const uint32_t*>(pSource);
		XMVECTOR v = XMLoadInt(sPtr);
		v = XMConvertVectorUIntToFloat(v, 0);
		return XMVectorSelect(g_XMIdentityR3, v, g_XMSelect1000);
	}

	case DXGI_FORMAT_R32_SINT:
	{
		const int32_t * __restrict sPtr = reinterpret_cast<const int32_t*>(pSource);
		XMVECTOR v = XMLoadInt(reinterpret_cast<const uint32_t*> (sPtr));
		v = XMConvertVectorIntToFloat(v, 0);
		return XMVectorSelect(g_XMIdentityR3, v, g_XMSelect1000);
	}

	case DXGI_FORMAT_D24_UNORM_S8_UINT:
	{
		const uint32_t * sPtr = reinterpret_cast<const uint32_t*>(pSource);
		float d = static_cast<float>(*sPtr & 0xFFFFFF) / 16777215.f;
		float s = static_cast<float>((*sPtr & 0xFF000000) >> 24);
		return XMVectorSet(d, s, 0.f, 1.f);
	}
	case DXGI_FORMAT_R8G8_UNORM:
		LOAD_ELEMENT2(XMUBYTEN2, XMLoadUByteN2, g_XMIdentityR3)

	case DXGI_FORMAT_R8G8_UINT:
		LOAD_ELEMENT2(XMUBYTE2, XMLoadUByte2, g_XMIdentityR3)

	case DXGI_FORMAT_R8G8_SNORM:
		LOAD_ELEMENT2(XMBYTEN2, XMLoadByteN2, g_XMIdentityR3)

	case DXGI_FORMAT_R8G8_SINT:
		LOAD_ELEMENT2(XMBYTE2, XMLoadByte2, g_XMIdentityR3)

	case DXGI_FORMAT_R16_FLOAT:
		{
			const HALF * __restrict sPtr = reinterpret_cast<const HALF*>(pSource);
			return XMVectorSet(XMConvertHalfToFloat(*sPtr), 0.f, 0.f, 1.f);
		}
	case DXGI_FORMAT_D16_UNORM:
	case DXGI_FORMAT_R16_UNORM:
	{
		const uint16_t* __restrict sPtr = reinterpret_cast<const uint16_t*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr) / 65535.f, 0.f, 0.f, 1.f);
	}

	case DXGI_FORMAT_R16_UINT:
	{
		const uint16_t * __restrict sPtr = reinterpret_cast<const uint16_t*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr), 0.f, 0.f, 1.f);

	}


	case DXGI_FORMAT_R16_SNORM:
	{
		const int16_t * __restrict sPtr = reinterpret_cast<const int16_t*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr) / 32767.f, 0.f, 0.f, 1.f);

	}

	case DXGI_FORMAT_R16_SINT:
	{
		const int16_t * __restrict sPtr = reinterpret_cast<const int16_t*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr), 0.f, 0.f, 1.f);
	}

	case DXGI_FORMAT_R8_UNORM:
	{
		const uint8_t * __restrict sPtr = reinterpret_cast<const uint8_t*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr) / 255.f, 0.f, 0.f, 1.f);
	}


	case DXGI_FORMAT_R8_UINT:
	{
		const uint8_t * __restrict sPtr = reinterpret_cast<const uint8_t*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr), 0.f, 0.f, 1.f);
	}

	case DXGI_FORMAT_R8_SNORM:
	{
		const char * __restrict sPtr = reinterpret_cast<const char*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr) / 127.f, 0.f, 0.f, 1.f);
	}


	case DXGI_FORMAT_R8_SINT:
	{
		const char * __restrict sPtr = reinterpret_cast<const char*>(pSource);
		return XMVectorSet(static_cast<float>(*sPtr), 0.f, 0.f, 1.f);
	}


	case DXGI_FORMAT_A8_UNORM:
	{
		const uint8_t * __restrict sPtr = reinterpret_cast<const uint8_t*>(pSource);
		return XMVectorSet(0.f, 0.f, 0.f, static_cast<float>(*sPtr) / 255.f);
	}


	case DXGI_FORMAT_R1_UNORM:
	{
		const uint8_t * __restrict sPtr = reinterpret_cast<const uint8_t*>(pSource);
		for (size_t bcount = 0; bcount < 8; ++bcount)
		{
			return XMVectorSet((((*sPtr >> bcount) & 0x1) ? 1.f : 0.f), 0.f, 0.f, 1.f);
		}
	}


	case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
		LOAD_ELEMENT3(XMFLOAT3SE, XMLoadFloat3SE, g_XMIdentityR3)

			//case DXGI_FORMAT_R8G8_B8G8_UNORM:
			//	{
			//		const XMUBYTEN4 * __restrict sPtr = reinterpret_cast<const XMUBYTEN4*>(pSource);

			//		XMVECTOR v = XMLoadUByteN4( sPtr );
			//		XMVECTOR v1 = XMVectorSwizzle<0, 3, 2, 1>( v );
			//		return XMVectorSelect( g_XMIdentityR3, v, g_XMSelect1110 );
			//		return XMVectorSelect( g_XMIdentityR3, v1, g_XMSelect1110 );
			//	}


			//case DXGI_FORMAT_G8R8_G8B8_UNORM:
			//	{
			//		const XMUBYTEN4 * __restrict sPtr = reinterpret_cast<const XMUBYTEN4*>(pSource);

			//		XMVECTOR v = XMLoadUByteN4( sPtr );
			//		XMVECTOR v0 = XMVectorSwizzle<1, 0, 3, 2>( v );
			//		XMVECTOR v1 = XMVectorSwizzle<1, 2, 3, 0>( v );
			//		return XMVectorSelect( g_XMIdentityR3, v0, g_XMSelect1110 );
			//		return XMVectorSelect( g_XMIdentityR3, v1, g_XMSelect1110 );
			//	}


	case DXGI_FORMAT_B5G6R5_UNORM:
		{
			static XMVECTORF32 s_Scale = { 1.f / 31.f, 1.f / 63.f, 1.f / 31.f, 1.f };
			const XMU565 * __restrict sPtr = reinterpret_cast<const XMU565*>(pSource);
			XMVECTOR v = XMLoadU565(sPtr);
			v = XMVectorMultiply(v, s_Scale);
			v = XMVectorSwizzle<2, 1, 0, 3>(v);
			return XMVectorSelect(g_XMIdentityR3, v, g_XMSelect1110);
		}


	case DXGI_FORMAT_B5G5R5A1_UNORM:
	{
		static XMVECTORF32 s_Scale = { 1.f / 31.f, 1.f / 31.f, 1.f / 31.f, 1.f };
		const XMU555 * __restrict sPtr = reinterpret_cast<const XMU555*>(pSource);
		XMVECTOR v = XMLoadU555(sPtr);
		v = XMVectorMultiply(v, s_Scale);
		return XMVectorSwizzle<2, 1, 0, 3>(v);
	}


	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	{
		const XMUBYTEN4 * __restrict sPtr = reinterpret_cast<const XMUBYTEN4*>(pSource);
		XMVECTOR v = XMLoadUByteN4(sPtr);
		return XMVectorSwizzle<2, 1, 0, 3>(v);
	}


	case DXGI_FORMAT_B8G8R8X8_UNORM:
	case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
	{
		const XMUBYTEN4 * __restrict sPtr = reinterpret_cast<const XMUBYTEN4*>(pSource);
		XMVECTOR v = XMLoadUByteN4(sPtr);
		v = XMVectorSwizzle<2, 1, 0, 3>(v);
		return XMVectorSelect(g_XMIdentityR3, v, g_XMSelect1110);
	}


#ifdef DXGI_1_2_FORMATS
	case DXGI_FORMAT_B4G4R4A4_UNORM:
	{
		static XMVECTORF32 s_Scale = { 1.f / 15.f, 1.f / 15.f, 1.f / 15.f, 1.f / 15.f };
		const XMUNIBBLE4 * __restrict sPtr = reinterpret_cast<const XMUNIBBLE4*>(pSource);


		XMVECTOR v = XMLoadUNibble4(sPtr);
		v = XMVectorMultiply(v, s_Scale);
		return XMVectorSwizzle<2, 1, 0, 3>(v);

	}


	// we don't support the video formats ( see IsVideo function )
#endif // DXGI_1_2_FORMATS

	default:
		throw std::invalid_argument("Unknown package method for DXGI Format.");
	}
}

void	DirectX::XMStoreDXGIFormat(void *pDestination, DXGI_FORMAT format, FXMVECTOR V)
{
	assert(pDestination);
	assert(IsValid(format) && !IsVideo(format) && !IsTypeless(format) && !IsCompressed(format) && !IsPacked(format));

	switch (format)
	{
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
		SAVE_ELEMENT(XMFLOAT4, XMStoreFloat4)

	case DXGI_FORMAT_R32G32B32A32_UINT:
		SAVE_ELEMENT(XMUINT4, XMStoreUInt4)

	case DXGI_FORMAT_R32G32B32A32_SINT:
		SAVE_ELEMENT(XMINT4, XMStoreSInt4)

	case DXGI_FORMAT_R32G32B32_FLOAT:
		SAVE_ELEMENT(XMFLOAT3, XMStoreFloat3)

	case DXGI_FORMAT_R32G32B32_UINT:
		SAVE_ELEMENT(XMUINT3, XMStoreUInt3)

	case DXGI_FORMAT_R32G32B32_SINT:
		SAVE_ELEMENT(XMINT3, XMStoreSInt3)

	case DXGI_FORMAT_R16G16B16A16_FLOAT:
		SAVE_ELEMENT(XMHALF4, XMStoreHalf4)

	case DXGI_FORMAT_R16G16B16A16_UNORM:
		SAVE_ELEMENT(XMUSHORTN4, XMStoreUShortN4)

	case DXGI_FORMAT_R16G16B16A16_UINT:
		SAVE_ELEMENT(XMUSHORT4, XMStoreUShort4)

	case DXGI_FORMAT_R16G16B16A16_SNORM:
		SAVE_ELEMENT(XMSHORTN4, XMStoreShortN4)

	case DXGI_FORMAT_R16G16B16A16_SINT:
		SAVE_ELEMENT(XMSHORT4, XMStoreShort4)

	case DXGI_FORMAT_R32G32_FLOAT:
		SAVE_ELEMENT(XMFLOAT2, XMStoreFloat2)

	case DXGI_FORMAT_R32G32_UINT:
		SAVE_ELEMENT(XMUINT2, XMStoreUInt2)

	case DXGI_FORMAT_R32G32_SINT:
		SAVE_ELEMENT(XMINT2, XMStoreSInt2)

	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		{
			float * dPtr = reinterpret_cast<float*>(pDestination);
			uint8_t* ps8 = reinterpret_cast<uint8_t*>(&dPtr[1]);
			*dPtr = XMVectorGetX(V);
			*ps8 = static_cast<uint8_t>(XMVectorGetY(V));
		}
		break;

	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
		SAVE_ELEMENT(XMUDECN4, XMStoreUDecN4);

	case DXGI_FORMAT_R10G10B10A2_UINT:
		SAVE_ELEMENT(XMUDEC4, XMStoreUDec4);

	case DXGI_FORMAT_R11G11B10_FLOAT:
		SAVE_ELEMENT(XMFLOAT3PK, XMStoreFloat3PK);

	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		SAVE_ELEMENT(XMUBYTEN4, XMStoreUByteN4)

	case DXGI_FORMAT_R8G8B8A8_UINT:
		SAVE_ELEMENT(XMUBYTE4, XMStoreUByte4)

	case DXGI_FORMAT_R8G8B8A8_SNORM:
		SAVE_ELEMENT(XMBYTEN4, XMStoreByteN4)

	case DXGI_FORMAT_R8G8B8A8_SINT:
		SAVE_ELEMENT(XMBYTE4, XMStoreByte4)

	case DXGI_FORMAT_R16G16_FLOAT:
		SAVE_ELEMENT(XMHALF2, XMStoreHalf2)

	case DXGI_FORMAT_R16G16_UNORM:
		SAVE_ELEMENT(XMUSHORTN2, XMStoreUShortN2)

	case DXGI_FORMAT_R16G16_UINT:
		SAVE_ELEMENT(XMUSHORT2, XMStoreUShort2)

	case DXGI_FORMAT_R16G16_SNORM:
		SAVE_ELEMENT(XMSHORTN2, XMStoreShortN2)

	case DXGI_FORMAT_R16G16_SINT:
		SAVE_ELEMENT(XMSHORT2, XMStoreShort2)

	case DXGI_FORMAT_D32_FLOAT:
	case DXGI_FORMAT_R32_FLOAT:
		SAVE_ELEMENT(float, XMStoreFloat);

	case DXGI_FORMAT_R32_UINT:
	{

		uint32_t* dPtr = reinterpret_cast<uint32_t*>(pDestination);
		XMVECTOR vUINT = XMConvertVectorFloatToUInt(V, 0);
		XMStoreInt(dPtr, vUINT);
	}
	break;

	case DXGI_FORMAT_R32_SINT:
	{
		uint32_t* dPtr = reinterpret_cast<uint32_t*>(pDestination);
		XMVECTOR vINT = XMConvertVectorFloatToInt(V, 0);
		XMStoreInt(dPtr, vINT);
	}
	break;

	case DXGI_FORMAT_D24_UNORM_S8_UINT:
	{
		uint32_t * sPtr = reinterpret_cast<uint32_t*>(pDestination);
		uint32_t s = XMVectorGetIntY(V);
		uint32_t d = static_cast<uint32_t>(XMVectorGetX(V) * 16777215.f);
		*sPtr = d & 0xFFFFFF | (s & 0xFF) << 24;
	}
	break;
	case DXGI_FORMAT_R8G8_UNORM:
		SAVE_ELEMENT(XMUBYTEN2, XMStoreUByteN2)

	case DXGI_FORMAT_R8G8_UINT:
		SAVE_ELEMENT(XMUBYTE2, XMStoreUByte2)

	case DXGI_FORMAT_R8G8_SNORM:
		SAVE_ELEMENT(XMBYTEN2, XMStoreByteN2)

	case DXGI_FORMAT_R8G8_SINT:
		SAVE_ELEMENT(XMBYTE2, XMStoreByte2)

	case DXGI_FORMAT_R16_FLOAT:
		{
			HALF * dPtr = reinterpret_cast<HALF*>(pDestination);
			*dPtr = XMConvertFloatToHalf(XMVectorGetX(V));
		}
		break;
	case DXGI_FORMAT_D16_UNORM:
	case DXGI_FORMAT_R16_UNORM:
	{
		uint16_t* dPtr = reinterpret_cast<uint16_t*>(pDestination);
		*dPtr = static_cast<uint16_t>(XMVectorGetX(V) * 65535.f);
	}
	break;
	case DXGI_FORMAT_R16_UINT:
	{
		uint16_t * dPtr = reinterpret_cast<uint16_t*>(pDestination);
		*dPtr = static_cast<uint16_t>(XMVectorGetX(V));
	}
	break;
	case DXGI_FORMAT_R16_SNORM:
	{
		int16_t * dPtr = reinterpret_cast<int16_t*>(pDestination);
		*dPtr = static_cast<uint16_t>(XMVectorGetX(V) * 32767.f);
	}
	break;
	case DXGI_FORMAT_R16_SINT:
	{
		int16_t * dPtr = reinterpret_cast<int16_t*>(pDestination);
		*dPtr = static_cast<uint16_t>(XMVectorGetX(V));
	}
	break;
	case DXGI_FORMAT_R8_UNORM:
	{
		uint8_t * dPtr = reinterpret_cast<uint8_t*>(pDestination);
		*dPtr = static_cast<uint8_t>(XMVectorGetX(V) * 255.f);
	}
	break;
	case DXGI_FORMAT_R8_UINT:
	{
		uint8_t * dPtr = reinterpret_cast<uint8_t*>(pDestination);
		*dPtr = static_cast<uint8_t>(XMVectorGetX(V));
	}
	break;
	case DXGI_FORMAT_R8_SNORM:
	{
		int8_t * dPtr = reinterpret_cast<int8_t*>(pDestination);
		*dPtr = static_cast<int8_t>(XMVectorGetX(V) * 127.f);
	}
	break;
	case DXGI_FORMAT_R8_SINT:
	{
		int8_t * dPtr = reinterpret_cast<int8_t*>(pDestination);
		*dPtr = static_cast<int8_t>(XMVectorGetX(V));
	}
	break;
	case DXGI_FORMAT_A8_UNORM:
	{
		uint8_t * dPtr = reinterpret_cast<uint8_t*>(pDestination);
		*dPtr = static_cast<uint8_t>(XMVectorGetW(V) * 255.f);
	}
	break;
	// Let's forgot about this xxx R1 ?!
	//case DXGI_FORMAT_R1_UNORM:
	//	{
	//		uint8_t * dPtr = reinterpret_cast<uint8_t*>(pDestination);
	//		for( size_t bcount = 0; bcount < 8; ++bcount )
	//		{
	//			return XMVectorSet( (((*dPtr >> bcount) & 0x1) ? 1.f : 0.f), 0.f, 0.f, 1.f );
	//		}
	//	}
		//break;
	case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
		SAVE_ELEMENT(XMFLOAT3SE, XMStoreFloat3SE)
			break;
	case DXGI_FORMAT_B5G6R5_UNORM:
	{
		static XMVECTORF32 s_Scale = { 31.f, 63.f, 31.f, 1.f };
		XMU565 * dPtr = reinterpret_cast<XMU565*>(pDestination);
		XMVECTOR vRGBA = XMVectorSwizzle<2, 1, 0, 3>(V);
		vRGBA = XMVectorMultiply(vRGBA, s_Scale);
		XMStoreU565(dPtr, vRGBA);
	}
	break;
	case DXGI_FORMAT_B5G5R5A1_UNORM:
	{
		static XMVECTORF32 s_Scale = { 31.f, 31.f, 31.f, 1.f };
		XMU555 * dPtr = reinterpret_cast<XMU555*>(pDestination);
		XMVECTOR vRGBA = XMVectorSwizzle<2, 1, 0, 3>(V);
		vRGBA = XMVectorMultiply(vRGBA, s_Scale);
		XMStoreU555(dPtr, vRGBA);
	}
	break;
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	{
		XMUBYTEN4 * dPtr = reinterpret_cast<XMUBYTEN4*>(pDestination);
		XMVECTOR vRGBA = XMVectorSwizzle<2, 1, 0, 3>(V);
		XMStoreUByteN4(dPtr, vRGBA);
	}
	break;
	case DXGI_FORMAT_B8G8R8X8_UNORM:
	case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
	{
		XMUBYTEN4 * dPtr = reinterpret_cast<XMUBYTEN4*>(pDestination);
		XMVECTOR vRGBA = XMVectorSwizzle<2, 1, 0, 3>(V);
		XMStoreUByteN4(dPtr, vRGBA);
	}
	break;
#ifdef DXGI_1_2_FORMATS
	case DXGI_FORMAT_B4G4R4A4_UNORM:
	{
		static XMVECTORF32 s_Scale = { 15.f, 15.f, 15.f, 15.f };
		XMUNIBBLE4 * dPtr = reinterpret_cast<XMUNIBBLE4*>(pDestination);
		XMVECTOR vRGBA = XMVectorSwizzle<2, 1, 0, 3>(V);
		vRGBA = XMVectorMultiply(vRGBA, s_Scale);
		XMStoreUNibble4(dPtr, vRGB);
	}
	break;
	// we don't support the video formats ( see IsVideo function )
#endif // DXGI_1_2_FORMATS

	default:
		throw std::invalid_argument("Unknown package method for DXGI Format.");
	}
}

DXGI_FORMAT DirectX::DXGIConvertFormatDSVToResource(DXGI_FORMAT format)
{
	DXGI_FORMAT resformat = DXGI_FORMAT_UNKNOWN;
	switch (format)
	{
	case DXGI_FORMAT::DXGI_FORMAT_D16_UNORM:
		resformat = DXGI_FORMAT::DXGI_FORMAT_R16_TYPELESS;
		break;
	case DXGI_FORMAT::DXGI_FORMAT_D24_UNORM_S8_UINT:
		resformat = DXGI_FORMAT::DXGI_FORMAT_R24G8_TYPELESS;
		break;
	case DXGI_FORMAT::DXGI_FORMAT_D32_FLOAT:
		resformat = DXGI_FORMAT::DXGI_FORMAT_R32_TYPELESS;
		break;
	case DXGI_FORMAT::DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		resformat = DXGI_FORMAT::DXGI_FORMAT_R32G8X24_TYPELESS;
		break;
	}

	return resformat;
}

DXGI_FORMAT DirectX::DXGIConvertFormatDSVToSRV(DXGI_FORMAT format)
{
	DXGI_FORMAT srvformat = DXGI_FORMAT_UNKNOWN;
	switch (format)
	{
	case DXGI_FORMAT::DXGI_FORMAT_D16_UNORM:
		srvformat = DXGI_FORMAT::DXGI_FORMAT_R16_UNORM;
		break;
	case DXGI_FORMAT::DXGI_FORMAT_D24_UNORM_S8_UINT:
		srvformat = DXGI_FORMAT::DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
		break;
	case DXGI_FORMAT::DXGI_FORMAT_D32_FLOAT:
		srvformat = DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT;
		break;
	case DXGI_FORMAT::DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		srvformat = DXGI_FORMAT::DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
		break;
	}
	return srvformat;
}


//=====================================================================================
// DXGI Format Utilities
//=====================================================================================

_Use_decl_annotations_
inline bool DXGIFormatTraits::IsValid(DXGI_FORMAT fmt)
{
#ifdef DXGI_1_2_FORMATS
	return (static_cast<size_t>(fmt) >= 1 && static_cast<size_t>(fmt) <= 115);
#else
	return (static_cast<size_t>(fmt) >= 1 && static_cast<size_t>(fmt) <= 99);
#endif
}

_Use_decl_annotations_
inline bool DXGIFormatTraits::IsCompressed(DXGI_FORMAT fmt)
{
	switch (fmt)
	{
	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC1_UNORM:
	case DXGI_FORMAT_BC1_UNORM_SRGB:
	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC4_UNORM:
	case DXGI_FORMAT_BC4_SNORM:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC6H_UF16:
	case DXGI_FORMAT_BC6H_SF16:
	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
		return true;

	default:
		return false;
	}
}

_Use_decl_annotations_
inline bool DXGIFormatTraits::IsPacked(DXGI_FORMAT fmt)
{
	return ((fmt == DXGI_FORMAT_R8G8_B8G8_UNORM) || (fmt == DXGI_FORMAT_G8R8_G8B8_UNORM));
}


_Use_decl_annotations_
inline bool DXGIFormatTraits::IsVideo(DXGI_FORMAT fmt)
{
#ifdef DXGI_1_2_FORMATS
	switch (fmt)
	{
	case DXGI_FORMAT_AYUV:
	case DXGI_FORMAT_Y410:
	case DXGI_FORMAT_Y416:
	case DXGI_FORMAT_NV12:
	case DXGI_FORMAT_P010:
	case DXGI_FORMAT_P016:
	case DXGI_FORMAT_YUY2:
	case DXGI_FORMAT_Y210:
	case DXGI_FORMAT_Y216:
	case DXGI_FORMAT_NV11:
		// These video formats can be used with the 3D pipeline through special view mappings
		return true;

	case DXGI_FORMAT_420_OPAQUE:
	case DXGI_FORMAT_AI44:
	case DXGI_FORMAT_IA44:
	case DXGI_FORMAT_P8:
	case DXGI_FORMAT_A8P8:
		// These are limited use video formats not usable in any way by the 3D pipeline
		return true;

	default:
		return false;
	}
#else // !DXGI_1_2_FORMATS
	//UNREFERENCED_PARAMETER(fmt);
	return false;
#endif
}

_Use_decl_annotations_
inline bool DXGIFormatTraits::IsSRGB(DXGI_FORMAT fmt)
{
	switch (fmt)
	{
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_BC1_UNORM_SRGB:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
		return true;

	default:
		return false;
	}
}

_Use_decl_annotations_
inline bool DXGIFormatTraits::IsTypeless(DXGI_FORMAT fmt)
{
	switch (fmt)
	{
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
	case DXGI_FORMAT_R32G32B32_TYPELESS:
	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R32G32_TYPELESS:
	case DXGI_FORMAT_R32G8X24_TYPELESS:
	case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
	case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R16G16_TYPELESS:
	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_R24G8_TYPELESS:
	case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
	case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
	case DXGI_FORMAT_R8G8_TYPELESS:
	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_R8_TYPELESS:
	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8X8_TYPELESS:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC7_TYPELESS:
		return true;

	default:
		return false;
	}
}

_Use_decl_annotations_
inline bool DXGIFormatTraits::HasAlpha(DXGI_FORMAT fmt)
{
	switch (fmt)
	{
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_UINT:
	case DXGI_FORMAT_R32G32B32A32_SINT:
	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
	case DXGI_FORMAT_R16G16B16A16_UNORM:
	case DXGI_FORMAT_R16G16B16A16_UINT:
	case DXGI_FORMAT_R16G16B16A16_SNORM:
	case DXGI_FORMAT_R16G16B16A16_SINT:
	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10A2_UINT:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_R8G8B8A8_UINT:
	case DXGI_FORMAT_R8G8B8A8_SNORM:
	case DXGI_FORMAT_R8G8B8A8_SINT:
	case DXGI_FORMAT_A8_UNORM:
	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_B5G5R5A1_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
#ifdef DXGI_1_2_FORMATS
	case DXGI_FORMAT_B4G4R4A4_UNORM:
#endif
		return true;

	default:
		return false;
	}
}


/// <summary>
/// Calculates the size of a DXGI_FORMAT in bits.
/// </summary>
/// <param name="format">The dxgi DXGI_FORMAT_</param>
/// <returns>size of in bits</returns>
size_t DXGIFormatTraits::SizeofDXGIFormatInBits(DXGI_FORMAT format)
{
	// Size from doc http://msdn.microsoft.com/en-us/library/bb173059%28VS.85%29.aspx
	switch (format)
	{
	case DXGI_FORMAT_R32G32B32A32_TYPELESS:
	case DXGI_FORMAT_R32G32B32A32_FLOAT:
	case DXGI_FORMAT_R32G32B32A32_UINT:
	case DXGI_FORMAT_R32G32B32A32_SINT:
		return 128;

	case DXGI_FORMAT_R32G32B32_TYPELESS:
	case DXGI_FORMAT_R32G32B32_FLOAT:
	case DXGI_FORMAT_R32G32B32_UINT:
	case DXGI_FORMAT_R32G32B32_SINT:
		return 96;

	case DXGI_FORMAT_R16G16B16A16_TYPELESS:
	case DXGI_FORMAT_R16G16B16A16_FLOAT:
	case DXGI_FORMAT_R16G16B16A16_UNORM:
	case DXGI_FORMAT_R16G16B16A16_UINT:
	case DXGI_FORMAT_R16G16B16A16_SNORM:
	case DXGI_FORMAT_R16G16B16A16_SINT:
	case DXGI_FORMAT_R32G32_TYPELESS:
	case DXGI_FORMAT_R32G32_FLOAT:
	case DXGI_FORMAT_R32G32_UINT:
	case DXGI_FORMAT_R32G32_SINT:
	case DXGI_FORMAT_R32G8X24_TYPELESS:
	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
	case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
	case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
		return 64;

	case DXGI_FORMAT_R10G10B10A2_TYPELESS:
	case DXGI_FORMAT_R10G10B10A2_UNORM:
	case DXGI_FORMAT_R10G10B10A2_UINT:
	case DXGI_FORMAT_R11G11B10_FLOAT:
	case DXGI_FORMAT_R8G8B8A8_TYPELESS:
	case DXGI_FORMAT_R8G8B8A8_UNORM:
	case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
	case DXGI_FORMAT_R8G8B8A8_UINT:
	case DXGI_FORMAT_R8G8B8A8_SNORM:
	case DXGI_FORMAT_R8G8B8A8_SINT:
	case DXGI_FORMAT_R16G16_TYPELESS:
	case DXGI_FORMAT_R16G16_FLOAT:
	case DXGI_FORMAT_R16G16_UNORM:
	case DXGI_FORMAT_R16G16_UINT:
	case DXGI_FORMAT_R16G16_SNORM:
	case DXGI_FORMAT_R16G16_SINT:
	case DXGI_FORMAT_R32_TYPELESS:
	case DXGI_FORMAT_D32_FLOAT:
	case DXGI_FORMAT_R32_FLOAT:
	case DXGI_FORMAT_R32_UINT:
	case DXGI_FORMAT_R32_SINT:
	case DXGI_FORMAT_R24G8_TYPELESS:
	case DXGI_FORMAT_D24_UNORM_S8_UINT:
	case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
	case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
	case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
	case DXGI_FORMAT_R8G8_B8G8_UNORM:
	case DXGI_FORMAT_G8R8_G8B8_UNORM:
	case DXGI_FORMAT_B8G8R8A8_UNORM:
	case DXGI_FORMAT_B8G8R8X8_UNORM:
	case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
	case DXGI_FORMAT_B8G8R8A8_TYPELESS:
	case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
	case DXGI_FORMAT_B8G8R8X8_TYPELESS:
	case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
		return 32;

	case DXGI_FORMAT_R8G8_TYPELESS:
	case DXGI_FORMAT_R8G8_UNORM:
	case DXGI_FORMAT_R8G8_UINT:
	case DXGI_FORMAT_R8G8_SNORM:
	case DXGI_FORMAT_R8G8_SINT:
	case DXGI_FORMAT_R16_TYPELESS:
	case DXGI_FORMAT_R16_FLOAT:
	case DXGI_FORMAT_D16_UNORM:
	case DXGI_FORMAT_R16_UNORM:
	case DXGI_FORMAT_R16_UINT:
	case DXGI_FORMAT_R16_SNORM:
	case DXGI_FORMAT_R16_SINT:
	case DXGI_FORMAT_B5G6R5_UNORM:
	case DXGI_FORMAT_B5G5R5A1_UNORM:

#ifdef DXGI_1_2_FORMATS
	case DXGI_FORMAT_B4G4R4A4_UNORM:
#endif
		return 16;

	case DXGI_FORMAT_R8_TYPELESS:
	case DXGI_FORMAT_R8_UNORM:
	case DXGI_FORMAT_R8_UINT:
	case DXGI_FORMAT_R8_SNORM:
	case DXGI_FORMAT_R8_SINT:
	case DXGI_FORMAT_A8_UNORM:
		return 8;

	case DXGI_FORMAT_R1_UNORM:
		return 1;

	case DXGI_FORMAT_BC1_TYPELESS:
	case DXGI_FORMAT_BC1_UNORM:
	case DXGI_FORMAT_BC1_UNORM_SRGB:
	case DXGI_FORMAT_BC4_TYPELESS:
	case DXGI_FORMAT_BC4_UNORM:
	case DXGI_FORMAT_BC4_SNORM:
		return 4;

	case DXGI_FORMAT_BC2_TYPELESS:
	case DXGI_FORMAT_BC2_UNORM:
	case DXGI_FORMAT_BC2_UNORM_SRGB:
	case DXGI_FORMAT_BC3_TYPELESS:
	case DXGI_FORMAT_BC3_UNORM:
	case DXGI_FORMAT_BC3_UNORM_SRGB:
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
	case DXGI_FORMAT_BC6H_TYPELESS:
	case DXGI_FORMAT_BC6H_UF16:
	case DXGI_FORMAT_BC6H_SF16:
	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
		return 8;

	default:
		return 0;
	}
}
