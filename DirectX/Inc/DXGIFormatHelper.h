#pragma once
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <dxgiformat.h>
#include <type_traits>

// DXGI Format Utilities
namespace DXGIFormatTraits
{
	bool IsValid( _In_ DXGI_FORMAT fmt );
	bool IsCompressed( _In_ DXGI_FORMAT fmt );
	bool IsPacked( _In_ DXGI_FORMAT fmt );
	bool IsVideo( _In_ DXGI_FORMAT fmt );
	bool IsSRGB( _In_ DXGI_FORMAT fmt );
	bool IsTypeless( _In_ DXGI_FORMAT fmt );
	bool HasAlpha( _In_ DXGI_FORMAT fmt );

	/// <summary>
	/// Calculates the size of a DXGI_FORMAT in bits.
	/// </summary>
	/// <param name="format">The dxgi DXGI_FORMAT_</param>
	/// <returns>size of in bits</returns>
	size_t SizeofDXGIFormatInBits(DXGI_FORMAT format);

	/// <summary>
	/// Calculates the size of a DXGI_FORMAT in bits.
	/// </summary>
	/// <param name="format">The dxgi DXGI_FORMAT_</param>
	/// <returns>size of in bytes</returns>
	inline size_t SizeofDXGIFormatInBytes(DXGI_FORMAT format)
	{
		return SizeofDXGIFormatInBits(format) / 8;
	}
}

namespace DirectX
{

	//-------------------------------------------------------------------------------------
	// Loads an pixel with a DXGI Format into standard RGBA XMVECTOR (aligned)
	//-------------------------------------------------------------------------------------
	XMVECTOR XMLoadDXGIFormat (const void *pSource , DXGI_FORMAT format );
	void	 XMStoreDXGIFormat(void *pDestination , DXGI_FORMAT format , FXMVECTOR V);

	DXGI_FORMAT DXGIConvertFormatDSVToResource(DXGI_FORMAT format);
	DXGI_FORMAT DXGIConvertFormatDSVToSRV(DXGI_FORMAT format);

	/// <summary>
	/// Deduce the proper DXGI_FORMAT for give type.
	/// </summary>
	/// <template param>The Type use in CPU</param>
	/// <returns>DXGI_FORMAT</returns>
	template <class T>
	struct ExtractDXGIFormat
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_UNKNOWN>
	{};
	template <>
	struct ExtractDXGIFormat<uint32_t>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32_UINT>
	{};
	template <>
	struct ExtractDXGIFormat<uint16_t>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R16_UINT>
	{};
	template <>
	struct ExtractDXGIFormat<uint8_t>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R8_UINT>
	{};
	template <>
	struct ExtractDXGIFormat<int32_t>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32_SINT>
	{};
	template <>
	struct ExtractDXGIFormat<int16_t>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R16_SINT>
	{};
	template <>
	struct ExtractDXGIFormat<int8_t>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R8_SINT>
	{};
	template <>
	struct ExtractDXGIFormat<float>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32_FLOAT>
	{};
	template <>
	struct ExtractDXGIFormat<XMVECTOR>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32A32_FLOAT>
	{};
	template <>
	struct ExtractDXGIFormat<XMFLOAT2>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32_FLOAT>
	{};
	template <>
	struct ExtractDXGIFormat<XMFLOAT3>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32_FLOAT>
	{};
	template <>
	struct ExtractDXGIFormat<XMFLOAT4>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32A32_FLOAT>
	{};
	template <>
	struct ExtractDXGIFormat<XMUINT2>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32_UINT>
	{};
	template <>
	struct ExtractDXGIFormat<XMUINT3>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32_UINT>
	{};
	template <>
	struct ExtractDXGIFormat<XMUINT4>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32A32_UINT>
	{};
	template <>
	struct ExtractDXGIFormat<XMINT4>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32A32_SINT>
	{};
	template <>
	struct ExtractDXGIFormat<XMINT3>
		: public std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32B32_SINT>
	{};
	template <>
	struct ExtractDXGIFormat<XMINT2>
		: public  std::integral_constant<DXGI_FORMAT,DXGI_FORMAT_R32G32_SINT>
	{}; 

	namespace PackedVector
	{
		template <DXGI_FORMAT Format>
		struct ExtractPackagedVectorType
		{
			typedef void type;
		};
		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32B32A32_FLOAT>{
			typedef XMFLOAT4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32B32A32_UINT>{
			typedef XMUINT4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32B32A32_SINT>{
			typedef XMINT4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32B32_FLOAT>{
			typedef XMFLOAT3 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32B32_UINT>{
			typedef XMUINT3 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32B32_SINT>{
			typedef XMINT3 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16B16A16_FLOAT>{
			typedef XMHALF4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16B16A16_UNORM>{
			typedef XMUSHORTN4 type;
		}; 

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16B16A16_UINT>{
			typedef XMUSHORT4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16B16A16_SNORM>{
			typedef XMSHORTN4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16B16A16_SINT>{
			typedef XMSHORT4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32_FLOAT>{
			typedef XMFLOAT2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32_UINT>{
			typedef XMUINT2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32G32_SINT>{
			typedef XMINT2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R10G10B10A2_UNORM>{
			typedef XMUDECN4 type;
		};
		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM>{
			typedef XMUDECN4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R10G10B10A2_UINT>{
			typedef XMUDEC4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R11G11B10_FLOAT>{
			typedef XMFLOAT3PK type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R8G8B8A8_UNORM>{
			typedef XMUBYTEN4 type;
		};
		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R8G8B8A8_UNORM_SRGB>{
			typedef XMUBYTEN4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R8G8B8A8_UINT>{
			typedef XMUBYTE4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R8G8B8A8_SNORM>{
			typedef XMBYTEN4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R8G8B8A8_SINT>{
			typedef XMBYTE4 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16_FLOAT>{
			typedef XMHALF2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16_UNORM>{
			typedef XMUSHORTN2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16_UINT>{
			typedef XMUSHORT2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16_SNORM>{
			typedef XMSHORTN2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R16G16_SINT>{
			typedef XMSHORT2 type;
		};

		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_D32_FLOAT>{
			typedef float  type;
		};
		template <>
		struct ExtractPackagedVectorType<DXGI_FORMAT_R32_FLOAT>{
			typedef float  type;
		};
	}

}