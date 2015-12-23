#pragma once
#include <VertexTypes.h>
#include <wrl\client.h>
#include <string>
#include <Effects.h>
#include "DirectXHelper.h"
#include "DXGIFormatHelper.h"
#include "Carmera.h"

namespace DirectX
{
	template <class _TVertex, UINT Stride = sizeof(_TVertex)>
	class VertexBuffer
	{
	public:
		typedef _TVertex VertexType;

		VertexBuffer(ID3D11Device *pDevice, int Capablity, const VertexType* pInitialData, UINT CPUAccessFlag = 0)
		{
			m_pBuffer = CreateVertexBuffer(pDevice,Capablity,pInitialData,CPUAccessFlag);
			m_Offset = 0;
		}

		operator ID3D11Buffer* const *()
		{
			return m_pBuffer.GetAddressOf() :
		}

		VertexBuffer(VertexBuffer& rhs, UINT offset)
		{
			m_pBuffer = rhs.m_pBuffer;
			Offset = offset;
		}

		unsigned int							Offset;

	protected:
		unsigned int							m_Size;
		unsigned int							m_Capablity;
		Microsoft::WRL::ComPtr<ID3D11Buffer>	m_pBuffer;
	};
}