#pragma once
#include <PrimitiveBatch.h>
#include <Effects.h>
#include <VertexTypes.h>
#include <CommonStates.h>
#include <wrl\client.h>
#include <GeometricPrimitive.h>
#include "DirectXMathExtend.h"
#include <SpriteBatch.h>
//#include "DirectXHelper.h"

namespace DirectX
{
	class PrimitveDrawer
	{
	public:

		typedef VertexPositionColor VertexType;
		PrimitveDrawer();
		explicit PrimitveDrawer(ID3D11DeviceContext *pContext);

		void Initialize(ID3D11DeviceContext *pContext);

		bool Ready() const;;

		ID3D11Device* GetDevice() const;

		ID3D11DeviceContext* GetDeviceContext() const;

		void Release();;

		CommonStates* GetStates() { return m_pStates.get(); }

		void SetWorld(DirectX::CXMMATRIX World);

		void SetProjection(DirectX::CXMMATRIX Projection);

		void SetView(DirectX::CXMMATRIX View);

		void Begin();

		void End();

		PrimitiveBatch<VertexPositionColor>* GetBatch();
		SpriteBatch* GetSpriteBatch();

		void XM_CALLCONV DrawLine(FXMVECTOR P0, FXMVECTOR P1, FXMVECTOR Color);
		void XM_CALLCONV DrawLine(FXMVECTOR P0, FXMVECTOR P1, float Width, FXMVECTOR Color);

		// Extended Line 
		void XM_CALLCONV DrawTriangle(FXMVECTOR P0, FXMVECTOR P1, FXMVECTOR P2, GXMVECTOR Color);

		void XM_CALLCONV DrawQuad(FXMVECTOR P0, FXMVECTOR P1, FXMVECTOR P2, GXMVECTOR P3, CXMVECTOR Color);

		void XM_CALLCONV DrawGeometry(GeometricPrimitive* geometry, FXMMATRIX World, CXMVECTOR Color);
		void XM_CALLCONV DrawSphere(FXMVECTOR Center,float Radius,FXMVECTOR Color);
		void XM_CALLCONV DrawSphere(FXMVECTOR Sphere,FXMVECTOR Color);

		void XM_CALLCONV DrawCylinder(FXMVECTOR P1, FXMVECTOR P2, float radius, FXMVECTOR Color);
		void XM_CALLCONV DrawCylinder(FXMVECTOR Position, FXMVECTOR YDirection, float height, float radius, FXMVECTOR Color);
		//void XM_CALLCONV DrawSphere(FXMVECTOR Position, float radius, FXMVECTOR Color);
		void XM_CALLCONV DrawCube(FXMVECTOR Position, FXMVECTOR HalfExtend, FXMVECTOR Orientation, GXMVECTOR Color);
		void XM_CALLCONV DrawCube(FXMVECTOR HalfExtend, FXMMATRIX WorldTransform , GXMVECTOR Color);
		void XM_CALLCONV DrawCone(FXMVECTOR Position, FXMVECTOR YDirection, float height, float radius, FXMVECTOR Color);

		GeometricPrimitive* GetCube()
		{
			return m_pCube.get();
		}
		GeometricPrimitive* GetCylinder()
		{
			return m_pCylinder.get();
		}
		GeometricPrimitive* GetSphere()
		{
			return m_pSphere.get();
		}
		GeometricPrimitive* GetCone()
		{
			return m_pCone.get();
		}
	protected:
		Matrix4x4	WorldMatrix;
		Matrix4x4	ViewMatrix;
		Matrix4x4	ProjectionMatrix;
		std::unique_ptr<DirectX::GeometricPrimitive> m_pCylinder;
		std::unique_ptr<DirectX::GeometricPrimitive> m_pSphere;
		std::unique_ptr<DirectX::GeometricPrimitive> m_pCube;
		std::unique_ptr<DirectX::GeometricPrimitive> m_pCone;
		::std::unique_ptr<PrimitiveBatch<VertexPositionColor>> m_pBatch;
		::std::unique_ptr<CommonStates>	m_pStates;
		::std::unique_ptr<BasicEffect>	m_pEffect;
		::std::unique_ptr<SpriteBatch>	m_pSprite;
		::Microsoft::WRL::ComPtr<ID3D11InputLayout> m_pInputLayout;
		::Microsoft::WRL::ComPtr<ID3D11InputLayout> m_pGeometryInputLayout;
		::Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_pContext;
		::Microsoft::WRL::ComPtr<ID3D11Device> m_pDevice;
	};

	namespace Visualizers
	{
		__PURE_APPDOMAIN_GLOBAL extern PrimitveDrawer g_PrimitiveDrawer;
	}
}
