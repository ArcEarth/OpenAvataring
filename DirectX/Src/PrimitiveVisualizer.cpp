#include "pch_directX.h"
#include "PrimitiveVisualizer.h"
#include <vector>

//using namespace DirectX;
namespace DirectX{
	namespace Internal{
		// 34 Vertecies
		static const XMVECTORF32 SphereVertices[] = {
			{0.0000f,1.0000f,0.0000f,0.0000f},
			{0.4600f,0.8880f,0.0000f,0.0000f},
			{-0.2300f,0.8880f,-0.3980f,0.0000f},
			{0.8160f,0.5770f,0.0000f,0.0000f},
			{0.2890f,0.8160f,-0.5000f,0.0000f},
			{-0.4080f,0.5770f,-0.7070f,0.0000f},
			{0.9910f,0.1370f,0.0000f,0.0000f},
			{0.7450f,0.2520f,-0.6170f,0.0000f},
			{0.1620f,0.2520f,-0.9540f,0.0000f},
			{-0.4950f,0.1370f,-0.8580f,0.0000f},
			{0.9430f,-0.3330f,0.0000f,0.0000f},
			{0.7610f,-0.5130f,-0.3980f,0.0000f},
			{0.4080f,-0.5770f,-0.7070f,0.0000f},
			{-0.0360f,-0.5130f,-0.8580f,0.0000f},
			{-0.4710f,-0.3330f,-0.8160f,0.0000f},
			{-0.2300f,0.8880f,0.3980f,0.0000f},
			{-0.5770f,0.8160f,0.0000f,0.0000f},
			{-0.4080f,0.5770f,0.7070f,0.0000f},
			{-0.9070f,0.2520f,-0.3370f,0.0000f},
			{-0.9070f,0.2520f,0.3370f,0.0000f},
			{-0.4950f,0.1370f,0.8580f,0.0000f},
			{-0.7250f,-0.5130f,-0.4600f,0.0000f},
			{-0.8160f,-0.5770f,0.0000f,0.0000f},
			{-0.7250f,-0.5130f,0.4600f,0.0000f},
			{-0.4710f,-0.3330f,0.8160f,0.0000f},
			{0.2890f,0.8160f,0.5000f,0.0000f},
			{0.1620f,0.2520f,0.9540f,0.0000f},
			{0.7450f,0.2520f,0.6170f,0.0000f},
			{-0.0360f,-0.5130f,0.8580f,0.0000f},
			{0.4080f,-0.5770f,0.7070f,0.0000f},
			{0.7610f,-0.5130f,0.3980f,0.0000f},
			{0.5770f,-0.8160f,0.0000f,0.0000f},
			{-0.0650f,-0.9390f,0.3370f,0.0000f},
			{-0.0650f,-0.9390f,-0.3370f,0.0000f},
		};

		// 64 Facets
		static const uint16_t SphereIndics[] = {
			2,1,0,
			4,3,1,
			2,4,1,
			5,4,2,
			7,6,3,
			4,7,3,
			8,7,4,
			5,8,4,
			9,8,5,
			11,10,6,
			7,11,6,
			12,11,7,
			8,12,7,
			13,12,8,
			9,13,8,
			14,13,9,
			15,2,0,
			16,5,2,
			15,16,2,
			17,16,15,
			18,9,5,
			16,18,5,
			19,18,16,
			17,19,16,
			20,19,17,
			21,14,9,
			18,21,9,
			22,21,18,
			19,22,18,
			23,22,19,
			20,23,19,
			24,23,20,
			1,15,0,
			25,17,15,
			1,25,15,
			3,25,1,
			26,20,17,
			25,26,17,
			27,26,25,
			3,27,25,
			6,27,3,
			28,24,20,
			26,28,20,
			29,28,26,
			27,29,26,
			30,29,27,
			6,30,27,
			10,30,6,
			11,30,10,
			31,29,30,
			11,31,30,
			12,31,11,
			32,28,29,
			31,32,29,
			33,32,31,
			12,33,31,
			13,33,12,
			23,24,28,
			32,23,28,
			22,23,32,
			33,22,32,
			21,22,33,
			13,21,33,
			14,21,13,
		};

		std::vector<VertexPositionColor> CylinderVertices;
		std::vector<uint16_t> CylinderIndices;
	}

	void XM_CALLCONV PrimitveDrawer::DrawCylinder(FXMVECTOR P1, FXMVECTOR P2, float radius, FXMVECTOR Color)
	{
		auto center = 0.5f * XMVectorAdd(P1, P2);
		auto dir = XMVectorSubtract(P1, P2);
		auto scale = XMVector3Length(dir);
		scale = XMVectorSet(radius, XMVectorGetX(scale), radius, 1.0f);
		XMVECTOR rot;
		if (XMVector4Equal(dir, g_XMZero))
			rot = XMQuaternionIdentity();
		else
			rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, dir);
		XMMATRIX world = XMMatrixAffineTransformation(scale, g_XMZero, rot, center);

		DrawGeometry(m_pCylinder.get(), world, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawCylinder(FXMVECTOR Position, FXMVECTOR YDirection, float height, float radius, FXMVECTOR Color)
	{
		XMVECTOR rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, YDirection);
		XMMATRIX world = XMMatrixAffineTransformation(XMVectorSet(radius,height,radius,1), g_XMZero, rot, Position);

		DrawGeometry(m_pCylinder.get(), world, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawGeometry(GeometricPrimitive * geometry, FXMMATRIX World, CXMVECTOR Color)
	{
		float alpha = XMVectorGetW(Color);
		m_pEffect->SetVertexColorEnabled(false);
		m_pEffect->SetWorld(XMMatrixMultiply(World,WorldMatrix));
		m_pEffect->SetDiffuseColor(XMVectorSetW(Color,1.0f));
		m_pEffect->SetAlpha(alpha);
		//geometry->Draw(World, ViewMatrix, ProjectionMatrix);
		geometry->Draw(m_pEffect.get(), m_pGeometryInputLayout.Get(), alpha < 0.99f);
		m_pEffect->SetWorld(WorldMatrix);
		m_pEffect->SetAlpha(1.0f);
		m_pEffect->SetDiffuseColor(Colors::White.v);
	}

	void XM_CALLCONV PrimitveDrawer::DrawSphere(FXMVECTOR Position, float radius, FXMVECTOR Color)
	{
		XMMATRIX world = XMMatrixAffineTransformation(XMVectorReplicate(radius), g_XMZero, XMQuaternionIdentity(), Position);
		DrawGeometry(m_pSphere.get(), world, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawSphere(FXMVECTOR Sphere, FXMVECTOR Color)
	{
		XMMATRIX world = XMMatrixAffineTransformation(XMVectorSplatW(Sphere), g_XMZero, XMQuaternionIdentity(), Sphere);
		DrawGeometry(m_pSphere.get(), world, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawCube(FXMVECTOR Position, FXMVECTOR HalfExtend, FXMVECTOR Orientation, GXMVECTOR Color)
	{
		XMMATRIX world = XMMatrixAffineTransformation(HalfExtend, g_XMZero, Orientation, Position);
		DrawGeometry(m_pCube.get(), world, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawCube(FXMVECTOR HalfExtend, FXMMATRIX WorldTransform, GXMVECTOR Color)
	{
		XMMATRIX world = XMMatrixScalingFromVector(HalfExtend) * WorldTransform;
		DrawGeometry(m_pCube.get(), world, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawCone(FXMVECTOR Position, FXMVECTOR YDirection, float height, float radius, FXMVECTOR Color)
	{
		XMVECTOR rot;
		if (XMVector4NearEqual(YDirection, g_XMZero.v, g_XMEpsilon.v))
			rot = XMQuaternionIdentity();
		else
			rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, YDirection);

		XMMATRIX world = XMMatrixAffineTransformation(XMVectorSet(radius, height, radius, 1), g_XMZero, rot, Position);
		DrawGeometry(m_pCone.get(), world, Color);
	}

	PrimitveDrawer::PrimitveDrawer()
	{
		Initialize(nullptr);
	}

	void PrimitveDrawer::Initialize(ID3D11DeviceContext * pContext)
	{
		WorldMatrix = Matrix4x4::Identity;
		ViewMatrix = Matrix4x4::Identity;
		ProjectionMatrix = Matrix4x4::Identity;

		if (pContext != nullptr)
		{
			m_pContext = pContext;
			m_pContext->GetDevice(&m_pDevice);
			m_pStates.reset(new CommonStates(m_pDevice.Get()));
			m_pBatch.reset(new PrimitiveBatch<VertexPositionColor>(pContext));
			m_pSprite.reset(new SpriteBatch(pContext));

			m_pEffect.reset(new BasicEffect(m_pDevice.Get()));
			m_pEffect->SetVertexColorEnabled(true);
			m_pEffect->SetLightingEnabled(false);
			m_pEffect->SetFogEnabled(false);
			m_pEffect->SetTextureEnabled(false);

			void const* shaderByteCode;
			size_t byteCodeLength;
			m_pEffect->GetVertexShaderBytecode(&shaderByteCode, &byteCodeLength);
			HRESULT hr = m_pDevice->CreateInputLayout(VertexPositionColor::InputElements,
				VertexPositionColor::InputElementCount,
				shaderByteCode, byteCodeLength,
				&m_pInputLayout);

			m_pEffect->SetVertexColorEnabled(false);
			m_pCylinder = DirectX::GeometricPrimitive::CreateCylinder(pContext, 1.0f, 2.0f,16);
			m_pCylinder->CreateInputLayout(m_pEffect.get(), &m_pGeometryInputLayout);
			m_pSphere = DirectX::GeometricPrimitive::CreateGeoSphere(pContext, 2.0f, 3);
			m_pCube = DirectX::GeometricPrimitive::CreateCube(pContext, 2.0f);
			m_pCone = DirectX::GeometricPrimitive::CreateCone(pContext,2.0f,1.0f,16);

			assert(SUCCEEDED(hr));
		}
	}

	bool PrimitveDrawer::Ready() const
	{
		return m_pContext;
	}

	ID3D11Device * PrimitveDrawer::GetDevice() const
	{
		return m_pDevice.Get();
	}

	ID3D11DeviceContext * PrimitveDrawer::GetDeviceContext() const
	{
		return m_pContext.Get();
	}

	void PrimitveDrawer::Release()
	{
		m_pBatch.release();
		m_pEffect.release();
		m_pInputLayout.Reset();
		m_pContext.Reset();
		m_pDevice.Reset();
	}

	PrimitveDrawer::PrimitveDrawer(ID3D11DeviceContext * pContext)
	{
		Initialize(pContext);
	}

	void PrimitveDrawer::SetWorld(DirectX::CXMMATRIX World)
	{
		m_pEffect->SetWorld(World);
		WorldMatrix = World;
	}

	void PrimitveDrawer::SetProjection(DirectX::CXMMATRIX Projection)
	{
		m_pEffect->SetProjection(Projection);
		ProjectionMatrix = Projection;
	}

	void PrimitveDrawer::SetView(DirectX::CXMMATRIX View)
	{
		m_pEffect->SetView(View);
		ViewMatrix = View;
	}

	void PrimitveDrawer::Begin()
	{
		m_pEffect->SetVertexColorEnabled(true);
		m_pEffect->Apply(m_pContext.Get());
		m_pEffect->SetDiffuseColor(Colors::White.v);
		m_pEffect->SetAlpha(1.0f);
		m_pContext->IASetInputLayout(m_pInputLayout.Get());
		m_pBatch->Begin();
	}

	void PrimitveDrawer::End()
	{
		Microsoft::WRL::ComPtr<ID3D11RasterizerState> pRSState;
		m_pContext->RSGetState(&pRSState);
		m_pContext->RSSetState(m_pStates->CullClockwise());
		m_pBatch->End();
		m_pContext->RSSetState(pRSState.Get());
	}

	inline PrimitiveBatch<VertexPositionColor>* PrimitveDrawer::GetBatch() { return m_pBatch.get(); }

	SpriteBatch * PrimitveDrawer::GetSpriteBatch()
	{
		return m_pSprite.get();
	}

	void XM_CALLCONV PrimitveDrawer::DrawLine(FXMVECTOR P0, FXMVECTOR P1, FXMVECTOR Color)
	{
		VertexType Vertices [] = { VertexType(P0,Color),VertexType(P1,Color) };
		m_pBatch->DrawLine(Vertices[0], Vertices[1]);
	}

	void XM_CALLCONV PrimitveDrawer::DrawLine(FXMVECTOR P0, FXMVECTOR P1, float Width, FXMVECTOR Color)
	{
		DrawLine(P0, P1, Color);
	}

	void XM_CALLCONV PrimitveDrawer::DrawTriangle(FXMVECTOR P0, FXMVECTOR P1, FXMVECTOR P2, GXMVECTOR Color)
	{
		VertexType Vertices [] = { VertexType(P0,Color),VertexType(P1,Color),VertexType(P2,Color) };
		m_pBatch->DrawTriangle(Vertices[0], Vertices[1], Vertices[2]);
	}

	void XM_CALLCONV PrimitveDrawer::DrawQuad(FXMVECTOR P0, FXMVECTOR P1, FXMVECTOR P2, GXMVECTOR P3, CXMVECTOR Color)
	{
		VertexType Vertices [] = { VertexType(P0,Color),VertexType(P1,Color),VertexType(P2,Color),VertexType(P3,Color) };
		m_pBatch->DrawQuad(Vertices[0], Vertices[1], Vertices[2], Vertices[3]);
	}

	//void XM_CALLCONV PrimitveDrawer::DrawSphere(FXMVECTOR Center,float Radius,FXMVECTOR Color)
	//{
	//	VertexPositionColor Vertices[34];
	//	XMVECTOR vRadius = XMVectorReplicate(Radius);

	//	for (int i = 0; i < 34; i++)
	//	{
	//		XMVECTOR vtr = Internal::SphereVertices[i] * vRadius + Center;
	//		XMStoreFloat3(&Vertices[i].position,vtr);
	//		XMStoreFloat4(&Vertices[i].color,Color);
	//	}
	//	m_pDirectXBatch->DrawIndexed(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,Internal::SphereIndics,64*3,Vertices,34);
	//}

	//void XM_CALLCONV PrimitveDrawer::DrawSphere(FXMVECTOR Sphere,FXMVECTOR Color)
	//{
	//	VertexPositionColor Vertices[34];
	//	XMVECTOR vRadius = XMVectorSwizzle<3,3,3,3>(Sphere);

	//	for (int i = 0; i < 34; i++)
	//	{
	//		XMVECTOR vtr = Internal::SphereVertices[i] * vRadius + Sphere;
	//		XMStoreFloat3(&Vertices[i].position,vtr);
	//		XMStoreFloat4(&Vertices[i].color,Color);
	//	}
	//	m_pDirectXBatch->DrawIndexed(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,Internal::SphereIndics,64*3,Vertices,34);
	//}

	namespace Visualizers
	{
		PrimitveDrawer g_PrimitiveDrawer;
	}
}