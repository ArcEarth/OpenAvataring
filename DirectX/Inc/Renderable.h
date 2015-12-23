#pragma once
#include "DirectXMathExtend.h"
#include "StepTimer.h"

struct ID3D11DeviceContext;

namespace DirectX{
	namespace Scene
	{
		// Interface for Rendering
		class IRenderable abstract
		{
		public:
			virtual void Render(ID3D11DeviceContext *pContext) = 0;
		};

		// Interface for setting View/Projection Matrix
		// Represent a 3D object that depend on Camera view and 3D location
		class IViewable abstract
		{
		public:
			virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) = 0;
			//virtual void XM_CALLCONV UpdateProjectionMatrix(DirectX::FXMMATRIX projection) = 0;
		};

		// Interface for Time-dependent Animation
		class ITimeAnimatable abstract
		{
		public:
			virtual void UpdateAnimation(StepTimer const& timer) = 0;
		};

		// helper struct that implements IViewable interface and stored matrix for future use
		struct ViewMatrixCache : virtual public IViewable
		{
		public:
			// Inherited via IViewable
			virtual void XM_CALLCONV UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection) override
			{
				ViewMatrix = view;
				ProjectionMatrix = projection;
			}
			//virtual void XM_CALLCONV UpdateProjectionMatrix(DirectX::FXMMATRIX projection) override
			//{
			//	ProjectionMatrix = projection;
			//}
			Matrix4x4	ViewMatrix;
			Matrix4x4	ProjectionMatrix;
		};
	}

}