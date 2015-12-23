#pragma once
#include "DirectXMathExtend.h"

namespace DirectX
{
	// Pure pose and Dynamic data for a bone
	// "Structure" information is not stored here
	// Each bone is a affine transform (Scale-Rotate-Translate) 
	// the bone will store this transform in Local frame and global frame
	XM_ALIGNATTR
	struct HierachicalTransform : public DirectX::AlignedNew<DirectX::XMVECTOR>
	{
	public:
		// Local Data
		XM_ALIGNATTR
		Quaternion LclRotation;
		// Local Translation, represent the offset vector (Pe - Po) in current frame 
		// Typical value should always be (0,l,0), where l = length of the bone
		// Should be constant among time!!!
		XM_ALIGNATTR
		Vector3	LclTranslation;
		float	LclTw; // Padding, w-coff of local translation

		XM_ALIGNATTR
		Vector3	LclScaling; // Local Scaling , adjust this transform to adjust bone length
		float	LclLength;  // offset should = 16 + 3x4 = 28, the Length before any Scaling

		// Global Data (dulplicate with Local)
		// Global Rotation
		XM_ALIGNATTR
		Quaternion GblRotation;
		// Global Position for the ending joint of this bone
		// Aka : End-Position
		XM_ALIGNATTR
		Vector3	GblTranslation;
		float	GblTw; // Padding
		XM_ALIGNATTR
		Vector3	GblScaling;
		float	GblLength;	 // Length of this bone, after scaling



		HierachicalTransform()
			: LclScaling(1.0f), LclLength(1.0f), GblScaling(1.0f), GblLength(1.0f), LclTw(1.0f), GblTw(1.0f)
		{}

		void UpdateGlobalTransform(const HierachicalTransform& parent)
		{
			GlobalTransform() = LocalTransform();
			GlobalTransform() *= parent.GlobalTransform();
		}

		void UpdateGlobalTransform(const IsometricTransform& global)
		{
			GlobalTransform() = LocalTransform();
			GlobalTransform() *= global;
		}

		inline const DirectX::IsometricTransform&		LocalTransform() const { return reinterpret_cast<const DirectX::IsometricTransform&>(*this); }
		inline DirectX::IsometricTransform&			LocalTransform() { return reinterpret_cast<DirectX::IsometricTransform&>(*this); }
		inline const DirectX::IsometricTransform&		GlobalTransform() const { return reinterpret_cast<const DirectX::IsometricTransform&>(this->GblRotation); }
		inline DirectX::IsometricTransform&			GlobalTransform() { return reinterpret_cast<DirectX::IsometricTransform&>(this->GblRotation); }
	};

}