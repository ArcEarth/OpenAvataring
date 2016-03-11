#pragma once
#include <vector>
#include "Math3D.h"
#include "String.h"

namespace Causality
{
	// rotations must be aligned
	// rotations of each bone in the chain
	// root is bone[0]
	// bone[0] is default to begin at identity (t=0,r=0)
	class ChainInverseKinematics
	{
	protected:
		std::vector<DirectX::Vector4, DirectX::XMAllocator>	m_bones;
		std::vector<Vector3>   m_jointWeights;
		std::vector<Vector3>   m_boneMinLimits; // Eular angles
		std::vector<Vector3>   m_boneMaxLimits; // Eular angles
		float				   m_tol;
		int					   m_maxItrs;

		void computeJointWeights();

	public:
		// Chain Reference Vector
		// m_bones[i] stands for the vector from bone[i]'s begin postition to end postition
		ChainInverseKinematics(size_t n);
		ChainInverseKinematics();

		void resize(size_t n);

		float tol() const { return m_tol; };
		void setTol(float value) { m_tol = value; };
		int maxIterations() const { return m_maxItrs; }
		void setmaxIterations(int value) { m_maxItrs = value; }

		// Accessors
		DirectX::Vector4& bone(int i) { return m_bones[i]; }
		const DirectX::Vector4& bone(int i) const { return m_bones[i]; }

		DirectX::Vector3& minRotation(int i) { return m_boneMinLimits[i]; }
		const DirectX::Vector3& minRotation(int i) const { return m_boneMinLimits[i]; }

		DirectX::Vector3& maxRotation(int i) { return m_boneMaxLimits[i]; }
		const DirectX::Vector3& maxRotation(int i) const { return m_boneMaxLimits[i]; }

		array_view<DirectX::Vector4> bones() { return m_bones; }
		array_view<const DirectX::Vector4> bones() const { return m_bones; }

		array_view<DirectX::Vector3> maxRotations() { return m_boneMaxLimits; }
		array_view<const DirectX::Vector3> maxRotations() const { return m_boneMaxLimits; }

		array_view<DirectX::Vector3> minRotations() { return m_boneMinLimits; }
		array_view<const DirectX::Vector3> minRotations() const { return m_boneMinLimits; }

		size_t size() const { return m_bones.size(); }

		XMVECTOR endPosition(array_view<const Quaternion> rotations) const;

		/// <summary>
		/// <para>Compute the end effctor position's Jaccobi Matrix Respect to joint's euler angles.</para>
		/// <para>Joint Euler Angle Rotation order are Roll(Z)-Patch(X)-Yaw(Y).</para>
		/// <para>Result array could be safly cast to column major Matrix with dimension (3x3n), which are exactly the Jaccobi .</para>
		/// <example>
		/// <code>
		/// vector&lt;Vector3&gt; jac(3*cik.size());
		/// cik.endPositionJaccobiRespectEuler(rotations,jac);
		/// eigenJac = Eigen::Matrix3Xf::Map(&amp;jac[0].x,3,3*cik.size());
		/// </code>
		/// </example>
		/// </summary>
		/// <param name="rotations">Local Rotation Quaternions for each joint</param>
		/// <param name="jacb">size should be 3n</param>
		void endPositionJaccobiRespectEuler(array_view<const Quaternion> rotations, _Out_ array_view<Vector3> jacb) const;
		void endPositionJaccobiRespectAxisAngle(array_view<const Quaternion> rotations, _Out_ array_view<Vector3> jacb) const;
		void endPositionJaccobiRespectLnQuaternion(array_view<const Quaternion> rotations, _Out_ array_view<Vector3> jacb) const;

		bool XM_CALLCONV solve(_In_ FXMVECTOR goal, _Inout_ array_view<Quaternion> rotations) const;

		// solve the IK optimization problem with a reference configuration
		// the reference are weighted in min-square manar
		bool XM_CALLCONV solveWithStyle(_In_ FXMVECTOR goal, _Inout_ array_view<Quaternion> rotations, _In_ array_view<Quaternion> styleReference, _In_ float styleReferenceWeight) const;

	protected:
		std::vector<Vector4, DirectX::XMAllocator> getRadiusVectors(array_view<const Quaternion> &rotations) const;

		// Static functions
	public:
		// Jaccobbi from a rotation radius vector (r) respect to a small rotation dr = (drx,dry,drz) in global reference frame
		static void jacobbiRespectAxisAngle(_Out_ Matrix4x4 &j, _In_reads_(3) const float* r);

		static XMMATRIX XM_CALLCONV jacobbiTransposeRespectEuler(const Vector3& r, const Vector3& euler, FXMVECTOR globalRot);

		static XMMATRIX XM_CALLCONV jacobbiTransposeRespectAxisAngle(const Vector3& r, FXMVECTOR qrot, FXMVECTOR globalRot);
	private:
		struct OptimizeFunctor;
	};

}