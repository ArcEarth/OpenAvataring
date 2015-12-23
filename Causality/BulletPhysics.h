#pragma once
#include "..\DirectX\Inc\Locatable.h"
#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>
#include <BulletCollision\CollisionDispatch\btGhostObject.h>

namespace Causality
{
	template <class _TTarget, class _TSource>
	const _TTarget& force_cast(const _TSource& v)
	{
		static_assert(sizeof(_TSource) >= sizeof(_TTarget), "Vector dimension must agree.");
		return reinterpret_cast<const _TTarget&>(v);
	}

	template <class _TTarget, class _TSource>
	_TTarget& force_cast(_TSource& v)
	{
		static_assert(sizeof(_TSource) >= sizeof(_TTarget), "Vector dimension must agree.");
		return reinterpret_cast<_TTarget&>(v);
	}

	template <class _TTarget, class _TSource>
	_TTarget force_cast(_TSource&& v)
	{
		static_assert(sizeof(_TSource) >= sizeof(_TTarget), "Vector dimension must agree.");
		return reinterpret_cast<_TTarget&&>(std::move(v));
	}

	template <class _T>
	using wrap_simd_t = std::conditional_t<std::alignment_of<_T>::value == std::alignment_of<float>::value, const _T&, _T>;

	template <class _TTarget, class _TSource>
	inline wrap_simd_t<_TTarget> vector_cast(const _TSource& v)
	{
		return force_cast<_TTarget>(v);
	}
	template <>
	FORCEINLINE wrap_simd_t<btVector3> vector_cast<btVector3, DirectX::XMVECTOR>(const DirectX::XMVECTOR& v)
	{
		btVector3 btv;
		btv.mVec128 = v;
		return btv;
	}

	template <>
	inline wrap_simd_t<btVector3> vector_cast<btVector3, DirectX::Vector3>(const DirectX::Vector3& v)
	{
		return btVector3(v.x, v.y, v.z);
	}

	template <>
	inline wrap_simd_t<btVector4> vector_cast<btVector4, DirectX::Vector4>(const DirectX::Vector4& v)
	{
		return btVector4(v.x, v.y, v.z, v.w);
	}

	template <>
	inline wrap_simd_t<btQuaternion> vector_cast<btQuaternion, DirectX::Quaternion>(const DirectX::Quaternion& v)
	{
		return btQuaternion(v.x,v.y,v.z,v.w);
	}

	template <>
	inline wrap_simd_t<btVector3> vector_cast<btVector3, DirectX::XMFLOAT3>(const DirectX::XMFLOAT3& v)
	{
		return btVector3(v.x,v.y,v.z);
	}

	template <>
	inline wrap_simd_t<btVector3> vector_cast<btVector3, DirectX::XMFLOAT4>(const DirectX::XMFLOAT4& v)
	{
		return btVector4(v.x, v.y, v.z, v.w);
	}

	template <>
	inline wrap_simd_t<btQuaternion> vector_cast<btQuaternion, DirectX::XMFLOAT4>(const DirectX::XMFLOAT4& v)
	{
		return btQuaternion(v.x, v.y, v.z, v.w);
	}

	class PhysicalRigid : virtual public DirectX::IRigid
	{
	public:
		PhysicalRigid();

		~PhysicalRigid();

		bool Disable();
		bool Enable(const std::shared_ptr<btDynamicsWorld> &pWorld = nullptr);
		bool IsEnabled() const;

		btRigidBody* GetBulletRigid() { return m_pRigidBody.get(); }
		btCollisionShape* GetBulletShape() { return m_pShape.get(); }
		const btRigidBody* GetBulletRigid() const { return m_pRigidBody.get(); }
		const btCollisionShape* GetBulletShape() const { return m_pShape.get(); }

		void InitializePhysics(const std::shared_ptr<btDynamicsWorld> &pWorld, const std::shared_ptr<btCollisionShape>& pShape, float mass, const DirectX::Vector3 & Pos = DirectX::Vector3::Zero, const DirectX::Quaternion & Rot = DirectX::Quaternion::Identity);

		// Inherited via IRigid
		virtual DirectX::Vector3 GetPosition() const override;
		virtual void SetPosition(const DirectX::Vector3 & p) override;
		virtual DirectX::Quaternion GetOrientation() const override;
		virtual void SetOrientation(const DirectX::Quaternion & q) override;
		virtual DirectX::Vector3 GetScale() const override;
		virtual void SetScale(const DirectX::Vector3 & s) override;

	protected:
		bool								m_IsEnabled;
		std::shared_ptr<btDynamicsWorld>	m_pWorld;
		std::unique_ptr<btRigidBody>		m_pRigidBody;
		std::shared_ptr<btCollisionShape>	m_pShape;
		mutable DirectX::Vector3			Scale;
		mutable DirectX::Vector3			Position;
		mutable DirectX::Quaternion			Orientation;
	};

	namespace Bullet
	{
		class CollisionShape
		{

		};
		class RigidObject : public btRigidBody
		{
		public:
			RigidObject(const std::shared_ptr<btCollisionShape>& pShape, float mass, const DirectX::Vector3 & Pos = DirectX::Vector3::Zero, const DirectX::Quaternion & Rot = DirectX::Quaternion::Identity);
			bool Disable();
			bool Enable();

			DirectX::Vector3 GetPosition() const;
			DirectX::Quaternion GetOrientation() const;
			DirectX::Vector3 GetScale() const;
			DirectX::IsometricTransform GetWorldTransform() const;
			DirectX::XMMATRIX GetWorldTransformMatrix() const;

		private:
			btDynamicsWorld	*pWorld;
			std::shared_ptr<btCollisionShape>	m_pShape;
		};

		class DynamicObject : RigidObject
		{
		};

		class KinmeticObject : RigidObject
		{
			//void SetTransform(const DirectX::RigidTransform& transform)
			//{
			//	btTransform btrans;
			//	btrans.setOrigin(transform.Translation);
			//	btrans.setRotation(transform.Rotation);
			//	getMotionState()->setWorldTransform()
			//}
			//void SetPosition(const DirectX::Vector3& pos)
			//{
			//}
			//void SetOrientation(const DirectX::Quaternion orientation)
			//{}
		};

		class GhostObject : public btPairCachingGhostObject
		{
			std::vector<btBroadphasePair*> GetCollisionPairs() {
				std::vector<btBroadphasePair*> collisionPairs;
				btManifoldArray   manifoldArray;
				btBroadphasePairArray& pairArray = this->getOverlappingPairCache()->getOverlappingPairArray();
				int numPairs = pairArray.size();
				for (int i = 0; i < numPairs; i++)
				{
					manifoldArray.clear();

					const btBroadphasePair& pair = pairArray[i];

					//unless we manually perform collision detection on this pair, the contacts are in the dynamics world paircache:
					btBroadphasePair* collisionPair = pWorld->getPairCache()->findPair(pair.m_pProxy0, pair.m_pProxy1);
					if (!collisionPair)
						continue;

					// Details to check the contactiong points
					if (collisionPair->m_algorithm)
						collisionPair->m_algorithm->getAllContactManifolds(manifoldArray);

					if (manifoldArray.size() > 0)
						collisionPairs.push_back(collisionPair);

					//for (int j = 0; j < manifoldArray.size(); j++)
					//{
					//	btPersistentManifold* manifold = manifoldArray[j];
					//	btScalar directionSign = (manifold->getBody0() == this) ? btScalar(-1.0) : btScalar(1.0);
					//	for (int p = 0; p < manifold->getNumContacts(); p++)
					//	{
					//		const btManifoldPoint&pt = manifold->getContactPoint(p);
					//		if (pt.getDistance() < 0.f)
					//		{
					//			const btVector3& ptA = pt.getPositionWorldOnA();;
					//			const btVector3& ptB = pt.getPositionWorldOnB();
					//			const btVector3& normalOnB = pt.m_normalWorldOnB;
					//			/// work here
					//		}
					//	}
					//}
				}
				return collisionPairs;
			}
			btCollisionWorld	*pWorld;
		};
	}


	//class PhysicalGeometryModel : public DirectX::Scene::IGeometryModel, public DirectX::Scene::IRigidLocalMatrix , public PhysicalRigid
	//{
	//	void InitializePhysicalRigid(float mass = 1.0f);
	//	static void CreateFromObjFile();
	//};
}