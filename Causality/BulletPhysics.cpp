#include "pch_bullet.h"
#include "BulletPhysics.h"
#include <type_traits>

using DirectX::Vector3;

Causality::PhysicalRigid::PhysicalRigid()
	: m_pWorld(nullptr), m_pRigidBody(nullptr), m_IsEnabled(false)
{ }

Causality::PhysicalRigid::~PhysicalRigid()
{
	Disable();
	m_pWorld = nullptr;
}

bool Causality::PhysicalRigid::Disable()
{
	if (m_IsEnabled && m_pWorld != nullptr)
	{
		m_pWorld->removeRigidBody(m_pRigidBody.get());
		m_IsEnabled = false;
		return true;
	}
	return false;
}

bool Causality::PhysicalRigid::Enable(const std::shared_ptr<btDynamicsWorld> &pWorld)
{
	if (m_IsEnabled && pWorld != m_pWorld)
	{
		Disable();
	}
	if (!m_IsEnabled && pWorld != nullptr)
	{
		m_pWorld = pWorld;
	}
	if (!m_IsEnabled && m_pWorld != nullptr)
	{
		m_pWorld->addRigidBody(m_pRigidBody.get());
		m_IsEnabled = true;
		return true;
	}
	return false;
}

bool Causality::PhysicalRigid::IsEnabled() const
{
	return m_IsEnabled;
}

void Causality::PhysicalRigid::InitializePhysics(const std::shared_ptr<btDynamicsWorld> &pWorld, const std::shared_ptr<btCollisionShape>& pShape, float mass, const DirectX::Vector3 & Pos, const DirectX::Quaternion & Rot)
{
	if (pShape != nullptr)
		m_pShape = pShape;
	btVector3 minbox, maxbox;
	m_pShape->getAabb(btTransform::getIdentity(), minbox, maxbox);
	m_pWorld = pWorld;
	btDefaultMotionState* initalState =
		new btDefaultMotionState(btTransform( vector_cast<btQuaternion>(Rot), vector_cast<btVector3>(Pos)),btTransform(btQuaternion::getIdentity(), 0.5*(minbox + maxbox)));
	btVector3 inertia(0, 0, 0);
	m_pShape->calculateLocalInertia(mass, inertia);
	btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(mass, initalState, m_pShape.get(), inertia);
	m_pRigidBody = std::make_unique<btRigidBody>(rigidBodyCI);
	Enable();
}

DirectX::Vector3 Causality::PhysicalRigid::GetPosition() const
{
	Position = vector_cast<Vector3>(m_pRigidBody->getCenterOfMassPosition());
	return Position;
}

void Causality::PhysicalRigid::SetPosition(const DirectX::Vector3 & p)
{
	btTransform transform = m_pRigidBody->getWorldTransform();
	transform.setOrigin(vector_cast<btVector3>(p));
	m_pRigidBody->setWorldTransform(transform);
}

DirectX::Quaternion Causality::PhysicalRigid::GetOrientation() const
{
	Orientation = vector_cast<DirectX::Quaternion>(m_pRigidBody->getOrientation());
	return Orientation;
}

void Causality::PhysicalRigid::SetOrientation(const DirectX::Quaternion & q)
{
	btTransform transform = m_pRigidBody->getWorldTransform();
	transform.setRotation(vector_cast<btQuaternion>(q));
	m_pRigidBody->setWorldTransform(transform);
}

DirectX::Vector3 Causality::PhysicalRigid::GetScale() const
{
	Scale = vector_cast<Vector3>(m_pShape->getLocalScaling());
	return Scale;
}

void Causality::PhysicalRigid::SetScale(const DirectX::Vector3 & s)
{
	m_pShape->setLocalScaling(vector_cast<btVector3>(s));
}

//void Causality::PhysicalGeometryModel::InitializePhysicalRigid(float mass)
//{
//	btTransform trans;
//	auto pShape = new btCompoundShape();
//	m_pShape.reset(pShape);
//
//	//trans.setOrigin(vector_cast<btVector3>(BoundOrientedBox.Center));
//	//trans.setRotation(vector_cast<btQuaternion>(BoundOrientedBox.Orientation));
//	//pShape->addChildShape(trans, new btBoxShape(vector_cast<btVector3>(BoundOrientedBox.Extents)));
//
//	for (const auto& part : Parts)
//	{
//		trans.setOrigin(vector_cast<btVector3>(part.BoundOrientedBox.Center));
//		trans.setRotation(vector_cast<btQuaternion>(part.BoundOrientedBox.Orientation));
//		pShape->addChildShape(trans, new btBoxShape(vector_cast<btVector3>(part.BoundOrientedBox.Extents)));
//	}
//}
