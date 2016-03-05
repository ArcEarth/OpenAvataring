#include "pch_bcl.h"
#include "SceneObject.h"
#include "Scene.h"

using namespace Causality;
using namespace DirectX;

REGISTER_SCENE_OBJECT_IN_PARSER(scene_object, SceneObject);

SceneObject::~SceneObject()
{
	int *p = nullptr;
}

SceneObject::SceneObject() {
	m_IsEnabled = true;
	m_IsStatic = false;
	m_TransformDirty = false;
}


Quaternion ParseRotation(const char* store)
{
	string rotstr;
	Quaternion rotation;
	if (rotstr[0] == '[')
	{
		Matrix4x4 mat = Matrix4x4::Identity;
		sscanf_s(rotstr.c_str(), "[%f,%f,%f;%f,%f,%f;%f,%f,%f]",
			&mat(0, 0), &mat(0, 1), &mat(0, 2),
			&mat(1, 0), &mat(1, 1), &mat(1, 2),
			&mat(2, 0), &mat(2, 1), &mat(2, 2)
			);
		auto m = XMLoad(mat);
		auto q = XMQuaternionRotationMatrix(m);
		rotation = q;
	}
	else if (rotstr[0] == '{') // Quaternion
	{
		Quaternion q;
		sscanf_s(rotstr.c_str(), "{%f,%f,%f,%f}", &q.x, &q.y, &q.z, &q.w);
		rotation = q;
	}
	else
	{
		Vector3 e;
		if (rotstr[0] == '<')
			sscanf_s(rotstr.c_str(), "<%f,%f,%f>", &e.x, &e.y, &e.z);
		else
			sscanf_s(rotstr.c_str(), "%f,%f,%f", &e.x, &e.y, &e.z);
		rotation = Quaternion::CreateFromYawPitchRoll(e.y, e.x, e.z);
	}
	return rotation;
}


void SceneObject::Parse(const ParamArchive * store)
{
	GetParam(store, "name", Name);
	GetParam(store, "tag", Tag);

	Vector3 scale(1.0f);
	Vector3 pos;
	Vector3 eular;
	const char* rotStr = nullptr;

	if (GetParam(store, "position", pos))
		SetPosition(pos);
	else
		GetParam(store, "translation", m_Transform.LclTranslation);

	if (GetParam(store, "scale", scale))
		m_Transform.LclScaling = scale;

	if (GetParam(store, "orientation", rotStr))
		SetOrientation(ParseRotation(rotStr));
	else if (GetParam(store, "rotation", rotStr))
		m_Transform.LclRotation = ParseRotation(rotStr);
	m_TransformDirty = true;
}

void SceneObject::AddChild(SceneObject * child)
{
	if (child != nullptr)
	{
		std::lock_guard<std::mutex> guard(Scene->ContentMutex());
		auto oldParent = child->parent();
		append_children_back(child);
		child->OnParentChanged(child,oldParent);
		this->OnChildAdded(this, child);
	}
}

void SceneObject::Remove() { this->isolate(); }

void SceneObject::Update(time_seconds const & time_delta) {
	UpdateTransformsChildWard();
}

DirectX::XMMATRIX SceneObject::GlobalTransformMatrix() const
{
	return this->m_Transform.GlobalTransform().TransformMatrix();
	//if (parent() != nullptr)
	//{
	//	DirectX::XMMATRIX mat = parent()->GetPosition();
	//	mat *= TransformMatrix();
	//	return mat;
	//}
	//else
	//{
	//	return TransformMatrix();
	//}
}

void XM_CALLCONV SceneObject::Move(FXMVECTOR p)
{
	SetPosition((XMVECTOR)GetPosition() + XMVector3Rotate(p, GetOrientation()));
}

void XM_CALLCONV SceneObject::Rotate(FXMVECTOR q)
{
	SetOrientation(XMQuaternionMultiply(q, GetOrientation()));
}

void SceneObject::SetTransformDirty()
{
	if (!m_TransformDirty)
	{
		m_TransformDirty = true;
		for (auto& child : children())
		{
			child.SetTransformDirty();
		}
	}
}

Vector3 SceneObject::GetPosition() const {
	if (m_TransformDirty)
		UpdateTransformsParentWard();
	return m_Transform.GblTranslation;
}

Quaternion SceneObject::GetOrientation() const {
	if (m_TransformDirty)
		UpdateTransformsParentWard();
	return m_Transform.GblRotation;
}

Vector3 SceneObject::GetScale() const {
	if (m_TransformDirty)
		UpdateTransformsParentWard();
	return m_Transform.GblScaling;
}

void SceneObject::SetPosition(const Vector3 & p)
{
	if (!parent())
	{
		m_Transform.LclTranslation = m_Transform.GblTranslation = p;
		m_TransformDirty = false;
	}
	else
	{
		XMVECTOR refQ = parent()->GetOrientation(); // parent global orientation
		XMVECTOR V = parent()->GetPosition();
		XMVECTOR S = parent()->GetScale();
		V = XMLoad(p) - V;
		refQ = XMQuaternionConjugate(refQ);
		V = XMVector3Rotate(V, refQ); // V *= Inverse(ref.Orientation)
		V /= S;
		m_Transform.LclTranslation = V;
	}
	SetTransformDirty();
}

void SceneObject::SetOrientation(const Quaternion & q)
{
	if (!parent())
	{
		m_Transform.LclRotation = m_Transform.GblRotation = q;
		m_TransformDirty = false;
	}
	else
	{
		XMVECTOR refQ = parent()->GetOrientation(); // parent global orientation
		refQ = XMQuaternionConjugate(refQ);
		m_Transform.LclRotation = XMQuaternionMultiply(q, refQ);
	}
	SetTransformDirty();
}

void SceneObject::SetScale(const Vector3 & s)
{
	if (!parent())
	{
		m_Transform.LclScaling = m_Transform.GblScaling = s;
	}
	else
	{
		XMVECTOR refS = parent()->GetScale();
		m_Transform.LclScaling = XMLoad(s) / refS;
	}
	SetTransformDirty();
}

void SceneObject::SetLocalTransform(const DirectX::IsometricTransform & lcl)
{
	m_Transform.LocalTransform() = lcl;
	SetTransformDirty();
}

const DirectX::IsometricTransform & SceneObject::GetGlobalTransform() const
{
	if (m_TransformDirty)
		UpdateTransformsParentWard();
	return m_Transform.GlobalTransform();
}

bool SceneObject::GetBoundingBox(BoundingBox & box) const
{
	if (has_child())
	{
		bool flag = true;
		for (auto& child : children())
		{
			if (flag)
				child.GetBoundingBox(box);
			else
			{
				BoundingBox tbox;
				child.GetBoundingBox(tbox);
				BoundingBox::CreateMerged(box, box, box);
			}
		}
	}
	return false;
}

bool SceneObject::GetBoundingGeometry(BoundingGeometry & geometry) const
{
	return false;
}

void SceneObject::UpdateTransformsParentWard() const
{
	if (!m_TransformDirty) return;
	const SceneObject *pObj = parent();

	if (pObj)
	{
		if (pObj->m_TransformDirty)
			pObj->UpdateTransformsParentWard();
		m_Transform.UpdateGlobalTransform(pObj->m_Transform);
	}
	else
		m_Transform.GlobalTransform() = m_Transform.LocalTransform();
	m_TransformDirty = false;
}

void SceneObject::UpdateTransformsChildWard()
{
	if (m_TransformDirty)
	{
		UpdateTransformsParentWard();
		for (auto& child : children())
		{
			child.UpdateTransformsChildWard();
		}
	}
}

SceneObjectParser::CreatorMapType & Causality::SceneObjectParser::Creators()
{
	static std::unique_ptr<CreatorMapType>
		g_pCreators;
	if (g_pCreators == nullptr)
		g_pCreators.reset(new CreatorMapType());
	return *g_pCreators;
}
