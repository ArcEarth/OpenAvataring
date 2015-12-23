#include "pch_bullet.h"
#include <iostream>
#include <numeric>
#include "Foregrounds.h"
#include "CausalityApplication.h"
#include "Common\PrimitiveVisualizer.h"
#include "Common\Extern\cpplinq.hpp"
#include <boost\format.hpp>

using namespace Causality;
using namespace DirectX;
using namespace DirectX::Scene;
using namespace std;
using namespace Platform;
using namespace Eigen;
using namespace concurrency;
using namespace DirectX::Visualizers;
extern wstring ResourcesDirectory;

//std::unique_ptr<DirectX::GeometricPrimitive> HandPhysicalModel::s_pCylinder;
//std::unique_ptr<DirectX::GeometricPrimitive> HandPhysicalModel::s_pSphere;

const static wstring SkyBoxTextures[6] = {
	ResourcesDirectory + wstring(L"Textures\\SkyBox\\GrimmNight\\Right.dds"),
	ResourcesDirectory + wstring(L"Textures\\SkyBox\\GrimmNight\\Left.dds"),
	ResourcesDirectory + wstring(L"Textures\\SkyBox\\GrimmNight\\Top.dds"),
	ResourcesDirectory + wstring(L"Textures\\SkyBox\\GrimmNight\\Bottom.dds"),
	ResourcesDirectory + wstring(L"Textures\\SkyBox\\GrimmNight\\Front.dds"),
	ResourcesDirectory + wstring(L"Textures\\SkyBox\\GrimmNight\\Back.dds"),
};

//std::unique_ptr<btBroadphaseInterface>               pBroadphase = nullptr;
//// Set up the collision configuration and dispatcher
//std::unique_ptr<btDefaultCollisionConfiguration>     pCollisionConfiguration = nullptr;
//std::unique_ptr<btCollisionDispatcher>               pDispatcher = nullptr;
//// The actual physics solver
//std::unique_ptr<btSequentialImpulseConstraintSolver> pSolver = nullptr;

std::queue<std::unique_ptr<WorldBranch>> WorldBranch::BranchPool;


float ShapeSimiliarity(const Eigen::VectorXf& v1, const Eigen::VectorXf& v2)
{
	auto v = v1 - v2;
	//dis = sqrt(v.dot(v);
	//1.414 - dis;
	auto theta = XMScalarACosEst(v1.dot(v2) * XMScalarReciprocalSqrtEst(v1.dot(v1) * v2.dot(v2))); // Difference in angular [0,pi/4]
	auto rhlo = abs(sqrt(v1.dot(v1)) - sqrtf(v2.dot(v2)));	// Difference in length [0,sqrt(2)]
	return 1.0f - 0.5f * (0.3f * rhlo / sqrtf(2.0f) + 0.7f * theta / (XM_PIDIV4));
}

Causality::WorldScene::WorldScene(const std::shared_ptr<DirectX::DeviceResources>& pResouce, const DirectX::ILocatable* pCamera)
	: States(pResouce->GetD3DDevice())
	, m_pCameraLocation(pCamera)
{
	m_HaveHands = false;
	m_showTrace = true;
	LoadAsync(pResouce->GetD3DDevice());
}

WorldScene::~WorldScene()
{
}

const float fingerRadius = 0.006f;
const float fingerLength = 0.02f;

class XmlModelLoader
{
};

WorldBranch::WorldBranch()
{
	pBroadphase.reset(new btDbvtBroadphase());

	// Set up the collision configuration and dispatcher
	pCollisionConfiguration.reset(new btDefaultCollisionConfiguration());
	pDispatcher.reset(new btCollisionDispatcher(pCollisionConfiguration.get()));

	// The actual physics solver
	pSolver.reset(new btSequentialImpulseConstraintSolver());
	// The world.
	pDynamicsWorld.reset(new btDiscreteDynamicsWorld(pDispatcher.get(), pBroadphase.get(), pSolver.get(), pCollisionConfiguration.get()));
	pDynamicsWorld->setGravity(btVector3(0, -1.0f, 0));

	IsEnabled = false;
}

void Causality::WorldScene::LoadAsync(ID3D11Device* pDevice)
{

	m_loadingComplete = false;
	pBackground = nullptr;

	//CD3D11_DEFAULT d;
	//CD3D11_RASTERIZER_DESC Desc(d);
	//Desc.MultisampleEnable = TRUE;
	//ThrowIfFailed(pDevice->CreateRasterizerState(&Desc, &pRSState));

	concurrency::task<void> load_models([this, pDevice]() {
		{
			lock_guard<mutex> guard(m_RenderLock);
			WorldBranch::InitializeBranchPool(1);
			WorldTree = WorldBranch::DemandCreate("Root");

			//std::vector<AffineTransform> subjectTrans(30);
			//subjectTrans.resize(20);
			//for (size_t i = 0; i < 20; i++)
			//{
			//	subjectTrans[i].Scale = XMVectorReplicate(1.1f + 0.15f * i);// XMMatrixTranslation(0, 0, i*(-150.f));
			//}
			//WorldTree->Fork(subjectTrans);

			WorldTree->Enable(DirectX::AffineTransform::Identity());
		}

		//m_pFramesPool.reset(new WorldBranchPool);
		//m_pFramesPool->Initialize(30);
		//{
		//	lock_guard<mutex> guard(m_RenderLock);
		//	for (size_t i = 0; i < 30; i++)
		//	{
		//		m_StateFrames.push_back(m_pFramesPool->DemandCreate());
		//		auto pFrame = m_StateFrames.back();
		//		//pFrame->Initialize();
		//		pFrame->SubjectTransform.Scale = XMVectorReplicate(1.0f + 0.1f * i);// XMMatrixTranslation(0, 0, i*(-150.f));
		//	}
		//	m_StateFrames.front()->Enable(DirectX::AffineTransform::Identity());
		//}

		auto Directory = App::Current()->GetResourcesDirectory();
		auto ModelDirectory = Directory / "Models";
		auto TextureDirectory = Directory / "Textures";
		auto texDir = TextureDirectory.wstring();
		pEffect = std::make_shared<BasicEffect>(pDevice);
		pEffect->SetVertexColorEnabled(false);
		pEffect->SetTextureEnabled(true);
		//pEffect->SetLightingEnabled(true);
		pEffect->EnableDefaultLighting();
		{
			void const* shaderByteCode;
			size_t byteCodeLength;
			pEffect->GetVertexShaderBytecode(&shaderByteCode, &byteCodeLength);
			pInputLayout = CreateInputLayout<VertexPositionNormalTexture>(pDevice, shaderByteCode, byteCodeLength);
		}

		//pBackground = std::make_unique<SkyDome>(pDevice, SkyBoxTextures);

		auto sceneFile = Directory / "Foregrounds.xml";
		tinyxml2::XMLDocument sceneDoc;
		sceneDoc.LoadFile(sceneFile.string().c_str());
		auto scene = sceneDoc.FirstChildElement("scene");
		auto node = scene->FirstChildElement();
		while (node)
		{
			if (!strcmp(node->Name(), "obj"))
			{
				auto path = node->Attribute("src");
				if (path != nullptr && strlen(path) != 0)
				{
					auto pModel = std::make_shared<ShapedGeomrtricModel>();
					GeometryModel::CreateFromObjFile(pModel.get(), pDevice, (ModelDirectory / path).wstring(), texDir);

					XMFLOAT3 v = pModel->BoundOrientedBox.Extents;
					v.y /= v.x;
					v.z /= v.x;
					m_ModelFeatures[pModel->Name] = Eigen::Vector2f(v.y, v.z);
					std::cout << "[Model] f(" << pModel->Name << ") = " << m_ModelFeatures[pModel->Name] << std::endl;

					float scale = 1.0f;
					float mass = 1.0f;
					Vector3 pos;

					auto attr = node->Attribute("scale");
					if (attr != nullptr)
					{
						stringstream ss(attr);

						ss >> scale;
						//model->SetScale(XMVectorReplicate(scale));
					}
					attr = node->Attribute("position");
					if (attr != nullptr)
					{
						stringstream ss(attr);

						char ch;
						ss >> pos.x >> ch >> pos.y >> ch >> pos.z;
						//model->SetPosition(pos);
					}

					attr = node->Attribute("mass");
					if (attr)
						mass = (float) atof(attr);

					AddObject(pModel, mass, pos, DirectX::Quaternion::Identity, DirectX::Vector3(scale));

					//auto pShape = model->CreateCollisionShape();
					//pShape->setLocalScaling(btVector3(scale, scale, scale));
					//btVector3 minb, maxb;
					//model->InitializePhysics(pDynamicsWorld, pShape, mass, pos, XMQuaternionIdentity());
					//model->GetBulletRigid()->setFriction(1.0f);
					//{
					//	std::lock_guard<mutex> guard(m_RenderLock);
					//	Models.push_back(model);
					//}
				}
			}
			else if (!strcmp(node->Name(), "cube"))
			{
				Vector3 extent(1.0f);
				Vector3 pos;
				Color color(255, 255, 255, 255);
				string name("cube");
				float mass = 1.0f;
				auto attr = node->Attribute("extent");
				if (attr != nullptr)
				{
					stringstream ss(attr);
					char ch;
					ss >> extent.x >> ch >> extent.y >> ch >> extent.z;
				}
				attr = node->Attribute("position");
				if (attr != nullptr)
				{
					stringstream ss(attr);

					char ch;
					ss >> pos.x >> ch >> pos.y >> ch >> pos.z;
				}
				attr = node->Attribute("color");
				if (attr != nullptr)
				{
					stringstream ss(attr);

					char ch;
					ss >> color.x >> ch >> color.y >> ch >> color.z;
					if (!ss.eof())
						ss >> ch >> color.w;
					color = color.ToVector4() / 255;
					color.Saturate();
				}
				attr = node->Attribute("name");
				if (attr)
					name = attr;

				attr = node->Attribute("mass");
				if (attr)
					mass = (float) atof(attr);

				auto pModel = make_shared<CubeModel>(name, extent, (XMVECTOR) color);

				XMFLOAT3 v = pModel->BoundOrientedBox.Extents;
				v.y /= v.x;
				v.z /= v.x;
				m_ModelFeatures[pModel->Name] = Eigen::Vector2f(v.y, v.z);

				AddObject(pModel, mass, pos, DirectX::Quaternion::Identity, DirectX::Vector3::One);
				//auto pShape = pModel->CreateCollisionShape();
				//pModel->InitializePhysics(nullptr, pShape, mass, pos);
				//pModel->Enable(pDynamicsWorld);
				//pModel->GetBulletRigid()->setFriction(1.0f);
				//{
				//	std::lock_guard<mutex> guard(m_RenderLock);
				//	Models.push_back(pModel);
				//}
			}
			node = node->NextSiblingElement();
		}

		m_loadingComplete = true;
	});
}

void Causality::WorldScene::SetViewIdenpendntCameraPosition(const DirectX::ILocatable * pCamera)
{
	m_pCameraLocation = pCamera;
}

void Causality::WorldScene::Render(ID3D11DeviceContext * pContext)
{
	if (pBackground)
		pBackground->Render(pContext);

	{
		pContext->IASetInputLayout(pInputLayout.Get());
		auto pAWrap = States.AnisotropicWrap();
		pContext->PSSetSamplers(0, 1, &pAWrap);
		pContext->RSSetState(pRSState.Get());
		std::lock_guard<mutex> guard(m_RenderLock);
		
		BoundingOrientedBox modelBox;
		using namespace cpplinq;

		// Render models
		for (const auto& model : Models)
		{
			auto superposition = ModelStates[model->Name];
			for (const auto& state : superposition)
			{
				auto mat = state.TransformMatrix();
				model->LocalMatrix = state.TransformMatrix(); //.first->GetRigidTransformMatrix();
				model->Opticity = state.Probability; //state.second;

				model->BoundOrientedBox.Transform(modelBox, mat);
				// Render if in the view frustum

				if (ViewFrutum.Contains(modelBox) != ContainmentType::DISJOINT)
					model->Render(pContext, pEffect.get());
			}
		}

		for (const auto& branch : WorldTree->leaves())
		{
				//Subjects
				for (const auto& item : branch.Subjects)
				{
					if (item.second)
					{
						item.second->Opticity = branch.Liklyhood();
						item.second->Render(pContext, nullptr);
					}
				}
		}
	}



	g_PrimitiveDrawer.Begin();

	Vector3 conners[8];

	ViewFrutum.GetCorners(conners);
	DrawBox(conners, Colors::Pink);
	BoundingOrientedBox obox;
	BoundingBox box;
	{
		//Draw axias
		DrawAxis();
		//g_PrimitiveDrawer.DrawQuad({ 1.0f,0,1.0f }, { -1.0f,0,1.0f }, { -1.0f,0,-1.0f }, { 1.0f,0,-1.0f }, Colors::Pink);


		auto& fh = m_HandDescriptionFeature;

		{
			std::lock_guard<mutex> guard(m_RenderLock);
			auto s = Models.size();
			if (m_HaveHands)
				std::cout << "Detail Similarity = {";
			for (size_t i = 0; i < s; i++)
			{
				const auto& model = Models[i];
				obox = model->GetOrientedBoundingBox();
				if (ViewFrutum.Contains(obox) != ContainmentType::DISJOINT)
				{
					obox.GetCorners(conners);
					DrawBox(conners, DirectX::Colors::DarkGreen);
				}
				if (m_HaveHands)
				{
					auto fm = m_ModelFeatures[model->Name];

					auto similarity = ShapeSimiliarity(fm, fh);
					m_ModelDetailSimilarity[model->Name] = similarity;
					std::cout << model->Name << ':' << similarity << " , ";
					Color c = Color::Lerp({ 1,0,0 }, { 0,1,0 }, similarity);
					for (size_t i = 0; i < 8; i++)
					{
						g_PrimitiveDrawer.DrawSphere(conners[i], 0.005f * similarity, c);
					}
				}
				auto pModel = dynamic_cast<CompositeModel*>(model.get());
				XMMATRIX transform = model->GetWorldMatrix();
				if (pModel)
				{
					for (const auto& part : pModel->Parts)
					{
						part->BoundOrientedBox.Transform(obox, transform);
						if (ViewFrutum.Intersects(obox))
						{
							obox.GetCorners(conners);
							DrawBox(conners, DirectX::Colors::Orange);
						}
					}
				}
			}
		}
		if (m_HaveHands)
			std::cout << '}' << std::endl;
	}

	if (m_HaveHands)
	{

		//for (auto& pRigid : m_HandRigids)
		//{
		//	g_PrimitiveDrawer.DrawSphere(pRigid->GetPosition(), 0.01f, Colors::Pink);
		//}

		auto pCamera = App::Current()->GetPrimaryCamera();

		XMMATRIX leap2world = m_FrameTransform;// XMMatrixScalingFromVector(XMVectorReplicate(0.001f)) * XMMatrixTranslation(0,0,-1.0) * XMMatrixRotationQuaternion(pCamera->GetOrientation()) * XMMatrixTranslationFromVector((XMVECTOR)pCamera->GetPosition());
		std::lock_guard<mutex> guard(m_HandFrameMutex);
		for (const auto& hand : m_Frame.hands())
		{
			auto palmPosition = XMVector3Transform(hand.palmPosition().toVector3<Vector3>(), leap2world);
			g_PrimitiveDrawer.DrawSphere(palmPosition, 0.02f, Colors::YellowGreen);
			//for (const auto& finger : hand.fingers())
			//{
			//	for (size_t i = 0; i < 4; i++)
			//	{
			//		const auto & bone = finger.bone((Leap::Bone::Type)i);
			//		XMVECTOR bJ = XMVector3Transform(bone.prevJoint().toVector3<Vector3>(), leap2world);
			//		XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), leap2world);
			//		//if (i == 0)
			//		//	g_PrimitiveDrawer.DrawSphere(bJ, 0.01f, DirectX::Colors::Lime);

			//		//// The unit of leap is millimeter
			//		//g_PrimitiveDrawer.DrawSphere(eJ, 0.01f, DirectX::Colors::Lime);
			//		g_PrimitiveDrawer.DrawLine(bJ, eJ, DirectX::Colors::White);
			//	}
			//}
		}

		// NOT VALIAD!~!!!!!
		//if (m_HandTrace.size() > 0 && m_showTrace)
		//{
		//	std::lock_guard<mutex> guard(m_RenderLock);
		//	//auto pJoints = m_HandTrace.linearize();
		//	m_CurrentHandBoundingBox.GetCorners(conners);
		//	DrawBox(conners, Colors::LimeGreen);
		//	m_HandTraceBoundingBox.GetCorners(conners);
		//	DrawBox(conners, Colors::YellowGreen);
		//	m_HandTraceModel.Primitives.clear();
		//	for (int i = m_HandTrace.size() - 1; i >= std::max<int>(0, (int) m_HandTrace.size() - TraceLength); i--)
		//	{
		//		const auto& h = m_HandTrace[i];
		//		float radius = (i + 1 - std::max<int>(0, m_HandTrace.size() - TraceLength)) / (std::min<float>(m_HandTrace.size(), TraceLength));
		//		for (size_t j = 0; j < h.size(); j++)
		//		{
		//			//m_HandTraceModel.Primitives.emplace_back(h[j], 0.02f);
		//			g_PrimitiveDrawer.DrawSphere(h[j], 0.005f * radius, Colors::LimeGreen);
		//		}
		//	}
		//}
	}
	g_PrimitiveDrawer.End();

	//if (m_HandTrace.size() > 0)
	//{
	//	if (!pBatch)
	//	{
	//		pBatch = std::make_unique<PrimitiveBatch<VertexPositionNormal>>(pContext, 204800,40960);
	//	}
	//	m_HandTraceModel.SetISO(0.33333f);
	//	m_HandTraceModel.Update();
	//	m_HandTraceModel.Tessellate(m_HandTraceVertices, m_HandTraceIndices, 0.005f);
	//	pBatch->Begin();
	//	pEffect->SetDiffuseColor(Colors::LimeGreen);
	//	//pEffect->SetEmissiveColor(Colors::LimeGreen);
	//	pEffect->SetTextureEnabled(false);
	//	pEffect->SetWorld(XMMatrixIdentity());
	//	pEffect->Apply(pContext);
	//	pBatch->DrawIndexed(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST, m_HandTraceIndices.data(), m_HandTraceIndices.size(), m_HandTraceVertices.data(), m_HandTraceVertices.size());
	//	pBatch->End();
	//}

}

void Causality::WorldScene::DrawAxis()
{
	g_PrimitiveDrawer.DrawSphere({ 0,0,0,0.02 }, Colors::Red);
	g_PrimitiveDrawer.DrawLine({ -5,0,0 }, { 5,0,0 }, Colors::Red);
	g_PrimitiveDrawer.DrawLine({ 0,-5,0 }, { 0,5,0 }, Colors::Green);
	g_PrimitiveDrawer.DrawLine({ 0,0,-5 }, { 0,0,5 }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ 5.05f,0,0 }, { 4.95,0.05,0 }, { 4.95,-0.05,0 }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ 5.05f,0,0 }, { 4.95,-0.05,0 }, { 4.95,0.05,0 }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ 5.05f,0,0 }, { 4.95,0,0.05 }, { 4.95,0,-0.05 }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ 5.05f,0,0 }, { 4.95,0,-0.05 }, { 4.95,0,0.05 }, Colors::Red);
	g_PrimitiveDrawer.DrawTriangle({ 0,5.05f,0 }, { -0.05,4.95,0 }, { 0.05,4.95,0 }, Colors::Green);
	g_PrimitiveDrawer.DrawTriangle({ 0,5.05f,0 }, { 0.05,4.95,0 }, { -0.05,4.95,0 }, Colors::Green);
	g_PrimitiveDrawer.DrawTriangle({ 0,5.05f,0 }, { 0.0,4.95,-0.05 }, { 0,4.95,0.05 }, Colors::Green);
	g_PrimitiveDrawer.DrawTriangle({ 0,5.05f,0 }, { 0.0,4.95,0.05 }, { 0,4.95,-0.05 }, Colors::Green);
	g_PrimitiveDrawer.DrawTriangle({ 0,0,5.05f }, { 0.05,0,4.95 }, { -0.05,0,4.95 }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ 0,0,5.05f }, { -0.05,0,4.95 }, { 0.05,0,4.95 }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ 0,0,5.05f }, { 0,0.05,4.95 }, { 0,-0.05,4.95 }, Colors::Blue);
	g_PrimitiveDrawer.DrawTriangle({ 0,0,5.05f }, { 0,-0.05,4.95 }, { 0,0.05,4.95 }, Colors::Blue);

}

void Causality::WorldScene::DrawBox(DirectX::SimpleMath::Vector3  conners [], DirectX::CXMVECTOR color)
{
	g_PrimitiveDrawer.DrawLine(conners[0], conners[1], color);
	g_PrimitiveDrawer.DrawLine(conners[1], conners[2], color);
	g_PrimitiveDrawer.DrawLine(conners[2], conners[3], color);
	g_PrimitiveDrawer.DrawLine(conners[3], conners[0], color);

	g_PrimitiveDrawer.DrawLine(conners[3], conners[7], color);
	g_PrimitiveDrawer.DrawLine(conners[2], conners[6], color);
	g_PrimitiveDrawer.DrawLine(conners[1], conners[5], color);
	g_PrimitiveDrawer.DrawLine(conners[0], conners[4], color);

	g_PrimitiveDrawer.DrawLine(conners[4], conners[5], color);
	g_PrimitiveDrawer.DrawLine(conners[5], conners[6], color);
	g_PrimitiveDrawer.DrawLine(conners[6], conners[7], color);
	g_PrimitiveDrawer.DrawLine(conners[7], conners[4], color);

}

void XM_CALLCONV Causality::WorldScene::UpdateViewMatrix(DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection)
{
	if (pEffect)
	{
		pEffect->SetView(view);
		pEffect->SetProjection(projection);
	}
	if (pBackground)
	{
		pBackground->UpdateViewMatrix(view,projection);
	}
	g_PrimitiveDrawer.SetView(view);
	g_PrimitiveDrawer.SetProjection( projection);
	
	// BoundingFrustum is assumpt Left-Handed
	BoundingFrustumExtension::CreateFromMatrixRH(ViewFrutum, projection);
	//BoundingFrustum::CreateFromMatrix(ViewFrutum, projection);
	// Fix the RH-projection matrix
	//XMStoreFloat4((XMFLOAT4*) &ViewFrutum.RightSlope, -XMLoadFloat4((XMFLOAT4*) &ViewFrutum.RightSlope));
	//XMStoreFloat2((XMFLOAT2*) &ViewFrutum.Near, -XMLoadFloat2((XMFLOAT2*) &ViewFrutum.Near));
	//ViewFrutum.LeftSlope = -ViewFrutum.LeftSlope;
	//ViewFrutum.RightSlope = -ViewFrutum.RightSlope;
	//ViewFrutum.TopSlope = -ViewFrutum.TopSlope;
	//ViewFrutum.BottomSlope = -ViewFrutum.BottomSlope;
	//ViewFrutum.Near = -ViewFrutum.Near;
	//ViewFrutum.Far = -ViewFrutum.Far;
	XMVECTOR det;
	auto invView = view;
	//invView.r[2] = -invView.r[2];
	invView = XMMatrixInverse(&det, invView);
	//invView.r[2] = -invView.r[2];
	//XMVECTOR temp = invView.r[2];
	//invView.r[2] = invView.r[1];
	//invView.r[1] = temp;
	ViewFrutum.Transform(ViewFrutum,invView);
	// Fix the RH-inv-view-matrix to LH-equalulent by swap row-y with row-z
	//XMStoreFloat3(&ViewFrutum.Origin, invView.r[3]);

	//XMStoreFloat4(&ViewFrutum.Orientation, XMQuaternionRotationMatrix(invView));;

	//for (const auto& item : m_HandModels)
	//{
	//	if (item.second)
	//		item.second->UpdateViewMatrix(view);
	//}
}

//void XM_CALLCONV Causality::WorldScene::UpdateProjectionMatrix(DirectX::FXMMATRIX projection)
//{
//	if (pEffect)
//		pEffect->SetProjection(projection);
//	if (pBackground)
//		pBackground->UpdateProjectionMatrix(projection);
//	
//	g_PrimitiveDrawer.SetProjection(projection);
//}

void Causality::WorldScene::UpdateAnimation(StepTimer const & timer)
{
	{
		lock_guard<mutex> guard(m_RenderLock);
		using namespace cpplinq;
		using namespace std::placeholders;
		float stepTime = (float) timer.GetElapsedSeconds();
		WorldTree->Evolution(stepTime, m_Frame, m_FrameTransform);
		ModelStates = WorldTree->CaculateSuperposition();
	}

	if (m_HandTrace.size() > 0)
	{
		{ // Critia section
			std::lock_guard<mutex> guard(m_HandFrameMutex);
			const int plotSize = 45;
			BoundingOrientedBox::CreateFromPoints(m_CurrentHandBoundingBox, m_HandTrace.back().size(), m_HandTrace.back().data(), sizeof(Vector3));
			m_TracePoints.clear();
			Color color = Colors::LimeGreen;
			for (int i = m_HandTrace.size() - 1; i >= std::max(0, (int) m_HandTrace.size() - plotSize); i--)
			{
				const auto& h = m_HandTrace[i];
				//float radius = (i + 1 - std::max(0U, m_HandTrace.size() - plotSize)) / (std::min<float>(m_HandTrace.size(), plotSize));
				for (size_t j = 0; j < h.size(); j++)
				{
					//g_PrimitiveDrawer.DrawSphere(h[j], 0.005 * radius, color);
					m_TracePoints.push_back(h[j]);
				}
			}
			//for (const auto& pModel : Children)
			//{
			//	//btCollisionWorld::RayResultCallback
			//	auto pRigid = dynamic_cast<PhysicalGeometryModel*>(pModel.get());
			//	pRigid->GetBulletRigid()->checkCollideWith()
			//}
			//auto pBat = dynamic_cast<PhysicalGeometryModel*>(Children[0].get());
			//pBat->SetPosition(vector_cast<Vector3>(m_Frame.hands().frontmost().palmPosition()));
		}
		CreateBoundingOrientedBoxFromPoints(m_HandTraceBoundingBox, m_TracePoints.size(), m_TracePoints.data(), sizeof(Vector3));
		XMFLOAT3 v = m_HandTraceBoundingBox.Extents;
		m_HandDescriptionFeature = Eigen::Vector2f(v.y / v.x, v.z / v.x);

		// ASSUMPTION: Extends is sorted from!


		//XMMATRIX invTrans = XMMatrixAffineTransformation(g_XMOne / XMVectorReplicate(m_HandTraceBoundingBox.Extents.x), XMVectorZero(), XMQuaternionInverse(XMLoadFloat4(&m_HandTraceBoundingBox.Orientation)), -XMLoadFloat3(&m_HandTraceBoundingBox.Center));
		//int sampleCount = std::min<int>(m_TraceSamples.size(), TraceLength)*m_TraceSamples[0].size();
		//for (const auto& model : Children)
		//{
		//	auto pModel = dynamic_cast<Model*>(model.get());
		//	auto inCount = 0;
		//	if (pModel)
		//	{
		//		auto obox = model->GetOrientedBoundingBox();
		//		XMMATRIX fowTrans = XMMatrixAffineTransformation(XMVectorReplicate(obox.Extents.x), XMVectorZero(), XMQuaternionIdentity(), XMVectorZero());
		//		fowTrans = invTrans * fowTrans;
		//		auto pSample = m_TraceSamples.back().data() + m_TraceSamples.back().size()-1;
		//		for (size_t i = 0; i < sampleCount; i++)
		//		{
		//			const auto& point = pSample[-i];
		//			XMVECTOR p = XMVector3Transform(point, fowTrans);
		//			int j;
		//			for ( j = 0; j < pModel->Parts.size(); j++)
		//			{
		//				if (pModel->Parts[j].BoundOrientedBox.Contains(p))
		//					break;
		//			}
		//			if (j >= pModel->Parts.size())
		//				inCount++;
		//		}
		//	}
		//	m_ModelDetailSimilarity[model->Name] = (float) inCount / (float)sampleCount;
		//}



	}

	//if (pGroundRigid)
	//	pGroundRigid->setLinearVelocity({ 0,-1.0f,0 });



	//pDynamicsWorld->stepSimulation(timer.GetElapsedSeconds(), 10);

	//for (auto& obj : Children)
	//{
	//	auto s = (rand() % 1000) / 1000.0;
	//	obj->Rotate(XMQuaternionRotationRollPitchYaw(0, 0.5f * timer.GetElapsedSeconds(), 0));
	//}
}

void Causality::WorldScene::OnHandsTracked(const UserHandsEventArgs & e)
{
	m_HaveHands = true;
	//const auto& hand = e.sender.frame().hands().frontmost();
	//size_t i = 0;
	//
	//XMMATRIX leap2world = e.toWorldTransform;
	//for (const auto& finger : hand.fingers())
	//{
	//	XMVECTOR bJ = XMVector3Transform(finger.bone((Leap::Bone::Type)0).prevJoint().toVector3<Vector3>(), leap2world);
	//	auto pState = m_HandRigids[i]->GetBulletRigid()->getMotionState();
	//	auto transform = btTransform::getIdentity();
	//	transform.setOrigin(vector_cast<btVector3>(bJ));
	//	if (!pState)
	//	{
	//		pState = new btDefaultMotionState(transform);
	//		m_HandRigids[i]->GetBulletRigid()->setMotionState(pState);
	//	}
	//	else
	//		pState->setWorldTransform(transform);
	//	

	//	i++;
	//	for (size_t boneIdx = 0; boneIdx < 4; boneIdx++) // bone idx
	//	{
	//		const auto & bone = finger.bone((Leap::Bone::Type)boneIdx);
	//		XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), leap2world);

	//		auto pState = m_HandRigids[i]->GetBulletRigid()->getMotionState();
	//		auto transform = btTransform::getIdentity();
	//		transform.setOrigin(vector_cast<btVector3>(eJ));
	//		if (!pState)
	//		{
	//			pState = new btDefaultMotionState(transform);
	//			m_HandRigids[i]->GetBulletRigid()->setMotionState(pState);
	//		}
	//		else
	//			pState->setWorldTransform(transform);

	//		i++;
	//	}
	//}

	m_Frame = e.sender.frame();
	for (auto& branch : WorldTree->leaves())
	{
		for (const auto& hand : m_Frame.hands())
		{
			auto & subjects = branch.Subjects;
			branch.AddSubjectiveObject(hand,e.toWorldTransform);
			//if (!subjects[hand.id()])
			//{
			//	subjects[hand.id()].reset(
			//		new HandPhysicalModel(
			//		pFrame->pDynamicsWorld,
			//		hand, e.toWorldTransform,
			//		pFrame->SubjectTransform)
			//		);
			//	//for (const auto& itm : pFrame->Objects)
			//	//{
			//	//	const auto& pObj = itm.second;
			//	//	if (!pObj->GetBulletRigid()->isStaticOrKinematicObject())
			//	//	{
			//	//		for (const auto& bone : subjects[hand.id()]->Rigids())
			//	//			pObj->GetBulletRigid()->setIgnoreCollisionCheck(bone.get(), false);
			//	//	}
			//	//}
			//}
		}
	}

	//for (size_t j = 0; j < i; j++)
	//{
	//	pDynamicsWorld->addRigidBody(m_HandRigids[j]->GetBulletRigid());
	//	m_HandRigids[j]->GetBulletRigid()->setGravity({ 0,0,0 });
	//}
}

void Causality::WorldScene::OnHandsTrackLost(const UserHandsEventArgs & e)
{
	m_Frame = e.sender.frame();
	m_FrameTransform = e.toWorldTransform;
	if (m_Frame.hands().count() == 0)
	{
		m_HaveHands = false;
		std::lock_guard<mutex> guard(m_HandFrameMutex);
		m_HandTrace.clear();
		m_TraceSamples.clear();
		WorldTree->Collapse();
		//for (const auto &pRigid : m_HandRigids)
		//{
		//	pDynamicsWorld->removeRigidBody(pRigid->GetBulletRigid());
		//}
	}
}

void Causality::WorldScene::OnHandsMove(const UserHandsEventArgs & e)
{
	std::lock_guard<mutex> guard(m_HandFrameMutex);
	m_Frame = e.sender.frame();
	m_FrameTransform = e.toWorldTransform;
	XMMATRIX leap2world = m_FrameTransform;
	//std::array<DirectX::Vector3, 25> joints;
	std::vector<DirectX::BoundingOrientedBox> handBoxes;
	float fingerStdev = 0.02f;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> normalDist(0, fingerStdev);
	std::uniform_real<float> uniformDist;

	// Caculate moving trace
	int handIdx = 0;
	for (const auto& hand : m_Frame.hands())
	{
		int fingerIdx = 0; // hand idx
		m_HandTrace.emplace_back();
		m_TraceSamples.emplace_back();
		auto& samples = m_TraceSamples.back();
		auto& joints = m_HandTrace.back();
		for (const auto& finger : hand.fingers())
		{
			XMVECTOR bJ = XMVector3Transform(finger.bone((Leap::Bone::Type)0).prevJoint().toVector3<Vector3>(), leap2world);
			joints[fingerIdx * 5] = bJ;
			for (size_t boneIdx = 0; boneIdx < 4; boneIdx++) // bone idx
			{
				const auto & bone = finger.bone((Leap::Bone::Type)boneIdx);
				XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), leap2world);
				joints[fingerIdx * 5 + boneIdx + 1] = eJ;


				//auto dir = eJ - bJ;
				//float dis = XMVectorGetX(XMVector3Length(dir));
				//if (abs(dis) < 0.001)
				//	continue;
				//XMVECTOR rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, dir);
				//bJ = eJ;
				//for (size_t k = 0; k < 100; k++) // bone idx
				//{
				//	float x = normalDist(gen);
				//	float h = uniformDist(gen);
				//	float z = normalDist(gen);
				//	XMVECTOR disp = XMVectorSet(x, h*dis, z, 1);
				//	disp = XMVector3Rotate(disp, rot);
				//	disp += bJ;
				//	samples[(fingerIdx * 4 + boneIdx) * 100 + k] = disp;
				//}
			}
			fingerIdx++;
		}
		handIdx++;
		while (m_HandTrace.size() > 60)
		{
			m_HandTrace.pop_front();
			//m_TraceSamples.pop_front();
		}

		// Cone intersection test section
		//Vector3 rayEnd = XMVector3Transform(hand.palmPosition().toVector3<Vector3>(), leap2world);
		//Vector3 rayBegin = m_pCameraLocation->GetPosition();
		//auto pConeShape = new btConeShape(100, XM_PI / 16 * 100);
		//auto pCollisionCone = new btCollisionObject();
		//pCollisionCone->setCollisionShape(pConeShape);
		//btTransform trans(
		//	vector_cast<btQuaternion>(XMQuaternionRotationVectorToVector(g_XMIdentityR1, rayEnd - rayBegin)),
		//	vector_cast<btVector3>(rayBegin));
		//pCollisionCone->setWorldTransform(trans);
		//class Callback : public btDynamicsWorld::ContactResultCallback
		//{
		//public:
		//	const IModelNode* pModel;
		//	Callback() {}

		//	void SetModel(const IModelNode* pModel)
		//	{
		//		this->pModel = pModel;
		//	}
		//	Callback(const IModelNode* pModel)
		//		: pModel(pModel)
		//	{
		//	}
		//	virtual	btScalar	addSingleResult(btManifoldPoint& cp, const btCollisionObjectWrapper* colObj0Wrap, int partId0, int index0, const btCollisionObjectWrapper* colObj1Wrap, int partId1, int index1)
		//	{
		//		cout << "point frustrum contact with "<< pModel->Name << endl;
		//		return 0;
		//	}
		//};
		//static map<string, Callback> callbackTable;


		//for (const auto& model : Children)
		//{
		//	auto pRigid = dynamic_cast<PhysicalRigid*>(model.get());
		//	callbackTable[model->Name].SetModel(model.get());
		//	pDynamicsWorld->contactPairTest(pRigid->GetBulletRigid(), pCollisionCone, callbackTable[model->Name]);
		//}
	}

	//int i = 0;
	//const auto& hand = m_Frame.hands().frontmost();
	//for (const auto& finger : hand.fingers())
	//{
	//	XMVECTOR bJ = XMVector3Transform(finger.bone((Leap::Bone::Type)0).prevJoint().toVector3<Vector3>(), leap2world);
	//	auto pState = m_HandRigids[i]->GetBulletRigid()->getMotionState();
	//	auto transform = btTransform::getIdentity();
	//	transform.setOrigin(vector_cast<btVector3>(bJ));
	//	m_HandRigids[i]->GetBulletRigid()->proceedToTransform(transform);


	//	i++;
	//	for (size_t boneIdx = 0; boneIdx < 4; boneIdx++) // bone idx
	//	{
	//		const auto & bone = finger.bone((Leap::Bone::Type)boneIdx);
	//		XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), leap2world);

	//		auto pState = m_HandRigids[i]->GetBulletRigid()->getMotionState();
	//		auto transform = btTransform::getIdentity();
	//		transform.setOrigin(vector_cast<btVector3>(eJ));
	//		m_HandRigids[i]->GetBulletRigid()->proceedToTransform(transform);

	//		i++;
	//	}
	//}


	std::cout << "[Leap] Hands Move." << std::endl;
}

void Causality::WorldScene::OnKeyDown(const KeyboardEventArgs & e)
{
}

void Causality::WorldScene::OnKeyUp(const KeyboardEventArgs & e)
{
	if (e.Key == 'T')
		m_showTrace = !m_showTrace;
}

void Causality::WorldScene::AddObject(const std::shared_ptr<IModelNode>& pModel, float mass, const DirectX::Vector3 & Position, const DirectX::Quaternion & Orientation, const Vector3 & Scale)
{
	lock_guard<mutex> guard(m_RenderLock);
	Models.push_back(pModel);
	auto pShaped = dynamic_cast<IShaped*>(pModel.get());
	auto pShape = pShaped->CreateCollisionShape();
	pShape->setLocalScaling(vector_cast<btVector3>(Scale));

	WorldTree->AddDynamicObject(pModel->Name, pShape, mass, Position, Orientation);
	//for (const auto& pFrame : m_StateFrames)
	//{
	//	auto pObject = std::shared_ptr<PhysicalRigid>(new PhysicalRigid());
	//	pObject->InitializePhysics(pFrame->pDynamicsWorld, pShape, mass, Position, Orientation);
	//	pObject->GetBulletRigid()->setFriction(1.0f);
	//	pObject->GetBulletRigid()->setDamping(0.8, 0.9);
	//	pObject->GetBulletRigid()->setRestitution(0.0);
	//	pFrame->Objects[pModel->Name] = pObject;
	//}
}

std::pair<DirectX::Vector3, DirectX::Quaternion> XM_CALLCONV CaculateCylinderTransform(FXMVECTOR P1, FXMVECTOR P2)
{
	std::pair<DirectX::Vector3, DirectX::Quaternion> trans;
	auto center = XMVectorAdd(P1, P2);
	center = XMVectorMultiply(center, g_XMOneHalf);
	auto dir = XMVectorSubtract(P1, P2);
	auto scale = XMVector3Length(dir);
	XMVECTOR rot;
	if (XMVector4Equal(dir, g_XMZero))
		rot = XMQuaternionIdentity();
	else
		rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1.v, dir);
	trans.first = center;
	trans.second = rot;
	return trans;
}

XMMATRIX Causality::HandPhysicalModel::CaculateLocalMatrix(const Leap::Hand & hand, const DirectX::Matrix4x4 & leapTransform)
{
	XMVECTOR palmCenter = hand.palmPosition().toVector3<Vector3>();
	return XMMatrixScalingFromCenter(m_InheritTransform.Scale, palmCenter) * ((RigidTransform&) m_InheritTransform).TransformMatrix() * (XMMATRIX) leapTransform;

}


Causality::HandPhysicalModel::HandPhysicalModel
(const std::shared_ptr<btDynamicsWorld> &pWorld,
const Leap::Hand & hand, const DirectX::Matrix4x4 & leapTransform,
const DirectX::AffineTransform &inheritTransform)
: m_Hand(hand)
{
	Color.G(0.5f);
	Color.B(0.5f);

	Id = hand.id();
	m_InheritTransform = inheritTransform;

	LocalMatrix = CaculateLocalMatrix(hand, leapTransform);
	XMMATRIX leap2world = LocalMatrix;
	int j = 0;
	for (const auto& finger : m_Hand.fingers())
	{
		for (size_t i = 0; i < 4; i++)
		{
			const auto & bone = finger.bone((Leap::Bone::Type)i);
			XMVECTOR bJ = XMVector3Transform(bone.prevJoint().toVector3<Vector3>(), leap2world);
			XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), leap2world);
			m_Bones[i + j * 4].first = bJ;
			m_Bones[i + j * 4].second = eJ;

			// Initalize rigid hand model
			auto center = 0.5f * XMVectorAdd(bJ, eJ);
			auto dir = XMVectorSubtract(eJ, bJ);
			auto height = std::max(XMVectorGetX(XMVector3Length(dir)), fingerLength);
			XMVECTOR rot;
			if (XMVector4Equal(dir, g_XMZero))
				rot = XMQuaternionIdentity();
			else
				rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, dir);
			shared_ptr<btCapsuleShape> pShape(new btCapsuleShape(fingerRadius, height));

			// Scaling in Y axis is encapsled in bJ and eJ
			btVector3 scl = vector_cast<btVector3>(m_InheritTransform.Scale);
			scl.setY(1.0f);
			pShape->setLocalScaling(scl);

			m_HandRigids.emplace_back(new PhysicalRigid());
			const auto & pRigid = m_HandRigids.back();
			//pRigid->GetBulletRigid()->setGravity({ 0,0,0 });
			pRigid->InitializePhysics(nullptr, pShape, 0, center, rot);
			const auto& body = pRigid->GetBulletRigid();
			body->setFriction(1.0f);
			body->setRestitution(0.0f);
			body->setCollisionFlags(body->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
			body->setActivationState(DISABLE_DEACTIVATION);
			//body->setAngularFactor(0.0f); // Rotation Along Y not affact

			pRigid->Enable(pWorld);
		}
	}

	//for (size_t i = 0; i < m_HandRigids.size(); i++)
	//{
	//	for (size_t j = 0; j < m_HandRigids.size(); j++)
	//	{
	//		if (i != j)
	//			m_HandRigids[i]->GetBulletRigid()->setIgnoreCollisionCheck(m_HandRigids[j]->GetBulletRigid(), true);
	//	}
	//}

}

bool Causality::HandPhysicalModel::Update(const Leap::Frame & frame, const DirectX::Matrix4x4 & leapTransform)
{
	m_Hand = frame.hand(Id);

	if (m_Hand.isValid())
	{
		XMMATRIX transform = CaculateLocalMatrix(m_Hand, leapTransform);
		LocalMatrix = transform;
		Color.R(m_Hand.grabStrength());
		LostFrames = 0;
		int j = 0;
		for (const auto& finger : m_Hand.fingers())
		{
			for (size_t i = 0; i < 4; i++)
			{
				const auto & bone = finger.bone((Leap::Bone::Type)i);
				XMVECTOR bJ = XMVector3Transform(bone.prevJoint().toVector3<Vector3>(), transform);
				XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), transform);
				m_Bones[i + j * 4].first = bJ;
				m_Bones[i + j * 4].second = eJ;

				// Rigid hand model

				auto & pRigid = m_HandRigids[i + j * 4];
				if (!pRigid->IsEnabled())
					pRigid->Enable();
				auto center = 0.5f * XMVectorAdd(bJ, eJ);
				auto dir = XMVectorSubtract(eJ, bJ);
				XMVECTOR rot;
				if (XMVector4Equal(dir, g_XMZero))
					rot = XMQuaternionIdentity();
				else
					rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, dir);
				auto trans = btTransform(vector_cast<btQuaternion>(rot), vector_cast<btVector3>(center));
				pRigid->GetBulletRigid()->getMotionState()->setWorldTransform(trans);
			}
			j++;
		}
		return true;
	}
	else
	{
		for (auto& pRigid : m_HandRigids)
		{
			pRigid->Disable();
		}
		LostFrames++;
		return false;
	}
}

// Inherited via IModelNode

void Causality::HandPhysicalModel::Render(ID3D11DeviceContext * pContext, DirectX::IEffect * pEffect)
{
	XMMATRIX leap2world = LocalMatrix;
	//auto palmPosition = XMVector3Transform(m_Hand.palmPosition().toVector3<Vector3>(), leap2world);
	//g_PrimitiveDrawer.DrawSphere(palmPosition, 0.02f, Colors::YellowGreen);
	Color.A(Opticity);
	XMVECTOR color = Color;
	//color = XMVectorSetW(color, Opticity);

	//g_PrimitiveDrawer.Begin();
	//for (const auto& bone : m_Bones)
	//{
	//	//g_PrimitiveDrawer.DrawSphere(bone.second, fingerRadius, jC);
	//	g_PrimitiveDrawer.DrawCylinder(bone.first, bone.second, fingerRadius * m_InheritTransform.Scale.x, color);
	//}
	//g_PrimitiveDrawer.End();

	for (const auto& pRigid : m_HandRigids)
	{
		g_PrimitiveDrawer.DrawCylinder(
			pRigid->GetPosition(),
			XMVector3Rotate(g_XMIdentityR1, pRigid->GetOrientation()),
			dynamic_cast<btCapsuleShape*>(pRigid->GetBulletShape())->getHalfHeight() * 2,
			fingerRadius * m_InheritTransform.Scale.x,
			color);
	}

	//for (const auto& finger : m_Hand.fingers())
	//{
	//	for (size_t i = 0; i < 4; i++)
	//	{
	//		const auto & bone = finger.bone((Leap::Bone::Type)i);
	//		XMVECTOR bJ = XMVector3Transform(bone.prevJoint().toVector3<Vector3>(), leap2world);
	//		XMVECTOR eJ = XMVector3Transform(bone.nextJoint().toVector3<Vector3>(), leap2world);
	//		//g_PrimitiveDrawer.DrawLine(bJ, eJ, Colors::LimeGreen);
	//		//g_PrimitiveDrawer.DrawCube(bJ, g_XMOne * 0.03, g_XMIdentityR3, Colors::Red);
	//		g_PrimitiveDrawer.DrawCylinder(bJ, eJ,0.015f,Colors::LimeGreen);

	//		//auto center = 0.5f * XMVectorAdd(bJ, eJ);
	//		//auto dir = XMVectorSubtract(eJ, bJ);
	//		//auto scale = XMVector3Length(dir);
	//		//XMVECTOR rot;
	//		//if (XMVector4LessOrEqual(XMVector3LengthSq(dir), XMVectorReplicate(0.01f)))
	//		//	rot = XMQuaternionIdentity();
	//		//else
	//		//	rot = XMQuaternionRotationVectorToVector(g_XMIdentityR1, dir);
	//		//XMMATRIX world = XMMatrixAffineTransformation(scale, g_XMZero, rot, center);
	//		//s_pCylinder->Draw(world, ViewMatrix, ProjectionMatrix,Colors::LimeGreen);
	//	}
	//}
}

inline void debug_assert(bool condition)
{
#ifdef DEBUG
	if (!condition)
	{
		_CrtDbgBreak();
		//std::cout << "assert failed." << std::endl;
	}
#endif
}

// normalized feild intensity equalent charge
XMVECTOR XM_CALLCONV FieldSegmentToPoint(FXMVECTOR P, FXMVECTOR L0, FXMVECTOR L1)
{
	if (XMVector4NearEqual(L0, L1, XMVectorReplicate(0.001f)))
	{
		XMVECTOR v = XMVectorAdd(L0, L1);
		v = XMVectorMultiply(v, g_XMOneHalf);
		v = XMVectorSubtract(v, P);
		XMVECTOR d = XMVector3LengthSq(v);
		v = XMVector3Normalize(v);
		v /= d;
		return v;
	}

	XMVECTOR s = XMVectorSubtract(L1, L0);
	XMVECTOR v0 = XMVectorSubtract(L0, P);
	XMVECTOR v1 = XMVectorSubtract(L1, P);

	XMMATRIX Rot;
	Rot.r[1] = XMVector3Normalize(s);
	Rot.r[2] = XMVector3Cross(v0, v1);
	Rot.r[2] = XMVector3Normalize(Rot.r[2]);
	Rot.r[0] = XMVector3Cross(Rot.r[1], Rot.r[2]);
	Rot.r[3] = g_XMIdentityR3;

	// Rotated to standard question:
	//  Y
	//  ^     *y1
	//  |     |
	//--o-----|x0----->X
	//  |     |
	//	|     |
	//	      *y0
	// Close form solution of the intergral : f(y0,y1) = <-y/(x0*sqrt(x0^2+y^2)),1/sqrt(x0^2+y^2),0> | (y0,y1)
	XMVECTOR Ds = XMVector3ReciprocalLength(s);
	XMVECTOR Ps = XMVector3Dot(v0, s);
	XMVECTOR Y0 = XMVectorMultiply(Ps, Ds);

	Ps = XMVector3Dot(v1, s);
	XMVECTOR Y1 = XMVectorMultiply(Ps, Ds);

	XMVECTOR X0 = XMVector3LengthSq(v1);
	Ps = XMVectorMultiply(Y1, Y1);
	X0 = XMVectorSubtract(X0, Ps);
	//debug_assert(XMVector4GreaterOrEqual(X0, XMVectorZero()));
	XMVECTOR R0 = XMVectorMultiplyAdd(Y0, Y0, X0);
	XMVECTOR R1 = XMVectorMultiplyAdd(Y1, Y1, X0);
	R0 = XMVectorReciprocalSqrt(R0);
	R1 = XMVectorReciprocalSqrt(R1);

	XMVECTOR Ry = XMVectorSubtract(R1, R0);

	R0 = XMVectorMultiply(R0, Y0);
	R1 = XMVectorMultiply(R1, Y1);
	XMVECTOR Rx = XMVectorSubtract(R0, R1);
	X0 = XMVectorReciprocalSqrt(X0);
	//debug_assert(!XMVectorGetIntX(XMVectorIsNaN(X0)));
	Rx = XMVectorMultiply(Rx, X0);
	Rx = XMVectorSelect(Rx, Ry, g_XMSelect0101);
	// Field intensity in P centered coordinate
	Rx = XMVectorAndInt(Rx, g_XMSelect1100);

	Rx = XMVectorMultiply(Rx, Ds);
	Rx = XMVector3Transform(Rx, Rot);

	//debug_assert(!XMVectorGetIntX(XMVectorIsNaN(Rx)));
	return Rx;
}

DirectX::XMVECTOR XM_CALLCONV Causality::HandPhysicalModel::FieldAtPoint(DirectX::FXMVECTOR P)
{
	// Palm push force 
	//XMVECTOR palmP = m_Hand.palmPosition().toVector3<Vector3>();
	//XMVECTOR palmN = m_Hand.palmNormal().toVector3<Vector3>();
	//auto dis = XMVectorSubtract(P,palmP);
	//auto mag = XMVectorReciprocal(XMVector3LengthSq(dis));
	//dis = XMVector3Normalize(dis);
	//auto fac = XMVector3Dot(dis, palmN);
	//mag = XMVectorMultiply(fac, mag);
	//return XMVectorMultiply(dis, mag);

	XMVECTOR field = XMVectorZero();
	for (const auto& bone : m_Bones)
	{
		XMVECTOR v0 = bone.first;
		XMVECTOR v1 = bone.second;
		XMVECTOR f = FieldSegmentToPoint(P, v0, v1);
		field += f;
		//XMVECTOR l = XMVector3LengthSq(XMVectorSubtract(v1,v0));
	}
	return field;
}

inline Causality::CubeModel::CubeModel(const std::string & name, DirectX::FXMVECTOR extend, DirectX::FXMVECTOR color)
{
	Name = name;
	m_Color = color;

	XMStoreFloat3(&BoundBox.Extents, extend);
	XMStoreFloat3(&BoundOrientedBox.Extents, extend);
}

std::shared_ptr<btCollisionShape> Causality::CubeModel::CreateCollisionShape()
{
	std::shared_ptr<btCollisionShape> pShape;
	pShape.reset(new btBoxShape(vector_cast<btVector3>(BoundBox.Extents)));
	return pShape;
}

void Causality::CubeModel::Render(ID3D11DeviceContext * pContext, DirectX::IEffect * pEffect)
{
	XMVECTOR extent = XMLoadFloat3(&BoundBox.Extents);
	XMMATRIX world = GetWorldMatrix();
	//XMVECTOR scale, pos, rot;
	XMVECTOR color = m_Color;
	color = XMVectorSetW(color, Opticity);
	g_PrimitiveDrawer.DrawCube(extent, world, color);
}

// !!!Current don't support dynamic scaling for each state now!!!
inline std::shared_ptr<btCollisionShape> Causality::ShapedGeomrtricModel::CreateCollisionShape()
{
	if (!m_pShape)
	{
		btTransform trans;
		std::shared_ptr<btCompoundShape> pShape(new btCompoundShape());
		//trans.setOrigin(vector_cast<btVector3>(model->BoundOrientedBox.Center));
		//trans.setRotation(vector_cast<btQuaternion>(model->BoundOrientedBox.Orientation));
		//pShape->addChildShape(trans, new btBoxShape(vector_cast<btVector3>(model->BoundOrientedBox.Extents)));
		for (const auto& part : Parts)
		{
			trans.setOrigin(vector_cast<btVector3>(part->BoundOrientedBox.Center));
			trans.setRotation(vector_cast<btQuaternion>(part->BoundOrientedBox.Orientation));
			pShape->addChildShape(trans, new btBoxShape(vector_cast<btVector3>(part->BoundOrientedBox.Extents)));
		}
		m_pShape = pShape;
		return m_pShape;
	}
	else
	{
		return m_pShape;
	}
}

void Causality::WorldBranch::InitializeBranchPool(int size, bool autoExpandation)
{
	for (size_t i = 0; i < 30; i++)
	{
		BranchPool.emplace(new WorldBranch());
	}
}

void Causality::WorldBranch::Reset()
{
	for (const auto& pair : Items)
	{
		pair.second->Disable();
	}
	Items.clear();
}

void Causality::WorldBranch::Collapse()
{
	//using namespace cpplinq;
	//using cref = decltype(m_StateFrames)::const_reference;
	//auto mlh = from(m_StateFrames)
	//	>> where([](cref pFrame) {return pFrame->IsEnabled; })
	//	>> max([](cref pFrame)->float {return pFrame->Liklyhood(); });
	//WordBranch master_frame;
	////for (auto & pFrame : m_StateFrames)
	////{
	////	if (pFrame->Liklyhood() < mlh)
	////	{
	////		pFrame->Disable();
	////		m_pFramesPool->Recycle(std::move(pFrame));
	////	}
	////	else
	////	{
	////		master_frame = std::move(pFrame);
	////	}
	////}
	//m_StateFrames.clear();
	//m_StateFrames.push_back(std::move(master_frame));
}

SuperpositionMap Causality::WorldBranch::CaculateSuperposition()
{
	using namespace cpplinq;
	SuperpositionMap SuperStates;

	auto itr = this->begin();
	auto eitr = this->end();

	NormalizeLiklyhood(CaculateLiklyhood());

	auto pItem = Items.begin();
	for (size_t i = 0; i < Items.size(); i++,++pItem)
	{
		const auto& pModel = pItem->second;
		auto& distribution = SuperStates[pItem->first];
		//auto&  = state.StatesDistribution;
		int j = 0;

		for (const auto& branch : leaves())
		{
			if (!branch.IsEnabled)
				continue;

			auto itrObj = branch.Items.find(pItem->first);

			if (itrObj == branch.Items.end())
				continue;
			auto pNew = itrObj->second;

			ProblistiscAffineTransform tNew;
			tNew.Translation = pNew->GetPosition();
			tNew.Rotation = pNew->GetOrientation();
			tNew.Scale = pNew->GetScale();
			tNew.Probability = branch.Liklyhood();

			auto itr = std::find_if(distribution.begin(), distribution.end(),
				[&tNew](std::remove_reference_t<decltype(distribution)>::const_reference trans) -> bool
			{
				return trans.NearEqual(tNew);
			});

			if (itr == distribution.end())
			{
				distribution.push_back(tNew);
			}
			else
			{
				itr->Probability += tNew.Probability;
			}
			j++;
		}
	}

	return SuperStates;
}

void Causality::WorldBranch::InternalEvolution(float timeStep, const Leap::Frame & frame, const DirectX::Matrix4x4 & leapTransform)
{
	auto& subjects = Subjects;

	BoundingSphere sphere;

	//if (!is_leaf()) return;
	for (auto itr = subjects.begin(); itr != subjects.end(); )
	{
		bool result = itr->second->Update(frame, leapTransform);
		// Remove hands lost track for 60+ frames
		if (!result)
		{
			if (itr->second->LostFramesCount() > 60)
				itr = subjects.erase(itr);
		}
		else
		{
			//std::vector<PhysicalRigid*> collideObjects;

			//for (auto& item : Items)
			//{
			//	btVector3 c;
			//	item.second->GetBulletShape()->getBoundingSphere(c, sphere.Radius);
			//	sphere.Center = vector_cast<Vector3>(sphere.Center);
			//	if (itr->second->OperatingFrustum().Contains(sphere) != ContainmentType::DISJOINT)
			//	{
			//		collideObjects.push_back(item.second.get());
			//	}
			//}
			//if (collideObjects.size() > 0)
			//{
			//	Fork(collideObjects);
			//}


			//const auto &pHand = itr->second;
			//for (auto& item : pFrame->Objects)
			//{
			//	const auto& pObj = item.second;
			//	if (pObj->GetBulletRigid()->isStaticObject())
			//		continue;
			//	//pObj->GetBulletShape()->
			//	auto force = pHand->FieldAtPoint(pObj->GetPosition()) * 0.00001f;
			//	pObj->GetBulletRigid()->clearForces();
			//	//vector_cast<btVector3>(force) * 0.01f
			//	std::cout << item.first << " : " << Vector3(force) << std::endl;
			//	pObj->GetBulletRigid()->applyCentralForce(vector_cast<btVector3>(force));
			//	pObj->GetBulletRigid()->activate();
			//}
			++itr;
		}
	}
	pDynamicsWorld->stepSimulation(timeStep, 10);
}

float Causality::WorldBranch::CaculateLiklyhood()
{
	if (is_leaf())
	{
		if (IsEnabled)
			_Liklyhood = 1;
		else
			_Liklyhood = 0;
		return _Liklyhood;
	}
	else
	{
		_Liklyhood = 0;
		for (auto& branch : children())
		{
			_Liklyhood += branch.CaculateLiklyhood();
		}
		return _Liklyhood;
	}
}

void Causality::WorldBranch::NormalizeLiklyhood(float total)
{
	for (auto& branch : nodes_in_tree())
	{
		branch._Liklyhood /= total;
	}
}

void Causality::WorldBranch::AddSubjectiveObject(const Leap::Hand & hand, const DirectX::Matrix4x4& leapTransform)
{
	if (!Subjects[hand.id()])
	{
		Subjects[hand.id()].reset(
			new HandPhysicalModel(
			pDynamicsWorld,
			hand, leapTransform,
			SubjectTransform)
			);
		//for (const auto& itm : pFrame->Objects)
		//{
		//	const auto& pObj = itm.second;
		//	if (!pObj->GetBulletRigid()->isStaticOrKinematicObject())
		//	{
		//		for (const auto& bone : subjects[hand.id()]->Rigids())
		//			pObj->GetBulletRigid()->setIgnoreCollisionCheck(bone.get(), false);
		//	}
		//}
	}
}

void Causality::WorldBranch::AddDynamicObject(const std::string &name, const std::shared_ptr<btCollisionShape> &pShape, float mass, const DirectX::Vector3 & Position, const DirectX::Quaternion & Orientation)
{
	for (auto& branch : nodes_in_tree())
	{
		auto pObject = std::shared_ptr<PhysicalRigid>(new PhysicalRigid());
		pObject->InitializePhysics(branch.pDynamicsWorld, pShape, mass, Position, Orientation);
		pObject->GetBulletRigid()->setFriction(1.0f);
		pObject->GetBulletRigid()->setDamping(0.8f, 0.9f);
		pObject->GetBulletRigid()->setRestitution(0.0);
		branch.Items[name] = pObject;
	}
}

void Causality::WorldBranch::Evolution(float timeStep, const Leap::Frame & frame, const DirectX::Matrix4x4 & leapTransform)
{
	using namespace cpplinq;
	vector<reference_wrapper<WorldBranch>> leaves;



	//auto levr = this->leaves();
	copy(this->leaves_begin(), leaves_end(), back_inserter(leaves));

	auto branchEvolution = [timeStep, &frame, &leapTransform](WorldBranch& branch) {
		branch.InternalEvolution(timeStep,frame, leapTransform);
	};
	//auto branchEvolution = std::bind(&WorldBranch::InternalEvolution, placeholders::_1, frame, leapTransform);

	if (leaves.size() >= 10)
		concurrency::parallel_for_each(leaves.begin(), leaves.end(), branchEvolution);
	else
		for_each(leaves.begin(), leaves.end(), branchEvolution);
}

void Causality::WorldBranch::Fork(const std::vector<PhysicalRigid*>& focusObjects)
{
	//int i = 0;
	//for (const auto& obj : focusObjects)
	//{
	//	auto branch = DemandCreate((boost::format("%s/%d") % this->Name % i++).str());
	//}
}

void Causality::WorldBranch::Fork(const std::vector<DirectX::AffineTransform>& subjectTransforms)
{
	for (int i = subjectTransforms.size() - 1; i >= 0; --i)
	{
		const auto& trans = subjectTransforms[i];
		auto branch = DemandCreate((boost::format("%s/%d") % this->Name % i).str());
		branch->Enable(trans);
		append_children_front(branch.release());
		//branch->SubjectTransform = trans;
	}
}
std::unique_ptr<WorldBranch> Causality::WorldBranch::DemandCreate(const string& branchName)
{
	if (!BranchPool.empty())
	{
		auto frame = std::move(BranchPool.front());
		BranchPool.pop();
		frame->Name = branchName;
		return frame;
	}
	else
		return nullptr;
}

void Causality::WorldBranch::Recycle(std::unique_ptr<WorldBranch>&& pFrame)
{
	pFrame->Reset();
	BranchPool.push(std::move(pFrame));
}

//inline void Causality::SkeletonModel::Render(ID3D11DeviceContext * pContext, DirectX::IEffect * pEffect)
//{
//	g_PrimitiveDrawer.DrawCylinder(Joints[0],Joints[1].Position)
//}
