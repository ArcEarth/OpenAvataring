#include "pch_bcl.h"
#include <fbxsdk.h>
#include "FbxParser.h"
//#include "EigenExtension.h"
#include <iostream>
#include <fstream>

#pragma comment(lib,"libfbxsdk.lib")

namespace fbx = fbxsdk;
using namespace fbx;

//using namespace fbx;
namespace Causality
{
	static const float PcaCutoff = 0.01f; // 0.2^2
	const size_t MaxBlendSize = 32U;
	const size_t ReducedBlendSize = 4U;
	static const float QuaternionFlipLimit = 0.2f * DirectX::XM_PI;
	static const float QuaternionFlipLimit2 = 0.1f * DirectX::XM_PI;

	inline DirectX::Vector4 ToDx(const fbx::FbxDouble4& v4)
	{
		return DirectX::Vector4(v4[0], v4[1], v4[2], v4[3]);
	}
	inline DirectX::Vector3 ToDx(const fbx::FbxDouble3& v3)
	{
		return DirectX::Vector3(v3[0], v3[1], v3[2]);
	}
	inline DirectX::Vector2 ToDx(const fbx::FbxDouble2& v2)
	{
		return DirectX::Vector2(v2[0], v2[1]);
	}
	inline DirectX::Quaternion ToDx(const fbx::FbxQuaternion& fbxQ)
	{
		return DirectX::Quaternion((float)fbxQ[0], (float)fbxQ[1], (float)fbxQ[2], (float)fbxQ[3]);
	}
	inline DirectX::Quaternion EularToDxQuaternion(const fbx::FbxVector4& eular)
	{
		fbx::FbxQuaternion fbxQ;
		fbxQ.ComposeSphericalXYZ(eular);
		return ToDx(fbxQ);
	}
	inline DirectX::Color ToDxColor(const fbx::FbxDouble3& v3)
	{
		return DirectX::Color(v3[0], v3[1], v3[2]);
	}
	struct FbxParser::Impl
	{
	public:
		Impl()
		{
			m_Armature = nullptr;
			m_Behavier = nullptr;
			m_Options.FileAxisSystem = AxisSystem::Unknown;
			m_Options.FlipNormals = false;
			m_Options.RewindTriangles = false;
			m_Options.ImportMode = (unsigned)Mode::None;
		}

		weak_ptr<fbx::FbxManager>   sw_SdkManager;
		shared_ptr<fbx::FbxManager> m_SdkManger;
		vector<fbx::FbxMesh*>		m_MeshNodes;
		vector<fbx::FbxNode*>		m_BoneNodes;

		BehavierSpace*          m_Behavier;
		StaticArmature*         m_Armature;

		std::list<SkinMeshData> m_SkinnedMeshes;
		vector<int>				m_ParentMap;
		int						m_FrameCount;
		fbx::FbxTime			m_FrameInterval;
		fbx::FbxTime			m_AnimationTime;
		int						m_NodeIdx = 0;
		FbxFileImportOptions	m_Options;


		void SetupAnimationPivots()
		{
			// Do this setup for each node (FbxNode).
			// We set up what we want to bake via ConvertPivotAnimationRecursive.
			// When the destination is set to 0, baking will occur.
			// When the destination value is set to the source¡¯s value, the source values will be retained and not baked.
			for (auto pNode : m_BoneNodes)
			{
				FbxVector4 lZero(0, 0, 0);

				// Activate pivot converting
				pNode->SetPivotState(FbxNode::eSourcePivot, FbxNode::ePivotActive);
				pNode->SetPivotState(FbxNode::eDestinationPivot, FbxNode::ePivotActive);

				// We want to set all these to 0 and bake them into the transforms.
				pNode->SetPostRotation(FbxNode::eDestinationPivot, lZero);
				pNode->SetPreRotation(FbxNode::eDestinationPivot, lZero);
				pNode->SetRotationOffset(FbxNode::eDestinationPivot, lZero);
				pNode->SetScalingOffset(FbxNode::eDestinationPivot, lZero);
				pNode->SetRotationPivot(FbxNode::eDestinationPivot, lZero);
				pNode->SetScalingPivot(FbxNode::eDestinationPivot, lZero);

				// This is to import in a system that supports rotation order.
				// If rotation order is not supported, do this instead:
				// pNode->SetRotationOrder(FbxNode::eDestinationPivot, FbxNode::eEulerXYZ);
				FbxEuler::EOrder lRotationOrder;// = FbxEuler::EOrder::eOrderZXY;
				pNode->GetRotationOrder(FbxNode::eSourcePivot, lRotationOrder);
				pNode->SetRotationOrder(FbxNode::eDestinationPivot, lRotationOrder);

				// Similarly, this is the case where geometric transforms are supported by the system.
				// If geometric transforms are not supported, set them to zero instead of
				// the source¡¯s geometric transforms.
				// Geometric transform = local transform, not inherited by children.
				pNode->SetGeometricTranslation(FbxNode::eDestinationPivot, pNode->GetGeometricTranslation(FbxNode::eSourcePivot));
				pNode->SetGeometricRotation(FbxNode::eDestinationPivot, pNode->GetGeometricRotation(FbxNode::eSourcePivot));
				pNode->SetGeometricScaling(FbxNode::eDestinationPivot, pNode->GetGeometricScaling(FbxNode::eSourcePivot));

				// Idem for quaternions.
				auto interop = pNode->GetQuaternionInterpolation(FbxNode::eSourcePivot);
				pNode->SetQuaternionInterpolation(FbxNode::eDestinationPivot, interop);
			}

			// When the setup is done, call ConvertPivotAnimationRecursive to the scene¡¯s root node.
			// Sampling rate e.g. 30.0.
		}

		int GetBoneNodeId(const fbx::FbxNode* pBone) const
		{
			auto itr = std::find(m_BoneNodes.begin(), m_BoneNodes.end(), pBone);
			if (itr == m_BoneNodes.end())
				return -1;
			else
				return itr - m_BoneNodes.begin();
		}

		template <typename T, size_t N>
		void writeVector(std::ostream & binaryFile, const std::array<T, N> & v)
		{
			for (auto i = 0; i < N; i++)
			{
				binaryFile << v[i];
			}
		}
		void writeVector(std::ostream & binaryFile, const fbx::FbxVector4 & v)
		{
			binaryFile << (float)v[0] << (float)v[1] << (float)v[2] << (float)v[3];
		}
		void writeVector(std::ostream & binaryFile, const fbx::FbxVector2 & v)
		{
			binaryFile << (float)v[0] << (float)v[1];
		}

		void SetTransformFromFbxMatrix(IsometricTransform& iso, const fbx::FbxAMatrix& glbM)
		{
			auto gt = glbM.GetT();
			auto gq = glbM.GetQ();
			auto gs = glbM.GetS();
			iso.Rotation = Quaternion(gq[0], gq[1], gq[2], gq[3]);//q;
			iso.Scale = Vector3(gs[0], gs[1], gs[2]);
			iso.Translation = Vector3(gt[0], gt[1], gt[2]);
		}

		void SetBoneGblFromFbxMatrix(Bone& bone, const fbx::FbxAMatrix& glbM)
		{
			auto gt = glbM.GetT();
			auto gq = glbM.GetQ();
			auto gs = glbM.GetS();
			bone.GblRotation = Quaternion(gq[0], gq[1], gq[2], gq[3]);//q;
			bone.GblScaling = Vector3(gs[0], gs[1], gs[2]);
			bone.GblTranslation = Vector3(gt[0], gt[1], gt[2]);
		}
		// Get the geometry deformation local to a node. It is never inherited by the
		// children.
		fbx::FbxAMatrix GetGeometry(FbxNode* pNode)
		{
			fbx::FbxVector4 lT, lR, lS;
			fbx::FbxAMatrix lGeometry;

			lT = pNode->GetGeometricTranslation(FbxNode::eSourcePivot);
			lR = pNode->GetGeometricRotation(FbxNode::eSourcePivot);
			lS = pNode->GetGeometricScaling(FbxNode::eSourcePivot);

			lGeometry.SetT(lT);
			lGeometry.SetR(lR);
			lGeometry.SetS(lS);

			return lGeometry;
		}


		SkinMeshData BuildSkinnedMesh(fbx::FbxMesh* pMesh)
		{
			auto lNode = pMesh->GetNode();
			assert(pMesh->IsTriangleMesh());

			auto lScene = pMesh->GetScene();

			auto numDef = pMesh->GetDeformerCount(fbx::FbxDeformer::eSkin);
			size_t numBones;

			constexpr int RootBoneId = 0;

			assert(numDef <= 1);
			if (numDef == 0)
				numBones = 0;


			fbx::FbxAMatrix bindpose;
			bool useBindPose = false;

			for (int poseid = 0; poseid < lScene->GetPoseCount(); poseid++)
			{
				auto lPose = lScene->GetPose(poseid);
				auto pid = lPose->Find(lNode);
				if (pid != -1)
				{
					auto mat = lPose->GetMatrix(pid);
					bindpose = reinterpret_cast<const fbx::FbxAMatrix&>(mat);
					useBindPose = true;
					break;
				}
			}

			fbx::FbxAMatrix pivot;
			pMesh->GetPivot(pivot);
			assert(pivot.IsIdentity());

			fbx::FbxAMatrix geo = GetGeometry(lNode);
			fbx::FbxAMatrix gbl = lNode->EvaluateGlobalTransform();
			fbx::FbxAMatrix bindTra; // default to identity

			if (numDef)
			{
				fbx::FbxStatus states;
				auto pDeformer = pMesh->GetDeformer(0, fbx::FbxDeformer::eSkin);
				fbx::FbxSkin* pSkin = static_cast<fbx::FbxSkin*>(pDeformer);
				auto numClusters = pSkin->GetClusterCount();
				if (numClusters)
				{
					auto pCluster = pSkin->GetCluster(0);

					if (pCluster)
					{
						//! In most case, this transform matrix is identical to all clusters
						//! Thus we hack the detail code with this
						auto link = pCluster->GetLink();
						auto name = link->GetName();
						pCluster->GetTransformMatrix(bindTra);
					}
				}
			}

			//pivot = geo;
			//if (useBindPose)
			//	pivot = bindpose * geo;
			//else
			pivot = /*gbl * */bindTra * geo;

			//pivot.SetR(fbxEular);
			pMesh->SetPivot(pivot);
			pMesh->ApplyPivot();

			auto numPositions = pMesh->GetControlPointsCount();

			// Vertices
			auto positions = pMesh->GetControlPoints();

			// Polygons
			auto indices = pMesh->GetPolygonVertices();

			// Assume all polygons are triangle
			auto numPolygons = pMesh->GetPolygonCount();

			auto numIndices = numPolygons * 3;

			// Get Skinning data
			std::vector<std::array<float, MaxBlendSize>> blendWeights(numPositions);
			std::vector<std::array<uint8_t, MaxBlendSize>> blendIndices(numPositions);
			std::vector<int> filled(numPositions);
			std::fill_n(filled.begin(), numPositions, 0);

			// Maxium blend bones per vertex
			ZeroMemory(blendWeights.data(), sizeof(float[MaxBlendSize])*numPositions);
			ZeroMemory(blendIndices.data(), sizeof(uint8_t[MaxBlendSize])*numPositions);

			for (int i = 0; i < numDef; i++)
			{
				auto pDeformer = pMesh->GetDeformer(i, fbx::FbxDeformer::eSkin);
				assert(pDeformer->GetDeformerType() == fbx::FbxDeformer::eSkin);
				fbx::FbxSkin* pSkin = static_cast<fbx::FbxSkin*>(pDeformer);
				auto numClusters = pSkin->GetClusterCount();
				numBones = m_BoneNodes.size();
				assert(numBones < 256);

				for (size_t cId = 0; cId < numClusters; cId++)
				{
					auto pCluster = pSkin->GetCluster(cId);
					auto lClusterMode = pCluster->GetLinkMode();

					assert(lClusterMode != fbx::FbxCluster::eAdditive);

					auto pBone = pCluster->GetLink();
					auto boneID = GetBoneNodeId(pBone);

					auto numPoints = pCluster->GetControlPointIndicesCount();
					auto indices = pCluster->GetControlPointIndices();
					auto weights = pCluster->GetControlPointWeights();
					for (size_t i = 0; i < numPoints; i++)
					{
						auto idx = indices[i];
						auto bidx = filled[idx]++;
						blendWeights[idx][bidx] = weights[i];
						blendIndices[idx][bidx] = boneID;
					}
				}
			}

			std::vector<SkinMeshData::VertexType> vertices(numPositions);


			// Per control-point geometry data
			for (size_t i = 0; i < numPositions; i++)
			{
				auto & v = vertices[i];

				v.position = DirectX::XMFLOAT3(positions[i][0], positions[i][1], positions[i][2]);

				v.SetColor(DirectX::Colors::White.v);

				if (filled[i] == 0)
				{
					filled[i] = 1;
					blendWeights[i][0] = 1.0f;
					blendIndices[i][0] = RootBoneId;
				}
				else if (filled[i] <= ReducedBlendSize) // normalize weights
				{
					//__assume(filled[i] > 0 && filled[i] <= ReducedBlendSize);

					float sum = std::accumulate(&blendWeights[i][0], &blendWeights[i][filled[i]],.0f);

					if (abs(sum - 1.0f) > 0.01f)
					{
						using DirectX::operator/=;
						auto v = DirectX::XMLoadFloat4(&blendWeights[i][0]);
						v /= sum;
						DirectX::XMStoreFloat4(&blendWeights[i][0], v);
					}

				} else if (filled[i] > ReducedBlendSize)
				{
					for (size_t j = 0; j < std::min(4, filled[i]); j++)
					{
						for (size_t k = j + 1; k < filled[i]; k++)
						{
							if (blendWeights[i][j] < blendWeights[i][k])
							{
								std::swap(blendWeights[i][j], blendWeights[i][k]);
								std::swap(blendIndices[i][j], blendIndices[i][k]);
							}
						}
					}

					float all = 0;
					for (int j = 0; j < filled[i]; j++)
						all += blendWeights[i][j];
					for (int j = 0; j < filled[i]; j++)
						blendWeights[i][j] /= all;

					float total = blendWeights[i][0] + blendWeights[i][1] + blendWeights[i][2] + blendWeights[i][3];

					// if not, it will be significant artificts
					assert(total > 0.65f);
					//blendWeights[i][0] += 1 - total;
					using DirectX::operator/=;
					auto v = DirectX::XMLoadFloat4(&blendWeights[i][0]);
					v /= total;
					DirectX::XMStoreFloat4(&blendWeights[i][0], v);

				}

				v.SetBlendIndices(DirectX::XMUINT4(blendIndices[i][0], blendIndices[i][1], blendIndices[i][2], blendIndices[i][3]));
				v.SetBlendWeights(reinterpret_cast<DirectX::XMFLOAT4&>(blendWeights[i]));
			}

			// Surface layer data
			// Get UV & Normals

			// Do not Overwite, Vertex normal not face normal, Cunter-Clock-wise
			pMesh->GenerateNormals(false, true, false);
			// Do not Overwite
			pMesh->GenerateTangentsDataForAllUVSets(false);

			auto numLayer = pMesh->GetLayerCount();
			auto leUV = pMesh->GetLayer(0)->GetUVs();
			auto leTagent = pMesh->GetLayer(0)->GetTangents();
			auto leNormal = pMesh->GetLayer(0)->GetNormals();

			auto enrMap = leNormal->GetMappingMode();
			auto enrRef = leNormal->GetReferenceMode();
			auto euvMap = leUV->GetMappingMode();
			auto euvRef = leUV->GetReferenceMode();
			auto etgMap = leTagent->GetMappingMode();
			auto etgRef = leTagent->GetReferenceMode();

			// assert Tagent shares the same map/ref mode with UVs
			assert(euvMap == etgMap);

			float episilon = 0.0001f;

			switch (enrMap)
			{
			case fbx::FbxLayerElement::eByControlPoint:
				for (size_t i = 0; i < numPositions; i++)
				{
					int id = i;
					if (enrRef == fbx::FbxLayerElement::eIndexToDirect)
						id = leNormal->GetIndexArray().GetAt(i);
					fbx::FbxDouble3 norfbx = leNormal->GetDirectArray().GetAt(id);

					vertices[i].normal = ToDx(norfbx);
				}
				break;
			case fbx::FbxLayerElement::eByPolygonVertex:
			{
				assert(leNormal->GetDirectArray().GetCount() == numIndices);
				int copyIdx = numPositions;
				typedef std::vector<std::pair<DirectX::Vector3, int>> uvstatckelem;
				std::vector<uvstatckelem> vertexNRs(numPositions);
				for (int ip = 0; ip < numPolygons; ip++)
				{
					for (size_t j = 0; j < 3U; j++)
					{
						auto idV = pMesh->GetPolygonVertex(ip, j);

						int idNr = ip * 3 + j;
						if (enrRef == fbx::FbxLayerElement::eIndexToDirect)
							idNr = leNormal->GetIndexArray().GetAt(idNr);

						auto snrfbx = leNormal->GetDirectArray().GetAt(idNr); // new surface uv
						DirectX::Vector3 snr(snrfbx[0], snrfbx[1], snrfbx[2]);
						auto& nrs = vertexNRs[idV];
						bool flag = false;

						if (nrs.empty())
						{
							nrs.emplace_back(snr, idV);
							vertices[idV].normal = snr;
						}
						else
						{
							for (auto& nr : nrs)
							{
								if (DirectX::Vector3::DistanceSquared(nr.first, snr) < episilon)
								{
									indices[ip * 3 + j] = nr.second;
									flag = true;
									break;
								}
							}
							if (!flag)
							{
								// This means we need to duplicate a vertex
								indices[ip * 3 + j] = numPositions;
								nrs.emplace_back(snr, numPositions++);
								vertices.push_back(vertices[idV]);
								vertices.back().normal = snr;
								//vertices.back().SetColor(DirectX::Colors::Green);
							}
						}
					}
				}
			}

			break;
			default:
				throw std::exception("not supported.");
				break;
			}

			switch (euvMap)
			{
			case fbx::FbxLayerElement::eByControlPoint:
				for (size_t i = 0; i < numPositions; i++)
				{
					int id = i;
					if (euvRef == fbx::FbxLayerElement::eIndexToDirect)
						id = leUV->GetIndexArray().GetAt(i);
					fbx::FbxDouble2 uvfbx = leUV->GetDirectArray().GetAt(id);

					vertices[i].textureCoordinate = ToDx(uvfbx);
				}
				break;
			case fbx::FbxLayerElement::eByPolygonVertex:
			{
				int copyIdx = numPositions;
				typedef std::vector<std::pair<DirectX::Vector2, int>> uvstatckelem;
				std::vector<uvstatckelem> vertexUVs(numPositions);


				//assert(leUV->GetDirectArray().GetCount() == numPolygons * 3);
				for (int ip = 0; ip < numPolygons; ip++)
				{
					for (size_t j = 0; j < 3U; j++)
					{
						auto idV = pMesh->GetPolygonVertex(ip, j);
						auto idTx = ip * 3 + j;// pMesh->GetTextureUVIndex(ip, j);

						if (euvRef == fbx::FbxLayerElement::eIndexToDirect)
							idTx = leUV->GetIndexArray().GetAt(idTx);

						auto suvfbx = leUV->GetDirectArray().GetAt(idTx); // new surface uv
						DirectX::Vector2 suv(suvfbx[0], suvfbx[1]);
						auto& uvs = vertexUVs[idV];
						bool flag = false;

						if (uvs.empty())
						{
							uvs.emplace_back(suv, idV);
							vertices[idV].textureCoordinate = suv;

							auto idTg = ip * 3 + j;
							if (etgRef == fbx::FbxLayerElement::eIndexToDirect)
								idTg = leNormal->GetIndexArray().GetAt(idTg);

							auto stgfbx = leTagent->GetDirectArray().GetAt(idTg); // new surface uv
							DirectX::Vector4 stg(stgfbx[0], stgfbx[1], stgfbx[2], stgfbx[3]);
							vertices[idV].tangent = stg;
						}
						else
						{
							for (auto& uv : uvs)
							{
								if (DirectX::Vector2::DistanceSquared(uv.first, suv) < episilon)
								{
									indices[ip * 3 + j] = uv.second;
									flag = true;
									break;
								}
							}
							if (!flag)
							{
								// This means we need to duplicate a vertex
								indices[ip * 3 + j] = numPositions;
								uvs.emplace_back(suv, numPositions++);
								vertices.push_back(vertices[idV]);
								vertices.back().textureCoordinate = suv;

								auto idTg = ip * 3 + j;
								if (etgRef == fbx::FbxLayerElement::eIndexToDirect)
									idTg = leNormal->GetIndexArray().GetAt(idTg);

								auto stgfbx = leTagent->GetDirectArray().GetAt(idTg); // new surface uv
								DirectX::Vector4 stg(stgfbx[0], stgfbx[1], stgfbx[2], stgfbx[3]);
								vertices.back().tangent = stg;
								//vertices.back().SetColor(DirectX::Colors::Green);
							}
						}
					}
				}
			}

			break;
			default:
				throw std::exception("not supported.");
				break;
			}

			SkinMeshData mesh;
			mesh.Name = pMesh->GetNode()->GetName();
			mesh.BonesCount = numBones;
			mesh.IndexCount = numIndices;
			mesh.VertexCount = numPositions;
			mesh.Vertices = new SkinMeshData::VertexType[numPositions];
			mesh.Indices = new SkinMeshData::IndexType[mesh.IndexCount];
			mesh.DefaultBoneTransforms = nullptr;

			if (numBones > 1)
			{
				if (m_Armature == nullptr)
					m_Armature = BuildArmatureFromBoneNodes();

				mesh.DefaultBoneTransforms = new DirectX::XMFLOAT4X4[numBones];
				for (int i = 0; i < numBones; i++)
				{
					DirectX::XMMATRIX boneTran = m_Armature->default_bone(i).GlobalTransform().TransformMatrix();
					DirectX::XMStore(mesh.DefaultBoneTransforms[i], boneTran);
				}
			}

			if (m_Options.RewindTriangles)
			{
				for (int i = 0; i < numPolygons; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						mesh.Indices[i * 3 + j] = indices[i * 3 + 2 - j];
					}
				}
				for (auto& v : vertices)
				{
					v.textureCoordinate.y = 1.0f - v.textureCoordinate.y;
				}
			}
			else
				std::copy(indices, indices + numIndices, mesh.Indices);
			std::copy(vertices.begin(), vertices.end(), mesh.Vertices);


			// Material
			auto pPhong = pMesh->GetNode()->GetSrcObject<fbx::FbxSurfacePhong>();
			if (pPhong)
			{
				auto& mat = mesh.Material;
				mat.Name = pPhong->GetName();
				mat.AmbientColor = ToDxColor(pPhong->Ambient.Get());
				mat.DiffuseColor = ToDxColor(pPhong->Diffuse.Get());
				mat.SpecularColor = ToDxColor(pPhong->Specular.Get());
				mat.EmissiveColor = ToDxColor(pPhong->Emissive.Get());
				auto pTexture = pPhong->Ambient.GetSrcObject<fbx::FbxFileTexture>();
				if (pTexture)
					mat.AmbientMapName = pTexture->GetFileName();
				pTexture = pPhong->Diffuse.GetSrcObject<fbx::FbxFileTexture>();
				if (pTexture)
					mat.DiffuseMapName = pTexture->GetFileName();
				pTexture = pPhong->Specular.GetSrcObject<fbx::FbxFileTexture>();
				if (pTexture)
					mat.SpecularMapName = pTexture->GetFileName();
				pTexture = pPhong->Emissive.GetSrcObject<fbx::FbxFileTexture>();
				if (pTexture)
					mat.EmissiveMapName = pTexture->GetFileName();
			}

			return mesh;
		}

		void ReorderBoneNodeByArmature()
		{
			if (m_Armature != nullptr)
			{
				decltype(m_BoneNodes) temp;
				temp.reserve(m_BoneNodes.size());

				auto getNode = [this](const Joint& joint) -> fbx::FbxNode* {
					return *std::find_if(m_BoneNodes.begin(), m_BoneNodes.end(),
						[&joint](auto pNode) ->bool {return pNode->GetName() == joint.Name;});
				};

				for (auto& j : m_Armature->joints())
				{
					temp.push_back(getNode(j));
				}

				m_BoneNodes = temp;
				//for (int i = 0; i < m_BoneNodes.size(); i++)
				//{
				//	std::cout << i << " : " << m_BoneNodes[i]->GetName() << std::endl;
				//}
			}
		}

		void FindBoneAndMesh(fbx::FbxScene* lScene)
		{
			m_MeshNodes.clear();
			m_BoneNodes.clear();
			fbx::FbxNode* lRootNode = lScene->GetRootNode();
			m_NodeIdx = 0;
			if (lRootNode) {
				for (int i = 0; i < lRootNode->GetChildCount(); i++)
					ParseNode(lRootNode->GetChild(i));
			}

			ReorderBoneNodeByArmature();
		}

		void ParseNode(fbx::FbxNode* pNode) {
			const char* nodeName = pNode->GetName();
			auto translation = pNode->LclTranslation.Get();
			auto rotation = pNode->LclRotation.Get();
			auto scaling = pNode->LclScaling.Get();

			//#if defined(_DEBUG)
			//			printf("<node name='%s' id='%d' translation='(%f, %f, %f)' rotation='(%f, %f, %f)' scaling='(%f, %f, %f)'>\n",
			//				nodeName,
			//				m_NodeIdx,
			//				translation[0], translation[1], translation[2],
			//				rotation[0], rotation[1], rotation[2],
			//				scaling[0], scaling[1], scaling[2]
			//				);
			//#endif

			bool isSkeleton = false;
			bool isMesh = false;
			// Print the node's attributes.
			for (int i = 0; i < pNode->GetNodeAttributeCount(); i++)
			{
				auto attr = pNode->GetNodeAttributeByIndex(i);
				if (attr->GetAttributeType() == FbxNodeAttribute::eSkeleton)
				{
					isSkeleton = true;
				}
				if (attr->GetAttributeType() == FbxNodeAttribute::eMesh)
				{
					isMesh = true;
				}
			}

			if (isMesh)
			{
				auto pMesh = pNode->GetMesh();
				m_MeshNodes.push_back(pMesh);
			}

			if (isSkeleton)
			{
				m_BoneNodes.push_back(pNode);
				++m_NodeIdx;
			}
			// Recursively print the children nodes.
			for (int j = 0; j < pNode->GetChildCount(); j++)
				ParseNode(pNode->GetChild(j));
		}

		StaticArmature* BuildArmatureFromBoneNodes()
		{
			auto numBones = m_BoneNodes.size();
			m_ParentMap.resize(numBones);
			std::vector<const char*> boneNames(numBones);

			for (size_t nodeIdx = 0; nodeIdx < numBones; nodeIdx++)
			{
				auto pNode = m_BoneNodes[nodeIdx];
				boneNames[nodeIdx] = pNode->GetName();

				m_ParentMap[nodeIdx] = -1;
				auto pParent = pNode->GetParent();
				while (pParent)
				{
					auto pItr = std::find(m_BoneNodes.rbegin() + (numBones - nodeIdx - 1), m_BoneNodes.rend(), pParent);
					if (pItr != m_BoneNodes.rend())
					{
						auto parentIdx = pItr - m_BoneNodes.rbegin();
						parentIdx = numBones - 1 - parentIdx;
						m_ParentMap[nodeIdx] = parentIdx;
						assert(m_BoneNodes[parentIdx] == pParent);
						break;
					}
					else
					{
						pParent = pParent->GetParent();
					}
				}
			}

			auto pArmature = new StaticArmature(numBones, m_ParentMap.data(), boneNames.data());
			BuildBindFrame(*pArmature);

			//ConstructDefaultFrameFromTimeZero(numBones, bind_frame);
			//bind_frame.RebuildGlobal(*pArmature);

			BuildJointMirrorRelation(*pArmature);
			return pArmature;
		}

		void BuildBindFrame(StaticArmature& armature)
		{
			auto &bind_frame = armature.bind_frame();
			bind_frame.resize(armature.size());

			auto lScene = m_BoneNodes[0]->GetScene();
			auto numPoses = lScene->GetPoseCount();

			for (int bid = 0; bid < m_BoneNodes.size(); bid++)
			{
				bool found = false;
				auto node = m_BoneNodes[bid];
				for (int poseidx = 0; poseidx < numPoses; poseidx++)
				{
					auto pose = lScene->GetPose(poseidx);
					assert(pose->IsBindPose());
					auto pid = pose->Find(node);
					if (pid != -1)
					{
						auto nonaffine = pose->GetMatrix(pid);
						auto& affine = reinterpret_cast<const fbx::FbxAMatrix&>(nonaffine);
						SetBoneGblFromFbxMatrix(bind_frame[bid], affine);
						found = true;
						break;
					}
				}

				if (!found) // Not found the bind pose, fall back to use the gbl transform at time-zero
				{
					auto affine = node->EvaluateGlobalTransform();
					SetBoneGblFromFbxMatrix(bind_frame[bid], affine);
				}
			}
			FrameRebuildLocal(armature, bind_frame);
		}

		void ConstructDefaultFrameFromTimeZero(const size_t &numBones, Causality::ArmatureFrame & bind_frame)
		{
			for (size_t nodeIdx = 0; nodeIdx < numBones; nodeIdx++)
			{
				auto pNode = m_BoneNodes[nodeIdx];

				auto lclM = pNode->EvaluateLocalTransform();
				auto t = lclM.GetT();
				auto q = lclM.GetQ();
				auto s = lclM.GetS();
				auto& bone = bind_frame[nodeIdx];
				bone.LclRotation = Quaternion(q[0], q[1], q[2], q[3]);//q;
				bone.LclScaling = Vector3(s[0], s[1], s[2]);
				bone.LclTranslation = Vector3(t[0], t[1], t[2]);
				auto glbM = pNode->EvaluateGlobalTransform();
				auto gt = glbM.GetT();
				auto gq = glbM.GetQ();
				auto gs = glbM.GetS();
				bone.GblRotation = Quaternion(gq[0], gq[1], gq[2], gq[3]);//q;
				bone.GblScaling = Vector3(gs[0], gs[1], gs[2]);
				bone.GblTranslation = Vector3(gt[0], gt[1], gt[2]);
				//bone.LclTranslation = DirectX::XMVector3Rotate(DirectX::XMLoadA(bone.LclTranslation), DirectX::XMQuaternionInverse(DirectX::XMLoadA(bone.LclRotation)));
			}
		}

		FbxScene* ImportSceneFromFile(const string& file)
		{
			if (sw_SdkManager.expired())
			{
				m_SdkManger.reset(fbx::FbxManager::Create(),
					[](fbx::FbxManager* sdkManager) { sdkManager->Destroy();});
				sw_SdkManager = m_SdkManger;

				// Create the io settings object.
				auto ios = fbx::FbxIOSettings::Create(m_SdkManger.get(), IOSROOT);
				m_SdkManger->SetIOSettings(ios);
			}
			else
			{
				m_SdkManger = sw_SdkManager.lock();
			}

			auto lSdkManager = m_SdkManger.get();
			auto lFilename = file.c_str();

			// Create an importer using our sdk manager.
			fbx::FbxImporter* lImporter = fbx::FbxImporter::Create(lSdkManager, "");

			// Use the first argument as the filename for the importer.
			if (!lImporter->Initialize(lFilename, -1, lSdkManager->GetIOSettings())) {
				printf("Call to FbxImporter::Initialize() failed.\n");
				printf("Please check the file path : ""%s""", lFilename);
				return nullptr;
			}

			// Create a new scene so it can be populated by the imported file.
			fbx::FbxScene* lScene = FbxScene::Create(lSdkManager, "default_scene");

			// Import the contents of the file into the scene.
			lImporter->Import(lScene);

			// Convert Axis System to what is used in this example, if needed
			if (m_Options.FileAxisSystem != AxisSystem::Unknown)
			{
				fbx::FbxAxisSystem axisSys;
				if (m_Options.FileAxisSystem == AxisSystem::DirectX)
				{
					axisSys = fbx::FbxAxisSystem(fbx::FbxAxisSystem::eYAxis, static_cast<fbx::FbxAxisSystem::EFrontVector>(-(fbx::FbxAxisSystem::eParityOdd)), fbx::FbxAxisSystem::eRightHanded);
				}
				else
				{
					auto epdfa = static_cast<fbx::FbxAxisSystem::EPreDefinedAxisSystem>(m_Options.FileAxisSystem);
					axisSys = fbx::FbxAxisSystem(epdfa);
				}

				lScene->GetGlobalSettings().SetAxisSystem(axisSys);
			}

			auto gblAxis = lScene->GetGlobalSettings().GetAxisSystem();
			fbx::FbxAxisSystem axisSys(fbx::FbxAxisSystem::EUpVector::eYAxis, fbx::FbxAxisSystem::EFrontVector::eParityOdd, fbx::FbxAxisSystem::ECoordSystem::eRightHanded);
			if (gblAxis != axisSys)
			{
				axisSys.ConvertScene(lScene);
			}

			// Convert Unit System to what is used in this example, if needed
			auto gblUnit = lScene->GetGlobalSettings().GetSystemUnit();
			if (gblUnit.GetScaleFactor() != 1.0)
			{
				//The unit in this example is meter.
				FbxSystemUnit::m.ConvertScene(lScene);
			}



			// The file has been imported; we can get rid of the importer.
			lImporter->Destroy();
			return lScene;
		}

		bool Load(const string & file, unsigned mode, const char* anim_name = nullptr)
		{

			auto lScene = ImportSceneFromFile(file);
			if (lScene == nullptr)
				return false;

			if (mode & (unsigned)Mode::ImportMeshs)
			{
				fbx::FbxGeometryConverter converter(m_SdkManger.get());
				auto succ = converter.Triangulate(lScene, true);
				assert(succ);
			}

			if (mode & (unsigned)Mode::CreateBehavierAndArmature)
			{
				m_Behavier = new BehavierSpace();
			}

			FindBoneAndMesh(lScene);

			if (mode & (unsigned)Mode::CreateBehavierAndArmature)
			{
				m_Armature = BuildArmatureFromBoneNodes();

				m_Behavier->SetArmature(*m_Armature);
			}

			if (mode & (unsigned)Mode::ImportMeshs)
			{
				fbx::FbxGeometryConverter converter(m_SdkManger.get());
				bool bReplace = true;

				for (auto& pMesh : m_MeshNodes)
				{
					//converter.SplitMeshesPerMaterial(lScene, true);
					//converter.EmulateNormalsByPolygonVertex(pMesh);

					if (!pMesh->IsTriangleMesh())
					{
						// Must do this before triangulating the mesh due to an FBX bug in TriangulateMeshAdvance
						int32_t LayerSmoothingCount = pMesh->GetLayerCount(fbx::FbxLayerElement::eSmoothing);
						for (int32_t i = 0; i < LayerSmoothingCount; i++)
							converter.ComputePolygonSmoothingFromEdgeSmoothing(pMesh, i);

						auto ConvertedNode = converter.Triangulate(pMesh, bReplace);
						if (ConvertedNode != nullptr && ConvertedNode->GetAttributeType() == fbx::FbxNodeAttribute::eMesh)
							pMesh = ConvertedNode->GetNode()->GetMesh();
					}

					m_SkinnedMeshes.emplace_back(BuildSkinnedMesh(pMesh));
				}
			}

			//for (auto pMesh : m_MeshNodes)
			//{
			//	pMesh->GetNode()->SetPreRotation(fbx::FbxNode::eSourcePivot, fbx::FbxVector4(0, 0, 0));
			//}

			if (mode & (unsigned)Mode::ImportAnimations)
				ImportAnimtionsToBehavierProfile(lScene, anim_name);

			m_MeshNodes.clear();
			m_BoneNodes.clear();
			lScene->Destroy(true);
			return true;
		}

		void ImportAnimtionsToBehavierProfile(fbx::FbxScene * lScene, const char* anim_name = nullptr)
		{
			//SetupAnimationPivots();
			//1.Extract the animation stacks using a pointer to an instance of the FbxScene (pScene).
			int numStacks = lScene->GetSrcObjectCount<fbx::FbxAnimStack>();
			int stackIdx = 0;
			if (anim_name != nullptr)
				stackIdx = numStacks - 1;
			for (; stackIdx < numStacks; stackIdx++)
			{
				fbx::FbxAnimStack* pAnimStack = lScene->GetSrcObject<fbx::FbxAnimStack>(stackIdx);
				auto pEvaluator = lScene->GetAnimationEvaluator();

				m_AnimationTime = pAnimStack->GetLocalTimeSpan().GetDuration();
				if (anim_name != nullptr)
					pAnimStack->SetName(anim_name);
				auto stackName = pAnimStack->GetName();

				//pAnimStack->BakeLayers(lScene->GetAnimationEvaluator(), 0, m_AnimationTime, m_FrameInterval);
				lScene->SetCurrentAnimationStack(pAnimStack);

				//double lSamplingRate = 30;
				//lScene->GetRootNode()->ConvertPivotAnimationRecursive(pAnimStack, FbxNode::eDestinationPivot, lSamplingRate);

				auto& anim = m_Behavier->AddAnimationClip(stackName);

				RasterizeFramesBuffer(pAnimStack, anim);
			}
		}

		// Two feature get in : Glbal postion / Log map of Local Rotation
		void RasterizeFramesBuffer(_In_ FbxAnimStack* pAnimStack, _Out_ BehavierSpace::animation_type& anim)
		{
			auto numBones = m_Armature->size();
			auto& buffer = anim.GetFrameBuffer();
			auto& dframe = m_Armature->bind_frame();
			auto nLayer = pAnimStack->GetMemberCount<FbxAnimLayer>();
			auto pLayer = pAnimStack->GetMember<FbxAnimLayer>();

			auto frameCount = CLIP_FRAME_COUNT;

			anim.Duration = time_seconds(pAnimStack->GetLocalTimeSpan().GetDuration().GetSecondDouble());
			auto interval = pAnimStack->GetLocalTimeSpan().GetDuration() / (double)frameCount;

			//anim.FrameInterval = time_seconds(m_FrameInterval.GetSecondDouble());
			anim.FrameInterval = anim.Duration / frameCount;

			buffer.resize(frameCount);

			std::vector<float> prev_angs(numBones);
			std::vector < XM_ALIGNATTR DirectX::Vector4,
				DirectX::AlignedAllocator < XMVECTOR >> prev_axiss(numBones);
			std::vector<std::vector<Vector3>> lclts;
			lclts.resize(numBones);
			for (auto& buf : lclts)
			{
				buf.resize(frameCount);
			}

			for (int bidx = 0; bidx < numBones; bidx++)
			{
				DirectX::XMVECTOR q = dframe[bidx].LclRotation;
				DirectX::XMVECTOR axis;
				float angle;
				DirectX::XMQuaternionToAxisAngle(&axis, &angle, q);
				prev_angs[bidx] = angle;
				prev_axiss[bidx] = axis;
			}


			fbx::FbxTime time = pAnimStack->GetLocalTimeSpan().GetStart();

			for (int i = 0; i < frameCount; i++)
			{
				auto& frame = buffer[i];
				frame.resize(numBones);
				for (size_t nodeIdx = 0; nodeIdx < numBones; nodeIdx++)
				{
					auto pNode = m_BoneNodes[nodeIdx];
					auto pParent = pNode->GetParent();
					const char *name = pNode->GetName();
					const char *parName = nullptr;
					if (pParent)
					{
						parName = pParent->GetName();
					}

					//auto lt = pNode->LclTranslation.EvaluateValue(time);
					//auto lr = pNode->LclRotation.EvaluateValue(time);
					//auto ls = pNode->LclScaling.EvaluateValue(time);

					//auto rp = pNode->RotationPivot.EvaluateValue(time);
					//auto sp = pNode->ScalingPivot.EvaluateValue(time);
					//auto rf = pNode->RotationOffset.EvaluateValue(time);
					//auto sf = pNode->ScalingOffset.EvaluateValue(time);
					//auto ptr = pNode->PostRotation.EvaluateValue(time);
					//auto prr = pNode->PreRotation.EvaluateValue(time);

					//FbxQuaternion qprr,qlr,qk, parQ;
					//qprr.ComposeSphericalXYZ(prr);
					//qlr.ComposeSphericalXYZ(lr);
					//qk = qlr * qprr;
					//qprr = qprr * qlr;

					//FbxEuler::EOrder lRotationOrder;
					//pNode->GetRotationOrder(FbxNode::eSourcePivot, lRotationOrder);
					//FbxAMatrix parM;

					//auto parM = pParent->EvaluateGlobalTransform(time);

					//auto preQ = pNode->GetPreRotation(fbx::FbxNode::EPivotSet::eSourcePivot);

					auto limits = pNode->GetRotationLimits();
					auto rmin = limits.GetMin();
					auto rmax = limits.GetMax();
					//auto curve = pNode->LclRotation.EvaluateValue(time);
					//t = curve->Evaluate(time);
					auto lclM = pNode->EvaluateLocalTransform(time);
					auto t = lclM.GetT();
					auto q = lclM.GetQ();
					auto s = lclM.GetS();

					//if (pParent)
					//	parQ = pParent->EvaluateLocalTransform(time).GetQ();

					//parM *= lclM;

					//auto glbM = pNode->EvaluateGlobalTransform(time);
					//auto gt = glbM.GetT();
					//auto gq = glbM.GetQ();
					//auto gs = glbM.GetS();

					//FbxVector4 rd(r);
					////rd *= (180.0 / DirectX::XM_PI);
					//FbxQuaternion fbxq;
					//fbxq.ComposeSphericalXYZ(rd);
					//DirectX::Quaternion q(fbxq[0], fbxq[1], fbxq[2], fbxq[3]);

					auto& bone = frame[nodeIdx];
					bone.LclRotation = Quaternion(q[0], q[1], q[2], q[3]);//q;

					// Fix Quaternion 
					{
						using namespace DirectX;
						XMVECTOR prev_axis = XMLoadA(prev_axiss[nodeIdx]);
						float prev_ang = prev_angs[nodeIdx];

						XMVECTOR q = XMLoadA(bone.LclRotation);
						XMVECTOR axis;
						float ang;
						XMQuaternionToAxisAngle(&axis, &ang, q);
						if (ang > XM_PI)
							ang -= XM_PI;

						XMVECTOR lnQ = XMQuaternionLn(q);
						XMVECTOR lnNegQ = XMQuaternionLn(-q);

						XMVECTOR dis = XMVectorZero();
						XMVECTOR flipDis = XMVectorZero();

						XMVECTOR prevQ;
						if (i > 0)
						{
							prevQ = XMLoadA(buffer[i - 1][nodeIdx].LclRotation);
						}
						else
						{
							prevQ = XMLoadA(dframe[nodeIdx].LclRotation);
						}

						flipDis = XMVector4Length(prevQ + q);
						dis = XMVector4Length(prevQ - q);

						/*if (i == 0 && ang > XM_PIDIV2)
						{
							q = -q;
							XMStoreA(bone.LclRotation, q);
							XMQuaternionToAxisAngle(&axis, &ang, q);
							if (ang > XM_PI)
								ang -= XM_PI;
						}
						else */if (/*i > 0
							&& */XMVector4Less(XMVector3Dot(prev_axis, axis), XMVectorZero())
							//&& abs(ang - prev_ang) > 0.05f

							//&& ((abs(ang + prev_ang - XM_2PI) < QuaternionFlipLimit)
							//	|| (abs(ang - prev_ang) > 0.05f && abs(ang + prev_ang) < QuaternionFlipLimit2))

							&& XMVector4Greater(dis, flipDis * 10.0f)

							)
						{
							q = -q;
							XMStoreA(bone.LclRotation, q);
							XMQuaternionToAxisAngle(&axis, &ang, q);
							if (ang > XM_PI)
								ang -= XM_PI;
						}

						prev_angs[nodeIdx] = ang;
						prev_axiss[nodeIdx] = axis;
					}

					bone.LclScaling = Vector3(s[0], s[1], s[2]);
					bone.LclTranslation = Vector3(t[0], t[1], t[2]);

					// Rotate back
					//bone.LclTranslation = DirectX::XMVector3Rotate(DirectX::XMLoadA(bone.LclTranslation), DirectX::XMQuaternionInverse(DirectX::XMLoadA(bone.LclRotation)));

					//bone.GblScaling = Vector3(gs[0], gs[1], gs[2]);
					//bone.GblRotation = Quaternion(gq[0], gq[1], gq[2], gq[3]);
					//bone.GblTranslation = Vector3(gt[0], gt[1], gt[2]);
				}
				//frame.Time = time_seconds(time.GetSecondDouble());
				FrameRebuildGlobal(*m_Armature, frame);

				time += interval;
			}

			//CaculateAnimationFeatures(anim);
			//Eigen::MatrixXf bone32(24, buffer.size());
			//for (int i = 0; i < buffer.size(); i++)
			//{
			//	using namespace DirectX;
			//	bone32.block(0, i, 4, 1) = Eigen::Vector4f::Map(&buffer[i][3].LclRotation.x);
			//	bone32.block(4, i, 4, 1) = Eigen::Vector4f::Map(&buffer[i][4].LclRotation.x);
			//	bone32.block(8, i, 4, 1) = Eigen::Vector4f::Map(&buffer[i][5].LclRotation.x);
			//	Quaternion lq = XMQuaternionLn(XMLoadA(buffer[i][3].LclRotation));
			//	bone32.block(12, i, 4, 1) = Eigen::Vector4f::Map(&lq.x);
			//	lq = XMQuaternionLn(XMLoadA(buffer[i][4].LclRotation));
			//	bone32.block(16, i, 4, 1) = Eigen::Vector4f::Map(&lq.x);
			//	lq = XMQuaternionLn(XMLoadA(buffer[i][5].LclRotation));
			//	bone32.block(20, i, 4, 1) = Eigen::Vector4f::Map(&lq.x);
			//}

			//string animName = pAnimStack->GetName();
			//string sceneName = pAnimStack->GetScene()->GetName();
			//using namespace std::string_literals;
			//std::ofstream ofs(sceneName + '_' + animName + ".csv");

			//for (int i = 0; i < frameCount; i++)
			//{
			//	for (int j = 1; j < numBones; j++)
			//	{
			//		DirectX::Vector3 roted;
			//		auto& bone = buffer[i][j];
			//		auto& pbone = buffer[i][m_Armature->at(j)->ParentID];
			//		roted = bone.LclTranslation;
			//		ofs << roted.x << ',' << roted.y << ',' << roted.z << ',';
			//		//roted = DirectX::XMVector3InverseRotate(bone.LclTranslation, bone.LclRotation);
			//		//ofs << roted.x << ',' << roted.y << ',' << roted.z << ',';
			//		//roted = DirectX::XMVector3Rotate(bone.LclTranslation, bone.LclRotation);
			//		//ofs << roted.x << ',' << roted.y << ',' << roted.z << ",|,";
			//	}
			//	ofs << std::endl;
			//}
			//ofs.close();
			//std::cout << lclts[0][0];
		}
	};


	FbxParser::AxisSystem FbxParser::ParseAxisSystem(const char * name)
	{
		if (!name)
			return AxisSystem::Unknown;

		gsl::cstring_span<> sv(name, strlen(name));

#define MAKE_OPTION_AXIS_SYS(Enum) if (sv == #Enum) return AxisSystem::Enum;
#define MAKE_ELSE_OPTION_AXIS_SYS(Enum) else MAKE_OPTION_AXIS_SYS(Enum)

		MAKE_OPTION_AXIS_SYS	 (MayaZUp)
		MAKE_ELSE_OPTION_AXIS_SYS(MayaYUp)
		MAKE_ELSE_OPTION_AXIS_SYS(Max)
		MAKE_ELSE_OPTION_AXIS_SYS(MotionBuilder)
		MAKE_ELSE_OPTION_AXIS_SYS(OpenGL)
		MAKE_ELSE_OPTION_AXIS_SYS(DirectX)
		MAKE_ELSE_OPTION_AXIS_SYS(Lightwave)
		return AxisSystem::Unknown;
	}

	FbxParser::FbxParser(const ParamArchive * parameters)
	{
		m_pImpl.reset(new FbxParser::Impl);

		if (!parameters) 
			return;
		const char * axisname = nullptr;
		GetParam(parameters, "file_axis_system", axisname);
		auto axisSys = ParseAxisSystem(axisname);

		SetFileAxisSystemOverhaul(axisSys);
	}

	FbxParser::FbxParser(const string & file, unsigned mode)
	{
		Load(file, mode);
	}

	void FbxParser::SetBehavierProfile(BehavierSpace * pBehav)
	{
		m_pImpl->m_Behavier = pBehav;
	}

	bool FbxParser::Load(const string & file, unsigned mode)
	{
		if (!m_pImpl)
			m_pImpl.reset(new FbxParser::Impl);
		return m_pImpl->Load(file, mode);
	}

	bool FbxParser::ImportBehavier(const string & file)
	{
		if (!m_pImpl)
			m_pImpl.reset(new FbxParser::Impl);
		return m_pImpl->Load(file, (unsigned)Mode::CreateBehavierAndArmature | (unsigned)Mode::ImportAnimations);
	}

	bool FbxParser::ImportArmature(const string & file)
	{
		if (!m_pImpl)
			m_pImpl.reset(new FbxParser::Impl);
		return m_pImpl->Load(file, (unsigned)Mode::CreateBehavierAndArmature);
	}

	bool FbxParser::ImportMesh(const string & file, bool rewind , bool flipNormal)
	{
		if (!m_pImpl)
			m_pImpl.reset(new FbxParser::Impl);
		// SKIN MESH DEBUG!!!
		//m_pImpl->m_SkinnedMeshes.emplace_back();
		//auto& mesh = m_pImpl->m_SkinnedMeshes.back();

		m_pImpl->m_Options.RewindTriangles = rewind;
		m_pImpl->m_Options.FlipNormals = flipNormal;
		return m_pImpl->Load(file, (unsigned)Mode::ImportMeshs);
	}

	bool FbxParser::ImportAnimation(const string & file, const string & animationName)
	{
		if (!m_pImpl)
			m_pImpl.reset(new FbxParser::Impl);
		return m_pImpl->Load(file, (unsigned)Mode::ImportAnimations, animationName.c_str());
	}

	void FbxParser::SetAdditionalTransform(const IsometricTransform & transform)
	{
		m_pImpl->m_Options.AdditionalTransform = transform;
	}

	void FbxParser::SetFileAxisSystemOverhaul(AxisSystem axis)
	{
		m_pImpl->m_Options.FileAxisSystem = axis;
	}

	const std::list<SkinMeshData>& FbxParser::GetMeshs()
	{
		return m_pImpl->m_SkinnedMeshes;
	}

	StaticArmature * FbxParser::GetArmature()
	{
		return m_pImpl->m_Armature;
	}

	BehavierSpace* FbxParser::GetBehavier()
	{
		return m_pImpl->m_Behavier;
	}

	FbxParser::~FbxParser()
	{

	}
	FbxParser::FbxParser()
	{

	}
}