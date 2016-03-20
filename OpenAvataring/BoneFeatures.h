#pragma once
#include "Causality\Armature.h"

namespace Causality
{
	namespace BoneFeatures
	{
		struct LclRotEulerAngleFeature
		{
			static const size_t Dimension = 3;

			template <class Derived>
			inline static void Get(_Out_ Eigen::DenseBase<Derived>& fv, _In_ const Bone& bone)
			{
				XM_ALIGNATTR Vector3 qs;
				DirectX::XMVECTOR q = XMLoadA(bone.LclRotation);
				q = DirectX::XMQuaternionEulerAngleYawPitchRoll(q);
				//q *= bone.GblLength;
				XMStoreA(qs, q);
				fv = Eigen::Vector3f::Map(&qs.x);
			}

			template <class Derived>
			inline static void Set(_Out_ Bone& bone, _In_ const Eigen::DenseBase<Derived>& fv)
			{
				// ensure continious storage
				Eigen::Vector3f cfv = fv;
				DirectX::XMVECTOR q = DirectX::XMLoadFloat3(reinterpret_cast<const DirectX::XMFLOAT3 *>(cfv.data()));
				//q /= bone.GblLength;
				q = DirectX::XMQuaternionRotationRollPitchYawFromVector(q);
				XMStoreA(bone.LclRotation, q);
			}
		};

		struct LclRotLnQuatFeature //: concept BoneFeature 
		{
			static const size_t Dimension = 3;

			template <class Derived>
			inline static void Get(_Out_ Eigen::DenseBase<Derived>& fv, _In_ const Bone& bone)
			{
				XM_ALIGNATTR Vector3 qs;
				using namespace DirectX;
				DirectX::XMVECTOR q;
				q = XMLoadA(bone.LclRotation);

				if (bone.LclRotation.w < .0f)
					q = XMVectorNegate(q);

				q = DirectX::XMQuaternionLn(q);
				//q *= bone.GblLength;
				XMStoreA(qs,q);
				fv = Eigen::Vector3f::Map(&qs.x);
			}

			template <class Derived>
			inline static void Set(_Out_ Bone& bone, _In_ const Eigen::DenseBase<Derived>& fv)
			{
				// ensure continious storage
				Eigen::Vector3f cfv = fv;
				DirectX::XMVECTOR q = DirectX::XMLoadFloat3(reinterpret_cast<const DirectX::XMFLOAT3 *>(cfv.data()));
				//q /= bone.GblLength;
				q = DirectX::XMQuaternionExp(q);
				XMStoreA(bone.LclRotation,q);
			}
		};

		struct LclTRFeature
		{
			static const size_t Dimension = 6;
			static const bool EnableBlcokwiseLocalization = false;
			typedef Eigen::Matrix<float, Dimension, 1> VectorType;

			template <class Derived>
			inline static void Get(_Out_ Eigen::DenseBase<Derived>& fv, _In_ const Bone& bone)
			{
				assert(fv.size() == Dimension);
				using DirectX::operator*=;
				DirectX::XMVECTOR q = XMLoadA(bone.LclRotation);
				q = DirectX::XMQuaternionLn(q);

				//! IMPORTANT
				q *= bone.GblLength;

				fv.segment<3>(0) = Eigen::Vector3f::Map(q.m128_f32);
				fv.segment<3>(3) = Eigen::Vector3f::Map(&bone.LclTranslation.x);
			}

			inline static VectorType Get(_In_ const Bone& bone)
			{
				VectorType fv;
				Get(fv, bone);
				return fv;
			}

			template <class Derived>
			inline static void Set(_Out_ Bone& bone, _In_ const Eigen::DenseBase<Derived>& fv)
			{
				// ensure continious storage
				VectorType cfv = fv;
				DirectX::XMVECTOR q = DirectX::XMLoadFloat3(reinterpret_cast<const DirectX::XMFLOAT3 *>(cfv.data()));

				//! IMPORTANT
				q /= bone.GblLength;

				q = DirectX::XMQuaternionExp(q);
				XMStoreA(bone.LclRotation,q);
				bone.LclTranslation = reinterpret_cast<const DirectX::Vector3&>(cfv.data()[3]);
			}
		};

		struct GblPosLclRotFeature
		{
			static const size_t Dimension = 6;
			typedef Eigen::Matrix<float, Dimension, 1> VectorType;

			template <class Derived>
			inline static void Get(_Out_ Eigen::DenseBase<Derived>& fv, _In_ const Bone& bone)
			{
				assert(fv.size() == Dimension);
				using DirectX::operator*=;
				DirectX::XMVECTOR q = XMLoadA(bone.LclRotation);
				q = DirectX::XMQuaternionLn(q);

				//! IMPORTANT
				q *= bone.GblLength;

				fv.segment<3>(0) = Eigen::Vector3f::Map(q.m128_f32);
				fv.segment<3>(3) = Eigen::Vector3f::Map(&bone.GblTranslation.x);
			}

			inline static VectorType Get(_In_ const Bone& bone)
			{
				VectorType fv;
				Get(fv, bone);
				return fv;
			}

			template <class Derived>
			inline static void Set(_Out_ Bone& bone, _In_ const Eigen::DenseBase<Derived>& fv)
			{
				// ensure continious storage
				VectorType cfv = fv;
				DirectX::XMVECTOR q = DirectX::XMLoadFloat3(reinterpret_cast<const DirectX::XMFLOAT3 *>(cfv.data()));

				//! IMPORTANT
				q /= bone.GblLength;

				q = DirectX::XMQuaternionExp(q);
				XMStoreA(bone.LclRotation,q);
				bone.GblTranslation = reinterpret_cast<const DirectX::Vector3&>(cfv.data()[3]);
			}
		};

		struct GblPosFeature
		{
			static const size_t Dimension = 3;

			template <class Derived>
			inline static void Get(_Out_ Eigen::DenseBase<Derived>& fv, _In_ const Bone& bone)
			{
				fv = Eigen::Vector3f::Map(&bone.GblTranslation.x);
			}

			template <class Derived>
			inline static void Set(_Out_ Bone& bone, _In_ const Eigen::DenseBase<Derived>& fv)
			{
				// ensure continious storage
				Eigen::Vector3f::Map(&bone.GblTranslation.x) = fv;
			}
		};

		struct QuadraticGblPosFeature
		{
			static const size_t Dimension = 9;

			template <class Derived>
			inline static void Get(_Out_ Eigen::DenseBase<Derived>& fv, _In_ const Bone& bone)
			{
				auto& pos = bone.GblTranslation;
				fv.segment<3>(0) = Eigen::Vector3f::Map(&pos.x);
				fv[3] = pos.x * pos.x;
				fv[4] = pos.y * pos.y;
				fv[5] = pos.z * pos.z;
				fv[6] = pos.x * pos.y;
				fv[7] = pos.y * pos.z;
				fv[8] = pos.z * pos.x;
			}

			template <class Derived>
			inline static void Set(_Out_ Bone& bone, _In_ const Eigen::DenseBase<Derived>& fv)
			{
				// ensure continious storage
				Eigen::Vector3f::Map(&bone.GblTranslation.x) = fv.segment<3>(0);
			}
		};
	}

	// Orientation data is wayyyyyyyy toooooo noisy
	typedef BoneFeatures::GblPosFeature		  InputFeature;
	typedef BoneFeatures::LclRotLnQuatFeature CharacterFeature;
}