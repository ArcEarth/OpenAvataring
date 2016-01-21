#pragma once
#include "ArmatureParts.h"
#include "BoneFeatures.h"

namespace Causality
{
	namespace ArmaturePartFeatures
	{
		template <class _BoneFeatureType>
		class AllJoints : public IArmaturePartFeature
		{
		public:
			typedef _BoneFeatureType BoneFeatureType;

			AllJoints() = default;

			AllJoints(const AllJoints&) = default;

			int GetDimension() const override {
				return -1;
			}

			int GetDimension(_In_ const ArmaturePart& block) const override
			{
				return block.Joints.size() * BoneFeatureType::Dimension;
			}

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				Eigen::RowVectorXf Y(block.Joints.size() * BoneFeatureType::Dimension);
				for (size_t j = 0; j < block.Joints.size(); j++)
				{
					auto jid = block.Joints[j]->ID;
					auto Yj = Y.segment<BoneFeatureType::Dimension>(j * BoneFeatureType::Dimension);
					BoneFeatureType::Get(Yj, frame[jid]);
				}

				return Y;
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{
				for (size_t j = 0; j < block.Joints.size(); j++)
				{
					auto jid = block.Joints[j]->ID;
					auto Xj = feature.segment<BoneFeatureType::Dimension>(j * BoneFeatureType::Dimension);
					BoneFeatureType::Set(frame[jid], Xj);
				}
			}
		};

		// Joints feature with pca
		template <class PartFeatureType>
		class Pcad : public PartFeatureType
		{
		public:
			static_assert(std::is_base_of<IArmaturePartFeature, PartFeatureType>::value, "PartFeatureType must be derived type of IArmaturePartFeature");

			std::vector<Eigen::MatrixXf> m_pcas;
			std::vector<Eigen::RowVectorXf> m_means;

			int GetDimension() const override {
				return -1;
			}

			int GetDimension(_In_ const ArmaturePart& block) const
			{
				auto bid = block.Index;
				return m_pcas[bid].cols();
			}

			Eigen::MatrixXf& GetPca(int bid) { return m_pcas[bid]; }
			const Eigen::MatrixXf& GetPca(int bid) const { return m_pcas[bid]; }

			Eigen::RowVectorXf& GetMean(int bid) { return m_means[bid]; }
			const Eigen::RowVectorXf& GetMean(int bid) const { return m_means[bid]; }
			void InitPcas(int size) {
				m_pcas.resize(size);
				m_means.resize(size);
			}
			template <class DerivedPca, class DerivedMean>
			void SetPca(int bid, const Eigen::MatrixBase<DerivedPca>& principleComponents, const Eigen::MatrixBase<DerivedMean>& mean) {
				assert(!m_means.empty() && "Call InitPcas before SetPca");
				m_means[bid] = mean;
				m_pcas[bid] = principleComponents;
			}

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				auto X = PartFeatureType::Get(block, frame);

				auto bid = block.Index;
				return (X - m_means[bid]) * m_pcas[bid];
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{
				auto bid = block.Index;
				auto X = (feature * m_pcas[bid].transpose() + m_means[bid]).eval();
				PartFeatureType::Set(block, frame, X);
			}
		};

		template <class PartFeatureType>
		class WithVelocity : public PartFeatureType
		{
		public:
			static_assert(std::is_base_of<IArmaturePartFeature, PartFeatureType>::value, "PartFeatureType must be derived type of IArmaturePartFeature");

			int GetDimension() const override {
				int dim = PartFeatureType::GetDimension();
				return dim < 0 ? -1 : dim * 2;
			}

			int GetDimension(_In_ const ArmaturePart& block) const
			{
				return PartFeatureType::GetDimension(block) * 2;
			}

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame, _In_ ArmatureFrameConstView last_frame, float frame_time) override
			{
				int dim = PartFeatureType::GetDimension(block);
				Eigen::RowVectorXf Y(dim * 2);

				Y.segment(0, dim) = PartFeatureType::Get(block, frame);
				Y.segment(dim, dim) = Y.segment(0, dim);
				Y.segment(dim, dim) -= PartFeatureType::Get(block, last_frame);
				Y.segment(dim, dim) /= frame_time;

				return Y;
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{
				int dim = PartFeatureType::GetDimension(block);
				assert(feature.cols() == dim * 2);
				PartFeatureType::Set(block, frame, feature.segment(0, dim));
			}
		};

		template <class PartFeatureType>
		class NormalizeVelocity : public WithVelocity<PartFeatureType>
		{
		private:
			float m_VelThred;
		public:
			NormalizeVelocity()
				: m_VelThred(0.01f)
			{
			}

			float GetVelocityThreshold() const
			{
				return m_VelThred;
			}

			void SetVelocityThreshold(float threshold)
			{
				m_VelThred = threshold;
			}

			using BaseType = WithVelocity<PartFeatureType>;
			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame, _In_ ArmatureFrameConstView last_frame, float frame_time) override
			{
				int dim = PartFeatureType::GetDimension(block);
				Eigen::RowVectorXf Y = BaseType::Get(block,frame,last_frame,frame_time);

				auto vel = Y.segment(dim, dim);
				auto norm = vel.norm();

				if (norm > m_VelThred)
					vel /= norm;
				else
					vel.setZero();

				return Y;
			}
		};

		template <class PartFeatureType>
		class Localize : public PartFeatureType
		{
		public:
			static_assert(std::is_base_of<IArmaturePartFeature, PartFeatureType>::value, "PartFeatureType must be derived type of IArmaturePartFeature");

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				Eigen::RowVectorXf Y = PartFeatureType::Get(block, frame);

				if (block.parent() != nullptr)
				{
					Eigen::RowVectorXf ref = PartFeatureType::Get(*block.parent(), frame);
					Y -= ref;
				}
				return Y;
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{
				assert(!"Localize could not be use to set frame");
			}
		};

		template <class BoneFeatureType>
		class Localize<AllJoints<BoneFeatureType>> : public AllJoints<BoneFeatureType>
		{
			typedef AllJoints<BoneFeatureType> PartFeatureType;
			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				Eigen::RowVectorXf Y = PartFeatureType::Get(block, frame);

				if (block.parent() != nullptr)
				{
					Eigen::RowVectorXf ref(BoneFeatureType::Dimension);
					BoneFeatureType::Get(ref, frame[block.parent()->Joints.back()->ID]);
					Y -= ref.replicate(1, block.Joints.size());
				}
				return Y;
			}
		};


		template <class _BoneFeatureType>
		class EndEffector : public IArmaturePartFeature
		{
		public:
			typedef _BoneFeatureType BoneFeatureType;
			EndEffector() = default;

			int GetDimension() const override{
				return BoneFeatureType::Dimension;
			}

			int GetDimension(_In_ const ArmaturePart& block) const override
			{
				return BoneFeatureType::Dimension;
			}

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				Eigen::RowVectorXf Y(BoneFeatureType::Dimension);

				BoneFeatureType::Get(Y, frame[block.Joints.back()->ID]);

				return Y;
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{
				assert(!"End Effector only block feature could not be use to set frame");
			}
		};
		
		template <class PartFeatureType>
		class Weighted : public PartFeatureType
		{
		public:
			static_assert(std::is_base_of<IArmaturePartFeature, PartFeatureType>::value, "PartFeatureType must be derived type of IArmaturePartFeature");
			using PartFeatureType::BoneFeatureType;

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				Eigen::RowVectorXf Y = PartFeatureType::Get(block, frame);
				Y.array() *= m_Weights.segment(m_Idices[block.Index], Y.size()).array();
				return Y;
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{

				Eigen::RowVectorXf Y = feature;
				Y.array() /= m_Weights.segment(m_Idices[block.Index], Y.size()).array();
				PartFeatureType::Set(block, frame, Y);
			}

			void InitializeWeights(const ShrinkedArmature& parts)
			{
				m_Idices.resize(parts.size());
				m_Idices[0] = 0;
				for (int i = 0; i < parts.size() - 1; i++)
				{
					m_Idices[i + 1] = m_Idices[i] + PartFeatureType::GetDimension(*parts[i]);
				}

				const auto bDim = BoneFeatureType::Dimension;
				//! Only works for ALLJoint
				m_Weights.resize(parts.Armature().size() * bDim);

				for (int jid = 0, i = 0; i < parts.size(); i++)
				{
					for (int j = 0; j < parts[i]->Joints.size(); j++)
					{
						m_Weights.segment(jid * bDim, bDim).setConstant(parts[i]->Wxj[j]);
						++jid;
					}
				}
			}

		private:
			Eigen::VectorXi	   m_Idices;
			Eigen::RowVectorXf m_Weights;
		};

		template <class PartFeatureType>
		class RelativeDeformation : public PartFeatureType
		{
		public:
			static_assert(std::is_base_of<IArmaturePartFeature, PartFeatureType>::value, "PartFeatureType must be derived type of IArmaturePartFeature");

			using PartFeatureType::PartFeatureType;

			void SetDefaultFrame(ArmatureFrameConstView frame)
			{
				m_rframe = m_rlframe = m_dframe = m_dframeInv = frame;
				for (auto& bone : m_dframeInv)
				{
					bone.LocalTransform().Inverse();
					bone.GlobalTransform().Inverse();
				}
			}

		protected:
			void GetRelativeFrame(const ArmaturePart & block, ArmatureFrameConstView frame)
			{
				for (auto joint : block.Joints)
				{
					auto i = joint->ID;
					auto& lt = m_rframe[i];
					lt.LocalTransform() = m_dframeInv[i].LocalTransform();
					lt.LocalTransform() *= frame[i].LocalTransform();

					lt.GlobalTransform() = m_dframeInv[i].LocalTransform();
					lt.GlobalTransform() *= frame[i].GlobalTransform();
				}
			}

			void SetAbsoluteFrame(const ArmaturePart & block, _Out_ ArmatureFrameView frame)
			{
				for (auto joint : block.Joints)
				{
					auto i = joint->ID;
					auto& lt = frame[i];
					lt.LocalTransform() = m_dframe[i].LocalTransform();
					lt.LclRotation *= m_rframe[i].LclRotation;
				}
			}

		public:
			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame) override
			{
				GetRelativeFrame(block, frame);
				return PartFeatureType::Get(block, m_rframe);
			}

			virtual Eigen::RowVectorXf Get(_In_ const ArmaturePart& block, _In_ ArmatureFrameConstView frame, _In_ ArmatureFrameConstView last_frame, float frame_time) override
			{
				GetRelativeFrame(block, frame);
				return PartFeatureType::Get(block, m_rframe);
			}

			virtual void Set(_In_ const ArmaturePart& block, _Out_ ArmatureFrameView frame, _In_ const Eigen::RowVectorXf& feature) override
			{
				PartFeatureType::Set(block, m_rframe, feature);
				SetAbsoluteFrame(block, frame);
			}

		private:
			ArmatureFrame m_dframe, m_dframeInv, m_rframe, m_rlframe;
		};
	}
}