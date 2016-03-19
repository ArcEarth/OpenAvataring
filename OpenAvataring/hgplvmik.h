#pragma once
#include "GaussianProcess.h"
#include "StylizedIK.h"
#include "ArmatureTransforms.h"

#include <Causality\SmartPointers.h>

namespace Causality
{
	class CharacterController;

	class HgplvmTransformer;

	class IHgplvmNode
	{
		virtual double likilihood(const Eigen::RowVectorXd& x);
		virtual double likilihood(const Eigen::RowVectorXd& x, Vector3 position);
		virtual double likilihood(const Eigen::RowVectorXd& x, double time_delta);
	};

	class HgplvmIkController
	{
	public:
		bool m_isLeaf;
		bool m_isRoot;
		int  m_parentId;

		union
		{
			StylizedChainIK m_sik;
			gplvm			m_gplvm;
			gpr				m_gpr;
		};

		struct ChildInfo
		{
			int Index;
			int Dimension;
			int Start;
		};

		std::vector<ChildInfo> m_children;

		void Initialize(CharacterController& characon);

	};

	class HgplvmTransformer
	{
	public:
		using row_vector_t = Eigen::RowVectorXd;
		using effector_vector_t = std::vector<std::pair<Vector3, int>>;
		using row_vector_block = Eigen::Block<row_vector_t, 1, -1>;

		//CharacterController&			m_controller;
		//CharacterClipinfo&				m_clipinfo;
		//PartilizedTransformer&			m_partTransformer;
		//vector<P2PTransform>&			m_activeTransformers;

		vector<HgplvmIkController>		m_nodes;

		sptr<IArmaturePartFeature>		m_feature;

		double learning_likilihood(const row_vector_t);
		double learning_likilihood_derv(const row_vector_t);

		double effctor_likilihood(int idx, const row_vector_t& x, Vector3 position) const;
		double node_likilihood(int idx, const row_vector_t& x) const;
		double action_likilihood(int idx, const row_vector_t& x, double time_delta) const;

		row_vector_t effctor_likilihood_derv(int idx, const row_vector_t& x, Vector3 position) const;
		row_vector_t node_likilihood_derv(int idx, const row_vector_t& x) const;
		row_vector_t action_likilihood_derv(int idx, const row_vector_t& x, double time_delta) const;

		row_vector_block getNodeInput(int idx, const row_vector_t& x) const;

		double likihood(row_vector_t &state, double time_delta, const effector_vector_t& effectors) const
		{


			for (auto& node : m_nodes)
			{
				node.m_isLeaf;
			}
		}

		double likihood_derv(row_vector_t &state, double time_delta, const effector_vector_t& effectors) const
		{

		}

		double transform(row_vector_t &state, double time_delta, const effector_vector_t& effectors) const
		{

		}

		HgplvmTransformer(CharacterController& controller)
		{

		}

	};


}