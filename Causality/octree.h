#pragma once

#include "Common\DirectXMathExtend.h"
#include <Eigen\Dense>

template <typename _Ty, typename _TScalar = float>
class octree
{
	template <typename _TScalar>
	struct vector_traits
	{
	};

	template <>
	struct vector_traits<float>
	{
		typedef DirectX::Vector3 vector_type;
	};

	template <>
	struct vector_traits<double>
	{
		typedef Eigen::Vector3d vector_type;
	};

	template <>
	struct vector_traits<int>
	{
		typedef Eigen::Vector3i vector_type;
	};

public:
	typedef typename vector_traits<_TScalar>::vector_type vector_type;
	typedef std::remove_reference_t<_Ty>	value_type;
	typedef const value_type&				const_reference;
	typedef value_type&						reference;

	struct node_type
	{
		// control block
		node_type*		parent;
		node_type*		children[8];
		vector_type		min;
		vector_type		max;
		bool			is_leaf; // if this node contains an entity
	};

	struct leaf_type : public node_type
	{
		value_type	entity;
	};

	bool insert_at(const vector_type& center, const_reference );
	bool insert_at(const vector_type& center, const vector_type& extent, const_reference);
};