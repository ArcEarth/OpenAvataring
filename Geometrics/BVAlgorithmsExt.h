#pragma once
#include <Eigen\Core>
#include <unsupported\Eigen\BVH>
#include <vector>

namespace Eigen
{
	namespace internal
	{
		template<typename BVH, typename Intersectable>
		class bvh_intersect_iterator
		{
		public:
			typedef typename BVH::Index Index;
			typedef typename BVH::VolumeIterator VolIter;
			typedef typename BVH::ObjectIterator ObjIter;

		private:
			const BVH &		m_tree;
			Intersectable &	m_intersector;

			// frame stack
			std::vector<Index> m_todo;

		public:
			bvh_intersect_iterator(const BVH &tree, Intersectable &intersector, typename BVH::Index root)
				: m_tree(tree), m_intersector(intersector), m_todo(1, root)
			{
			}

			bool next()
			{
				VolIter vBegin = VolIter(), vEnd = VolIter();
				ObjIter oBegin = ObjIter(), oEnd = ObjIter();

				while (!todo.empty())
				{
					m_tree.getChildren(m_todo.back(), vBegin, vEnd, oBegin, oEnd);

					m_todo.pop_back();

					for (; vBegin != vEnd; ++vBegin) //go through child volumes
						if (m_intersector.intersectVolume(m_tree.getVolume(*vBegin)))
							m_todo.push_back(*vBegin);

					for (; oBegin != oEnd; ++oBegin) //go through child objects
						if (m_intersector.intersectObject(*oBegin))
							return true; //intersector said to stop query			
				}
				return false;
			}

			Index current() const { return todo.back(); }
		};
	}
}