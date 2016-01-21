#pragma once
//#include "DirectXMathExtend.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <iterator>

// We need AlignedBox 
#include <Eigen\Core>
#include <Eigen\src\Geometry\AlignedBox.h>
#include <iterator_range.h>

namespace Geometrics {
	using std::iterator_range;

	//concept ExtBVH
	//{
	//  Typedefs:
	//	typedef Index;
	//	typedef Object;
	//	typedef Volume;
	//	typedef VolumeIterator;
	//  (*VolumeIterator == Index)
	//	typedef ObjectIterator;
	//  
	//  Interfaces:
	//	Index getRootIndex() const;
	//	bool isObject(Index index) const;
	//	const Object& getObject(Index index) const;
	//	const Volume& getVolume(Index index) const;
	//	bool getChildren(Index index, VolumeIterator &outVBegin, VolumeIterator &outVEnd) const;
	//};

	namespace internal {
		//internal pair class for the BVH--used instead of std::pair because of alignment
		template<typename Scalar, int Dim>
		struct box_int_pair
		{
			typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;

			typedef Eigen::AlignedBox<Scalar, Dim> BoxType;

			box_int_pair() = default;
			box_int_pair(const VectorType &c, VectorType & e, int i) : center(c),extent(e), index(i) {}

			_MM_ALIGN16
			VectorType center;
			_MM_ALIGN16
			VectorType extent;
			int index;

			inline BoxType getBox() const
			{
				return BoxType(center - extent, center + extent);
			}
		};
	} // end namespace internal

	// A subtree indicate a node in the tree
	// It also have the interface for this sub-tree that rooted with that node
	// Ensentially a pointer to a tree and a index
	// Can be used as a base for iterator
	template <typename _TBVH>
	struct SubBvh
	{
	public:
		typedef _TBVH BVHType;
		typedef BVHType  TreeType;
		typedef typename TreeType::Index Index;
		typedef typename TreeType::Volume Volume;
		typedef typename TreeType::Object Object;
		typedef typename TreeType::VolumeIterator VolumeIterator;
		typedef typename TreeType::ObjectIterator ObjectIterator;
	protected:
		const TreeType*	_tree;
		Index			_index;

	public:
		SubBvh(const TreeType& tree)
			: _tree(&tree), _index(tree.getRootIndex())
		{
		}

		SubBvh(const TreeType& tree, Index index)
			: _tree(&tree), _index(index)
		{
		}

		const TreeType& getTree() const
		{
			return *_tree;
		}

		Index getIndex() const {
			return _index;
		}

		const Volume& getVolume() const
		{
			return _tree->getVolume(_index);
		}

		const Object& getObject() const
		{
			return _tree->getObject(_index);
		}

		bool isObject() const
		{
			return _tree->isObject(_index);
		}

		bool getChildren(VolumeIterator &outVBegin, VolumeIterator &outVEnd) const
		{
			return _tree->getChildren(_index, outVBegin, outVEnd);
		}

		// Since each node in BVH is also a BVH, we give it this interface
#pragma region BVH-Interface
		inline Index getRootIndex() const
		{
			return _index;
		}

		inline bool isObject(Index index) const
		{
			return _tree->isObject(index);
		}

		inline const Object& getObject(Index index) const
		{
			return _tree->isObject(index);
		}

		inline bool getChildren(Index index, VolumeIterator &outVBegin, VolumeIterator &outVEnd)
		{
			return _tree->getChildren(index, outVBegin, outVEnd);
		}

		inline void getChildren(Index index, VolumeIterator &outVBegin, VolumeIterator &outVEnd,
			ObjectIterator &outOBegin, ObjectIterator &outOEnd) const
		{
			_tree->getChildren(index, outVBegin, outVEnd, outOBegin, outOEnd);
		}

		inline const Volume& getVolume(Index index) const
		{
			return _tree->getVolume(index);
		}
#pragma endregion

	public:
		bool operator==(const SubBvh& rhs) const
		{
			return this->_tree == rhs._tree && this->_index == rhs._index;
		}

		bool operator!=(const SubBvh& rhs) const
		{
			return this->_tree != rhs._tree || this->_index != rhs._index;
		}
	};

	template <typename _TBVH, typename _TPredicator>
	struct BvhPredIter : public SubBvh<_TBVH>, public std::iterator<std::forward_iterator_tag, const typename _TBVH::Object>
	{
	public:
		//static_assert(std::is_same<bool, decltype(std::declval<_TPredicator>()(declval<Volume>()))>, "TPred must have operator(const Volume&)->bool");
		//static_assert(std::is_same<bool, decltype(std::declval<_TPredicator>()(declval<Object>()))>, "TPred must have operator(const Object&)->bool");

		typedef BvhPredIter self_type;
		typedef _TPredicator predicate_type;

		typedef SubBvh<_TBVH> SubTreeType;

		//typedef std::forward_iterator_tag iterator_category;
		//typedef const Object value_type;
		//typedef const value_type& reference;
		//typedef const value_type* pointer;
		//typedef ptrdiff_t different_type;

	protected:
		using SubBvh::_index;
		using SubBvh::_tree;

		predicate_type	   _pred; // _pred(const Volume& vol) &&  _pred(const Object& vol) must exist
		std::vector<Index> _todo; // stack for saving context in DFS visit

	public:
		BvhPredIter(const TreeType& tree, const _TPredicator& pred)
			: SubTreeType(tree), _pred(pred)
		{
		}

		BvhPredIter(const TreeType& tree, Index index, const _TPredicator& pred)
			: SubTreeType(tree, index), _pred(pred)
		{
		}

		BvhPredIter(const SubTreeType& tree, const _TPredicator& pred)
			: SubTreeType(tree), _pred(pred)
		{
		}

	public:
		void init()
		{
			if (_pred(getVolume()))
				_todo.push_back(_index);
			else
				_index = -1;
		}

		bool nextVolume()
		{
			VolumeIterator vBegin, vEnd;
			while (!this->_todo.empty())
			{
				this->_index = this->_todo.back();
				this->_todo.pop_back();
				this->_tree->getChildren(this->_index, vBegin, vEnd);

				for (;vBegin != vEnd; ++vBegin)
					if (this->_pred(this->_tree->getVolume(*vBegin)))
						this->_todo.push_back(*vBegin);
			}

			this->_index = -1;
			return true;
		}

		bool nextObject()
		{
			VolumeIterator vBegin, vEnd;
			while (!this->_todo.empty())
			{
				this->_index = this->_todo.back();
				this->_todo.pop_back();
				this->_tree->getChildren(this->_index, vBegin, vEnd);

				for (;vBegin != vEnd; ++vBegin)
					if (this->_pred(this->_tree->getVolume(*vBegin)))
						this->_todo.push_back(*vBegin);

				if (isObject() && this->_pred(getObject()))
					return true;
			}

			this->_index = -1;
			return true;
		}

	public:
		// Iterator interface
		const Object& operator*()
		{
			return getObject();
		}

		self_type& operator++()
		{
			nextObject();
			return *this;
		}

		// Very BAD!!!
		self_type operator++(int)
		{
			self_type itr(*this);
			++itr;
			return itr;
		}
	};

	// Return all intersected objects as a iterator range
	template <typename _TBVH, typename TPred>
	inline iterator_range<BvhPredIter<_TBVH, TPred>>
	BVFindAllIf(const _TBVH& bvh, typename _TBVH::Index index, const TPred &pred)
	{
		typedef BvhPredIter<_TBVH, TPred> ItrType;

		ItrType itr(bvh, index, pred);
		itr.init();

		ItrType enditr(bvh, -1, pred);

		iterator_range<ItrType> range(std::move(itr), std::move(enditr));
		return range;
	}

	template <typename _TBVH, typename TPred>
	inline auto BVFindAllIf(const _TBVH& bvh, const TPred &pred)
	{
		return BVFindAllIf(bvh,bvh.getRootIndex(), pred);
	}

	// Initialize form N objects, object will be sorted in KdTree mannar
	// Index range in [0,N) are objects
	// Index range in [N,2N-1) are intermediate nodes
	// ===== Object Acesss =====
	// Grab object at index i with getObject(i) or at(i) or operator[i]
	// Acess all object with vector interface, e.g. begin(), end(), size()
	// ===== Spatial Query =====
	// Do spatial intersection queries by findObjects(pred)
	template <typename _TScaler, size_t _Dim, typename _TObject, class _TAabb = Eigen::AlignedBox<_TScaler, _Dim>>
	class KdAabbTree
	{
	public:
		typedef _TScaler Scalar;
		static const size_t Dim = _Dim;

		typedef int Index;
		typedef _TAabb		AabbType;
		typedef _TObject	ObjectType;
		typedef AabbType	Volume;
		typedef ObjectType	Object;

		typedef std::vector<AabbType, Eigen::aligned_allocator<AabbType>> VolumeList;
		typedef std::vector<ObjectType, Eigen::aligned_allocator<ObjectType>> ObjectList;

		typedef const Index*	VolumeIterator; //an iterator type over node children--returns Index
		typedef const Object*	ObjectIterator; //an iterator over object (leaf) children--returns const Object &

		typedef SubBvh<KdAabbTree> SubTreeType;

	private:
		std::vector<Index>	m_children; //children of x are children[2x] and children[2x+1], indices bigger than boxes.size() index into objects.
		VolumeList			m_boxes;
		ObjectList			m_objects;
		Index				m_root;

		std::function<Volume(const Object&)>
							m_getBox;
	public:
		KdAabbTree(const std::function<Volume(const Object&)> &getBox)
			: m_getBox(getBox)
		{}
		// Eigen::BVH and extension interfaces
		inline SubTreeType getSubTree(Index index)
		{
			return SubTreeType(*this, index);
		}

		inline bool  hasNode(Index index);

		inline Index getRootIndex() const
		{
			return m_root;
		}

		inline bool isObject(Index index) const
		{
			return index < m_objects.size();
		}

		inline const Object& getObject(Index index) const
		{
			return m_objects[index];
		}

		// Important!!! This methods work not like another getChildren, as all object's bounding box is also returned
		inline bool getChildren(Index index, VolumeIterator &outVBegin, VolumeIterator &outVEnd) const
		{
			if (index < m_objects.size()) // it's a object, a leaf node thus no children
			{
				outVBegin = outVEnd;
			}
			else // it's a intermediate node
			{
				outVBegin = &m_children[getChildIndex(index)];
				outVEnd = outVBegin + 2;
			}
			return outVBegin != outVEnd;
		}

		void getChildren(Index index, VolumeIterator &outVBegin, VolumeIterator &outVEnd,
			ObjectIterator &outOBegin, ObjectIterator &outOEnd) const
		{
			//inlining this function should open lots of optimization opportunities to the compiler
			if (index < 0) {
				outVBegin = outVEnd;
				if (!m_objects.empty())
					outOBegin = &(m_objects[0]);
				outOEnd = outOBegin + m_objects.size(); //output all objects--necessary when the tree has only one object
				return;
			}
			else {
				int numObjs = m_objects.size();
				Index cidx = getChildIndex(index); // first child index

				if (m_children[cidx + 1] > numObjs) // both children are ondes
				{
					outVBegin = &(m_children[cidx]);
					outVEnd = outVBegin + 2;
					outOBegin = outOEnd;
				} else if (m_children[cidx] < numObjs) { //if both children are leaf
					outVBegin = outVEnd;
					assert(m_children[cidx] + 1 == m_children[cidx + 1]);
					outOBegin = &(m_objects[m_children[cidx]]);
					outOEnd = outOBegin + 2;
				}
				else //if the first child is a volume and the second is an object
				{
					outVBegin = &(m_children[cidx]);
					outVEnd = outVBegin + 1;
					outOBegin = &(m_objects[m_children[cidx + 1]]);
					outOEnd = outOBegin + 1;
				}
			}
		}

		inline const Volume& getVolume(Index index) const
		{
			return m_boxes[index];
		}

		//// use this method to update all bounding volumes contains this object after you altered it
		//inline void updateObject(Index index);

	public:
		// Vector like STL container interface for objects
		void assign(ObjectList&& objects);
		void assign(const ObjectList& objects);

		template <typename OItr>
		void assign(OItr obegin, OItr oend)
		{
			m_objects.assign(obegin, oend);
		}

		void resize(size_t n)
		{
			m_objects.resize(n);
			m_boxes.resize(n * 2 - 1);
			m_children.resize((n - 1) * 2);
		}

		ObjectList& getObjectList() { return m_objects; }

		inline Object& operator[](unsigned int index) { return m_objects[index]; }
		inline const Object& operator[](unsigned int index) const { return m_objects[index]; }
		inline Object& at(unsigned int index) { return m_objects.at(index); }
		inline const Object& at(unsigned int index) const { return m_objects.at(index); }
		inline size_t size() const { return m_objects.size(); }
		
		inline bool empty() const { return m_objects.empty(); }

		inline auto begin() { return m_objects.begin(); }
		inline auto end() { return m_objects.end(); }
		inline auto begin() const { return m_objects.begin(); }
		inline auto end() const { return m_objects.end(); }
		inline auto cbegin() const { return m_objects.cbegin(); }
		inline auto cend() const { return m_objects.cend(); }

		inline void push_back(const Object& obj) { m_objects.push_back(obj); }

		inline Object& back() { return m_objects.back(); }

		inline void clear() { 
			m_root = -1;
			m_children.clear();
			m_boxes.clear();
			m_objects.clear();
		}

		// using this method to update the aabb tree structure after you significantly changed the objects
		inline void rebuild()
		{
			init();
		}

	protected:
		typedef internal::box_int_pair<Scalar, Dim> IndexedBox;
		typedef std::vector<IndexedBox, Eigen::aligned_allocator<IndexedBox>> IndexedBoxList;
		struct VectorComparator //compares vectors, or, more specificall, VIPairs along a particular dimension
		{
			VectorComparator(int inDim) : dim(inDim) {}
			inline bool operator()(const IndexedBox &v1, const IndexedBox &v2) const { return v1.center[dim] < v2.center[dim]; }
			int dim;
		};

		inline Index getChildIndex(Index index) const
		{
			return ((Index)m_objects.size() * 2 - 2 - index) * 2;
		}

		// pre-condition : m_objects is initialized
		void init()
		{
			// allocate storage
			Index n = static_cast<Index>(m_objects.size());
			m_boxes.resize(n * 2 - 1);
			m_children.resize((n-1) * 2);

			// compute the indexed_boxes for partition
			IndexedBoxList idxBoxes(m_objects.size());
			for (Index i = 0; i < n; i++)
			{
				auto box = m_getBox(m_objects[i]);
				idxBoxes[i].index = i;
				idxBoxes[i].center = box.center();
				idxBoxes[i].extent = box.max() - idxBoxes[i].center;
			}

			// build the aabb tree
			m_root = build(idxBoxes, 0, n, 0);

			// sync object's index with it's index in m_boxes
			ObjectList temp(n);
			temp.swap(m_objects);
			for (Index i = 0; i < n; i++)
				m_objects[i] = temp[idxBoxes[i].index];
		}

		Index createNode(Index left, Index right)
		{
			m_boxes.push_back(m_boxes[left].merged(m_boxes[right]));

			Index idx = (Index)m_boxes.size() - 1;
			// get children array index for this node
			idx = getChildIndex(idx);
			m_children[idx] = left; //there are objects.size() - 1 tree nodes
			m_children[idx + 1] = right;

			idx = (Index)m_boxes.size() - 1;
			return idx;
		}

		Index build(IndexedBoxList &idxBoxes, Index from, Index to, int dim)
		{
			eigen_assert(to - from > 1);
			if (to - from == 2) {

				m_boxes[from] = idxBoxes[from].getBox();
				m_boxes[from + 1] = idxBoxes[from + 1].getBox();

				return createNode(from, from + 1);
			}
			else if (to - from == 3) {
				Index mid = from + 2;
				std::nth_element(idxBoxes.begin() + from, idxBoxes.begin() + mid,
					idxBoxes.begin() + to, VectorComparator(dim)); //partition
				
				Index idx1 = build(idxBoxes, from, mid, (dim + 1) % Dim);

				m_boxes[mid] = idxBoxes[mid].getBox();

				return createNode(idx1, mid);
			}
			else {
				Index mid = from + (to - from) / 2;
				std::nth_element(idxBoxes.begin() + from, idxBoxes.begin() + mid,
					idxBoxes.begin() + to, VectorComparator(dim)); //partition

				Index idx1 = build(idxBoxes, from, mid, (dim + 1) % Dim);
				Index idx2 = build(idxBoxes, mid, to, (dim + 1) % Dim);
				return createNode(idx1, idx2);
			}
		}
	};
	//typedef KdAabbTree<float,3,>
}

namespace std
{
	//template <typename _TScaler, size_t _Dim, typename _TObject, typename _TAabb, typename _TPred>
	//class iterator_traits<typename Geometrics::KdAabbTree<_TScaler, _Dim, _TObject, _TAabb>::BvhPredIter<_TPred>>
	//{
	//public:
	//	typedef std::forward_iterator_tag iterator_category;
	//	typedef const _TObject value_type;
	//	typedef const value_type& reference;
	//	typedef const value_type* pointer;
	//	typedef ptrdiff_t different_type;
	//};
}