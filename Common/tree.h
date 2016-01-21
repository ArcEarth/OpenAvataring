// Copyright (c) by Yupeng Zhang. All Rights Reserved.  -*- C++ -*-
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#ifndef STDX_TREE_H
#define STDX_TREE_H
#include <utility>
#include <vector>
#include <queue>
#include <stack>
#include <cassert>
#include "iterator_range.h"

namespace stdx
{
	using std::iterator_range;

	struct tree_node_relation_descants_ownership {};
	struct tree_node_relation_no_decants_ownership {};
	// Use and only it as the base of your tree node like:
	// class Ty : public foward_tree_node<Ty>

	// Left-Child, Right-Sibling Method Storaged tree node base (3 pointer overhaul for each node), efficent in memery usage, but bad in "upward traveling" or "reverse order tranveling"
	template<typename _Ty, bool _DescendabtsOwnership = true>
	class foward_tree_node
	{
	public:
		//typedef foward_tree_node<_Ty, _DescendabtsOwnership> self_type;
		//static_assert(std::is_base_of<self_type, _Ty>::value, "_Ty should be a derived type of tree_node");

		typedef _Ty value_type;
		typedef value_type& reference;
		typedef value_type* pointer;
		typedef value_type const & const_reference;
		typedef value_type const * const_pointer;

	public:
		//_Ty Entity;
	protected:
		// Next sibling node
		pointer _sibling;
		// First child node
		pointer _child;
		// Physical parent node of this node in data structure
		// May be Logical Parent or Prev Sibling
		pointer _parent;
		//		foward_tree_node *_next;

	private:
		template <typename T>
		inline static void internal_delete(std::enable_if_t<_DescendabtsOwnership, T>* &pData) {
			if (pData) {
				delete pData;
				pData = nullptr;
			}
		}
		template <typename T>
		inline static void internal_delete(std::enable_if_t<!_DescendabtsOwnership, T>* &pData) {
			pData = nullptr;
		}

		// Basic Properties
	public:

		foward_tree_node()
			: _parent(nullptr), _sibling(nullptr), _child(nullptr)
		{
		}

		~foward_tree_node()
		{
			internal_delete<_Ty>(_sibling);
			internal_delete<_Ty>(_child);
#ifdef _DEBUG
			_parent = nullptr;
#endif
		}

		foward_tree_node& operator = (foward_tree_node&& rhs)
		{
			// fixup parent/child pointers
			if (rhs->_parent)
				if (rhs->_parent->_child == &rhs)
					rhs->_parent->_child = this;
				else
					rhs->_parent->_sibling = this;
			if (rhs->_child)
				rhs->_child->_parent = this;
			if (rhs->_sibling)
				rhs->_sibling->_parent = this;

			this->_parent = rhs._parent;
			this->_sibling = rhs._sibling;
			this->_child = rhs._child;
		}

		// Logical Parent for this node
		const_pointer parent() const {
			const_pointer p = static_cast<const_pointer>(this);
			while (p->_parent && p->_parent->_child != p)
				p = p->_parent;
			return p->_parent;
		}
		// Logical Parent for this node
		pointer parent() {
			pointer = static_cast<pointer>(this);;
			while (p->_parent && p->_parent->_child != p)
				p = p->_parent;
			return p->_parent;
		}

		pointer next_sibling()
		{
			return _sibling;
		}
		const_pointer next_sibling() const
		{
			return _sibling;
		}
		pointer prev_sibling()
		{
			if (this->_parent->_sibling == this)
				return _parent;
			else
				return nullptr;
		}
		const_pointer prev_sibling() const
		{
			if (this->_parent->_sibling == this)
				return _parent;
			else
				return nullptr;
		}

		// check if the node is a LOGICAL root node of a tree
		bool has_child() const { return _child; }
		bool is_leaf() const { return !_child; }
		// if this node is Logical Root
		bool is_root() const {
			return (!this->parent());
		}
		bool is_tree_root() const
		{
			return !_parent && !_sibling;
		}
		bool is_forest_root() const
		{
			return !_parent && _sibling;
		}

		// Iterators and Ranges
	public:

		struct const_iterator_base
		{
		public:
			typedef const _Ty value_type;
			typedef value_type const & reference;
			typedef value_type const * pointer;
			typedef std::forward_iterator_tag iterator_category;
			typedef int difference_type;
		protected:
			const_iterator_base() {}
			const_iterator_base(const pointer ptr) : current(ptr) {}
			pointer current;
		public:
			pointer get() const {
				return current;
			}

			bool is_valid() const
			{
				return current;
			}

			template <class _TItr>
			bool equal(const _TItr & rhs) const {
				return current == rhs.current;
			}
		};

		struct mutable_iterator_base
		{
		public:
			typedef _Ty value_type;
			typedef value_type& reference;
			typedef value_type* pointer;
			typedef int difference_type;
		protected:
			pointer current;
			mutable_iterator_base() {}
			mutable_iterator_base(const pointer ptr) : current(ptr) {}
		public:
			pointer get() const {
				return current;
			}

			bool is_valid() const
			{
				return current;
			}

			template <class _TItr>
			bool equal(const _TItr & rhs) const {
				return current == rhs.current;
			}
		};

		// _TBase must be basic_iterator or const_basic_iterator
		template <typename _TBase>
		class depth_first_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef depth_first_iterator self_type;
			typedef std::forward_iterator_tag iterator_category;
		public:
			depth_first_iterator(void) {}

			explicit depth_first_iterator(const pointer ptr)
				: base_type(ptr) {}

			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}

			self_type& operator ++() {
				move_to_next();
				return *this;
			}

			void move_to_next()
			{
				if (!current) return;
				if (current->_child)
				{
					current = current->_child;
				}
				else if (current->_sibling) current = current->_sibling;
				else
				{
					// move_to_next_from_this_subtree
					while ((current->_parent) && (!current->_parent->_sibling || current->_parent->_sibling == current))
						current = current->_parent;
					if (current->_parent)
						current = current->_parent->_sibling;
					else
						current = nullptr;
				}
			}

			inline void move_to_next_from_this_subtree()
			{
				if (current->_sibling)
					current = current->_sibling;
				else {
					while ((current->_parent) && (!current->_parent->_sibling || current->_parent->_sibling == current))
						current = current->_parent;
					if (current->_parent)
						current = current->_parent->_sibling;
					else
						current = nullptr;
				}
			}

			reference operator * () const {
				return *current;
			}
			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};
		template <typename _TBase>
		class leaf_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef leaf_iterator self_type;
			typedef std::forward_iterator_tag iterator_category;
		public:
			leaf_iterator(void) {}

			//explicit leaf_iterator(const pointer ptr)
			//	: base_type(ptr) {
			//	while (current && current->_child)
			//		current = current->_child;
			//}

			static self_type create_end(const pointer ptr)
			{
				leaf_iterator itr;
				itr.current = ptr;
				if (ptr == nullptr) return itr;
				itr.move_to_next_from_this_subtree();
				return itr;
			}

			static self_type create_begin(const pointer ptr)
			{
				leaf_iterator itr;
				itr.current = ptr;
				while (itr.current && itr.current->_child)
					itr.current = itr.current->_child;
				return itr;
			}

			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}

			self_type& operator ++() {
				move_to_next();
				return *this;
			}

			void move_to_next()
			{
				if (!current) return;

				if (current->_child)
				{
					current = current->_child;
					while (current->_child)
						current = current->_child;
				}
				else
					move_to_next_from_this_subtree();
			}

			// when all the desendants of this node is visited
			inline void move_to_next_from_this_subtree()
			{
				if (current->_sibling)
				{
					current = current->_sibling;
					// move to next
					//if (current->_child)
					//	move_to_next();
					while (current->_child)
						current = current->_child;
				}
				else
				{
					while ((current->_parent) && (!current->_parent->_sibling || current->_parent->_sibling == current))
						current = current->_parent;
					if (current->_parent) //  && current->_parent->_sibling && current->_parent->_sibling != current
					{
						current = current->_parent->_sibling;
						// move to next
						//if (current && current->_child)
						//	move_to_next();
						while (current->_child)
							current = current->_child;

					}
					else
						current = nullptr;
				}
			}

			reference operator * () const {
				return *current;
			}
			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};

		template <typename _TBase>
		class breadth_first_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef std::forward_iterator_tag iterator_category;
			typedef breadth_first_iterator self_type;
		private:
			std::queue<pointer> node_queue;
			bool				ignore_sibling;
		public:
			breadth_first_iterator(void)
				: base_type(nullptr)
			{}

			// When ignore_root_sibling is set to True, the BFS-travel will ingore the siblings of current
			explicit breadth_first_iterator(const pointer ptr, bool ignore_root_sibling = false)
				: base_type(ptr), ignore_sibling(ignore_root_sibling)
			{
				node_queue.push(nullptr);
			}

			// Copy and bfs-iterator is expensive!
			breadth_first_iterator(const self_type& rhs)
				: base_type(ptr), node_queue(rhs)
			{}

			breadth_first_iterator(self_type&& rhs)
				: base_type(ptr), node_queue(std::move(rhs))
			{}

			// Copy and bfs-iterator is expensive!
			self_type& operator=(const self_type& rhs)
			{
				node_queue = rhs;
				current = rhs.current;
			}
			self_type& operator=(self_type&& rhs)
			{
				node_queue = std::move(rhs);
				current = rhs.current;
			}

			// Copy and bfs-iterator is expensive!
			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}
			self_type& operator ++() {
				move_to_next();
				return *this;
			}

			void move_to_next()
			{
				if (!current) return;
				if (ignore_sibling)
				{
					current = current->_child;
					ignore_sibling = false;
				}
				if (current->_sibling)
				{
					node_queue.push(current->_child);
					current = current->_sibling;
				}
				else if (!node_queue.empty())
				{
					current = node_queue.front();
					node_queue.pop();
				}
				current = nullptr;
			}

			reference operator * () const {
				return *current;
			}
			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};

		// Bidirectional-Sibling-Iterator 
		template <typename _TBase>
		class sibling_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef sibling_iterator self_type;
			typedef std::bidirectional_iterator_tag iterator_category;
		public:
			sibling_iterator(void)
				: base_type(nullptr)
			{}

			explicit sibling_iterator(const pointer ptr)
				: base_type(ptr)
			{}

			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}

			self_type& operator ++() {
				if (!current) return *this;
				current = current->_sibling;
				return *this;
			}
			self_type operator --(int) {
				self_type other(current);
				--(*this);
				return other;
			}

			self_type& operator --() {
				if (!current) return;
				current = current->prev_sibling();
				return *this;
			}

			reference operator * () const {
				return *current;
			}
			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};

		typedef depth_first_iterator<mutable_iterator_base>		mutable_depth_first_iterator;
		typedef breadth_first_iterator<mutable_iterator_base>	mutable_breadth_first_iterator;
		typedef sibling_iterator<mutable_iterator_base>			mutable_sibling_iterator;
		typedef leaf_iterator<mutable_iterator_base>			mutable_leaf_iterator;
		typedef depth_first_iterator<const_iterator_base>		const_depth_first_iterator;
		typedef breadth_first_iterator<const_iterator_base>		const_breadth_first_iterator;
		typedef sibling_iterator<const_iterator_base>			const_sibling_iterator;
		typedef leaf_iterator<const_iterator_base>				const_leaf_iterator;

		typedef const_depth_first_iterator const_iterator;
		typedef mutable_depth_first_iterator iterator;

		// Immutable ranges

		// iterator throught this and all it's "next" siblings 
		const_sibling_iterator siblings_begin() const {
			return const_sibling_iterator(static_cast<const_pointer>(this));
		}

		const_sibling_iterator siblings_end() const {
			return const_sibling_iterator(nullptr);
		}

		// Iterator though all it's direct children
		const_sibling_iterator children_begin() const {
			return const_sibling_iterator(static_cast<const_pointer>(this)->_child);
		}

		const_sibling_iterator children_end() const {
			return const_sibling_iterator(nullptr);
		}
		const_leaf_iterator leaves_begin() const
		{
			return const_leaf_iterator::create_begin(static_cast<const_pointer>(this));
		}
		const_leaf_iterator leaves_end() const
		{
			return const_leaf_iterator::create_end(static_cast<const_pointer>(this));
		}

		const_depth_first_iterator descendants_begin() const {
			return const_depth_first_iterator(static_cast<const_pointer>(this)->_child);
		}

		const_depth_first_iterator descendants_end() const {
			return const_depth_first_iterator(static_cast<const_pointer>(this));
		}
		// breadth_first_iterator can self determine if it has meet the end
		const_breadth_first_iterator descendants_breadth_first_begin() const {
			return const_breadth_first_iterator(static_cast<const_pointer>(this)->_child);
		}
		// just an null-iterator
		const_sibling_iterator descendants_breath_first_end() const {
			return const_sibling_iterator(nullptr);
		}
		// Depth first begin iterator to all nodes inside this sub-tree
		const_depth_first_iterator begin() const {
			return const_depth_first_iterator(static_cast<const_pointer>(this));
		}
		// Depth first end iterator to all nodes inside this sub-tree
		const_depth_first_iterator end() const {
			auto itr = const_depth_first_iterator(static_cast<const_pointer>(this));
			itr.move_to_next_from_this_subtree();
			return itr;
		}
		// breadth_first_iterator can self determine if it has meet the end
		const_breadth_first_iterator breadth_first_begin() const {
			return const_breadth_first_iterator(static_cast<const_pointer>(this), true);
		}
		// just an null-iterator
		const_sibling_iterator breath_first_end() const {
			return const_sibling_iterator(nullptr);
		}

		// Mutable ranges

		// iterator throught this and all it's "next" siblings 
		mutable_sibling_iterator siblings_begin() {
			return mutable_sibling_iterator(static_cast<pointer>(this));
		}

		mutable_sibling_iterator siblings_end() {
			return mutable_sibling_iterator(nullptr);
		}
		// List-like iterator over children
		mutable_sibling_iterator children_begin() {
			return mutable_sibling_iterator(static_cast<pointer>(this)->_child);
		}
		// List-like iterator over children
		mutable_sibling_iterator children_end() {
			return mutable_sibling_iterator(nullptr);
		}
		mutable_leaf_iterator leaves_begin()
		{
			return mutable_leaf_iterator::create_begin(static_cast<pointer>(this));
		}
		mutable_leaf_iterator leaves_end()
		{
			return mutable_leaf_iterator::create_end(static_cast<pointer>(this));
		}
		// Depth first descendants begin iterator
		mutable_depth_first_iterator descendants_begin() {
			return mutable_depth_first_iterator(static_cast<pointer>(this)->_child);
		}
		// Depth first descendants end iterator
		mutable_depth_first_iterator descendants_end() {
			return mutable_depth_first_iterator(static_cast<pointer>(this));
		}
		// Breadth first descendants iterator can self determine if it has meet the end
		mutable_breadth_first_iterator descendants_breadth_first_begin() {
			return mutable_breadth_first_iterator(static_cast<pointer>(this)->_child);
		}
		// just an null-iterator
		mutable_sibling_iterator descendants_breath_first_end() {
			return mutable_sibling_iterator(nullptr);
		}
		// begin iterator to all nodes inside this sub-tree
		mutable_depth_first_iterator begin() {
			return mutable_depth_first_iterator(static_cast<pointer>(this));
		}
		// end iterator to all nodes inside this sub-tree
		mutable_depth_first_iterator end() {
			auto itr = mutable_depth_first_iterator(static_cast<pointer>(this));
			itr.move_to_next_from_this_subtree();
			return itr;
		}
		// breadth_first_iterator can self determine if it has meet the end
		mutable_breadth_first_iterator breadth_first_begin() {
			return const_breadth_first_iterator(static_cast<pointer>(this), true);
		}
		// just an null-iterator
		mutable_sibling_iterator breath_first_end() {
			return mutable_sibling_iterator(nullptr);
		}

		iterator_range<const_sibling_iterator>
			children() const
		{
			return iterator_range<const_sibling_iterator>(children_begin(), children_end());
		}
		iterator_range<const_depth_first_iterator>
			nodes_in_tree() const
		{
			return iterator_range<const_depth_first_iterator>(begin(), end());
		}
		iterator_range<const_leaf_iterator>
			leaves() const
		{
			return iterator_range<const_leaf_iterator>(leaves_begin(), leaves_end());
		}
		iterator_range<const_depth_first_iterator>
			descendants() const
		{
			return iterator_range<const_depth_first_iterator>(descendants_begin(), descendants_end());
		}
		iterator_range<mutable_sibling_iterator>
			children()
		{
			return iterator_range<mutable_sibling_iterator>(children_begin(), children_end());
		}
		iterator_range<mutable_depth_first_iterator>
			nodes_in_tree()
		{
			return iterator_range<mutable_depth_first_iterator>(begin(), end());
		}
		iterator_range<mutable_leaf_iterator>
			leaves()
		{
			return iterator_range<mutable_leaf_iterator>(leaves_begin(), leaves_end());
		}
		iterator_range<mutable_depth_first_iterator>
			descendants()
		{
			return iterator_range<mutable_depth_first_iterator>(descendants_begin(), descendants_end());
		}




		// use this method to extract this node (and all of its sub node) as a sub-tree 
		// and remove it from the oringinal parent
		// this method ensure the rest of tree structure is not affected
		void isolate() {
			if (!this->_parent)
#ifdef _DEBUG
				throw new std::exception("Can not separate the root tree_node.");
#else
				return;
#endif

			if (this->_parent->_child == this) {
				this->_parent->_child = this->_sibling;
			}
			else
			{
				this->_parent->_sibling = this->_sibling;
			}
			if (this->_sibling)
				this->_sibling->_parent = this->_parent;
			this->_parent = nullptr;
			this->_sibling = nullptr;
		}

		inline void append_children_front(pointer node)
		{
			assert(node && !node->_parent);
			auto rptr = node;
			while (rptr->_sibling)
				rptr = rptr->_sibling;
			rptr->_sibling = this->_child;
			if (this->_child)
				this->_child->_parent = rptr;
			this->_child = node;
			node->_parent = static_cast<pointer>(this);
		}

		inline void append_children_back(pointer node)
		{
			assert(node && !node->_parent);
			auto rptr = this->_child;
			if (rptr == nullptr) {
				this->_child = node;
				node->_parent = static_cast<pointer>(this);
			}
			else {
				while (rptr->_sibling != nullptr)
					rptr = rptr->_sibling;
				rptr->_sibling = node;
				node->_parent = rptr;
			}
		}

		inline void insert_as_siblings_after(pointer node)
		{
			assert(node && !node->_parent);
			node->_parent = this;
			if (_sibling)
			{
				auto rptr = node;
				while (rptr->_sibling)
					rptr = rptr->_sibling;
				rptr->_sibling = this->_sibling;
			}
			this->_sibling = node;
		}

		inline void insert_as_siblings_before(pointer node)
		{
			assert(node && !node->_parent);
			auto rptr = node;
			while (rptr->_sibling)
				rptr = rptr->_sibling;

			if (!this->_parent)
			{
				rptr->_sibling = this;
				this->_parent = node;
			}
			else {
				node->_parent = this->_parent;
				rptr->_sibling = this;
				if (this->_parent->_sibling == this)
					this->_parent->sibling = pTree;
				else // this->parent->child == this
					this->_parent->_child = pTree;
				this->_parent = rptr;
			}
		}
	};

	// Each node contains Left-Sibling, Right-Sibling, First-Child, Last-Child, Parent
	// 5 pointer overhaul for each node
	template<typename _Ty, bool _DescendabtsOwnership = true>
	class tree_node
	{
	public:
		//typedef tree_node<_Ty, _DescendabtsOwnership> self_type;
		//static_assert(std::is_base_of<self_type, _Ty>::value, "_Ty should be a derived type of tree_node");

		typedef _Ty value_type;
		typedef value_type& reference;
		typedef value_type* pointer;
		typedef value_type const & const_reference;
		typedef value_type const * const_pointer;

	protected:
		pointer _prev_sibling;
		pointer _next_sibling;
		// First child node
		pointer _first_child;
		pointer _last_child;
		// Physical parent node of this node in data structure
		// May be Logical Parent or Prev Sibling
		pointer _parent;
		//Properties
	private:
		template <typename T>
		inline static void internal_delete(std::enable_if_t<_DescendabtsOwnership, T>* &pData) {
			if (pData) {
				delete pData;
//#ifdef _DEBUG
				pData = nullptr;
//#endif
			}
		}

		template <typename T>
		inline static void internal_delete(std::enable_if_t<!_DescendabtsOwnership, T>* &pData) {
//#ifdef _DEBUG
			pData = nullptr;
//#endif
		}

	public:
		//static pointer null = nullptr;// = new tree_node();

		tree_node()
			: _prev_sibling(nullptr), _next_sibling(nullptr), _first_child(nullptr), _last_child(nullptr), _parent(nullptr)
		{
		}

		~tree_node()
		{
			auto child = _first_child;
			while (child != nullptr)
			{
				auto* pNext = child->_next_sibling;
				internal_delete<_Ty>(child);
				child = pNext;
			}
#ifdef _DEBUG
			_parent = nullptr;
#endif
		}

		// None-copyable, this should only use as pointer or reference
		tree_node operator= (const tree_node&) = delete;

		//tree_node& operator = (tree_node&& rhs)
		//{
		//	this->_parent = rhs._parent;
		//	this->_prev_sibling = rhs._prev_sibling;
		//	this->_next_sibling = rhs._next_sibling;
		//	this->_last_child = rhs._last_child;
		//	this->_first_child = rhs._first_child;
		//}

		bool is_null() const
		{
			return this == nullptr;
		}

		bool operator==(nullptr_t) const
		{
			return is_null();
		}

		// Logical Parent for this node
		const_pointer parent() const {
			return _parent;
		}
		// Logical Parent for this node
		pointer parent() {
			return _parent;
		}
		pointer next_sibling()
		{
			return _next_sibling;
		}
		const_pointer next_sibling() const
		{
			return _next_sibling;
		}
		pointer prev_sibling()
		{
			return _prev_sibling;
		}
		const_pointer prev_sibling() const
		{
			return _prev_sibling;
		}
		pointer first_child()
		{
			return _first_child;
		}
		const_pointer first_child() const
		{
			return _first_child;
		}
		pointer last_child()
		{
			return _last_child;
		}
		const_pointer last_child() const
		{
			return _last_child;
		}

		// check if the node is a LOGICAL root node of a tree
		bool has_child() const { return _first_child; }
		bool is_leaf() const { return !_first_child; }
		// if this node is Logical Root
		bool is_root() const {
			return (!this->_parent);
		}

		// get the ultimate root of this tree-node
		// Time is O(h) , where h is this node's height
		pointer root()
		{
			pointer node = static_cast<pointer>(this);
			while (node->_parent)
				node = node->_parent;
			return node;
		}

		// get the ultimate root of this tree-node
		// Time is O(h) , where h is this node's height
		const_pointer root() const
		{
			return const_cast<pointer>(static_cast<const_pointer>(this))->root();
		}

		// Modifiers
	public:
		// use this method to extract this node (and all of its sub node) as a sub-tree 
		// and remove it from the oringinal parent
		// this method ensure the rest of tree structure is not affected
		void isolate() {
			if (!this->_parent)
#ifdef _DEBUG
				throw new std::exception("Can not separate the root tree_node.");
#else
				return;
#endif

			if (this->_parent->_first_child == this)
				this->_parent->_first_child = this->_next_sibling;
			if (this->_parent->_last_child == this)
				this->_parent->_last_child = this->_prev_sibling;

			if (this->_prev_sibling)
				this->_prev_sibling->_next_sibling = this->_next_sibling;
			if (this->_next_sibling)
				this->_next_sibling->_prev_sibling = this->_prev_sibling;

			this->_parent = nullptr;
			this->_next_sibling = nullptr;
			this->_prev_sibling = nullptr;
		}

		inline void append_children_front(pointer node)
		{
			assert(node && !node->_parent && !node->_next_sibling);
			if (this->_first_child)
			{
				this->_first_child->_prev_sibling = node;
				node->_next_sibling = this->_first_child;
			}
			else
				this->_last_child = node;
			this->_first_child = node;

			do
			{
				node->_parent = static_cast<pointer>(this);
				node = node->_prev_sibling;
			} while (node);


		}

		inline void append_children_back(pointer node)
		{
			assert(node && !node->_parent && !node->_prev_sibling);
			if (this->_last_child)
			{
				this->_last_child->_next_sibling = node;
				node->_prev_sibling = this->_last_child;
			}
			else
				this->_first_child = node;
			this->_last_child = node;

			do
			{
				node->_parent = static_cast<pointer>(this);
				node = node->_next_sibling;
			} while (node);

	
		}

		void insert_sibling_after(pointer node)
		{
			assert(node && !node->_parent && !node->_prev_sibling);

			auto rptr = node; // right most sibling
			while (rptr->_next_sibling)
			{
				rptr->_parent = static_cast<pointer>(this);
				rptr = rptr->_next_sibling;
			}

			if (this->_next_sibling)
			{
				rptr->_next_sibling = this->_next_sibling;
				this->_next_sibling->_prev_sibling = rptr;
			}
			else
			{
				if (this->_parent)
					this->_parent->_last_child = rptr;
			}
			this->_next_sibling = node;
			node->_prev_sibling = static_cast<pointer>(this);
		}

		void insert_sibling_before(pointer node)
		{
			assert(node && !node->_parent && !node->_next_sibling);

			auto lptr = node; // left most sibling
			while (lptr->_prev_sibling)
			{
				lptr->_parent = static_cast<pointer>(this);
				lptr = lptr->_prev_sibling;
			}

			if (this->_prev_sibling)
			{
				lptr->_prev_sibling = this->_prev_sibling;
				this->_prev_sibling->_next_sibling = lptr;
			}
			else
			{
				if (this->_parent)
					this->_parent->_last_child = rptr;
			}
			this->_prev_sibling = node;
			node->_next_sibling = static_cast<pointer>(this);
		}

		pointer remove_child(pointer child)
		{
			child->isolate();
			return child;
		}

		// Iterators
	public:
		struct const_iterator_base
		{
		public:
			typedef const _Ty value_type;
			typedef value_type const & reference;
			typedef value_type const * pointer;
			typedef std::forward_iterator_tag iterator_category;
			typedef int difference_type;
		protected:
			const_iterator_base() {}
			const_iterator_base(const pointer ptr) : current(ptr) {}
			pointer current;
		public:
			pointer get() const {
				return current;
			}

			bool is_valid() const
			{
				return current;
			}

			template <class _TItr>
			bool equal(const _TItr & rhs) const {
				return current == rhs.current;
			}
		};

		struct mutable_iterator_base
		{
		public:
			typedef _Ty value_type;
			typedef value_type& reference;
			typedef value_type* pointer;
			typedef int difference_type;
		protected:
			pointer current;
			mutable_iterator_base() {}
			mutable_iterator_base(const pointer ptr) : current(ptr) {}
		public:
			pointer get() const {
				return current;
			}

			bool is_valid() const
			{
				return current;
			}

			template <class _TItr>
			bool equal(const _TItr & rhs) const {
				return current == rhs.current;
			}
		};

		template <typename _TBase>
		class depth_first_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef depth_first_iterator self_type;
			typedef std::bidirectional_iterator_tag iterator_category;
		public:
			depth_first_iterator(void) {}

			explicit depth_first_iterator(const pointer ptr)
				: base_type(ptr) {}

			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}

			self_type& operator ++() {
				move_to_next();
				return *this;
			}

			self_type operator --(int) {
				self_type other(current);
				--(*this);
				return other;
			}

			self_type& operator --() {
				move_to_prev();
				return *this;
			}

			void move_to_next()
			{
				if (!current) return;
				if (current->_first_child)
					current = current->_first_child;
				else if (current->_next_sibling)
					current = current->_next_sibling;
				else // move_to_next_from_this_subtree
				{
					while (current->_parent && !current->_parent->_next_sibling)
						current = current->_parent;
					if (current->_parent)
						current = current->_parent->_next_sibling;
					else
						current = nullptr;
				}
			}

			void move_to_prev()
			{
				if (!current) return;
				if (current->_prev_sibling)
				{
					current = current->_prev_sibling;
					while (current->_last_child)
						current = current->_last_child;
				}
				else if (current->_parent) // don't have prev-sibling, previous dfs-travel node must be it's parent
					current = current->_parent;
			}

		protected:
			inline void move_to_next_from_this_subtree()
			{
				if (current->_next_sibling)
					current = current->_next_sibling;
				else // move_to_next_from_this_subtree
				{
					while (current->_parent && !current->_parent->_next_sibling)
						current = current->_parent;
					if (current->_parent)
						current = current->_parent->_next_sibling;
					else
						current = nullptr;
				}
			}
		public:
			static inline self_type create_begin(const pointer ptr)
			{
				return self_type(ptr);
			}
			static inline self_type create_end(const pointer ptr)
			{
				self_type itr(ptr);
				if (ptr == nullptr) return itr;
				itr.move_to_next_from_this_subtree();
				return itr;
			}
			static inline self_type create_rbegin(const pointer ptr)
			{
				self_type itr(ptr);
				itr.move_to_prev();
				return itr;
			}
			static inline self_type create_rend(const pointer ptr)
			{
				self_type itr(ptr);
				while (itr.current->_last_child)
					itr.current = itr.current->_last_child;
				return itr;
			}

		public:

			reference operator * () const {
				return *current;
			}
			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};

		template <typename _TBase>
		class breadth_first_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef std::forward_iterator_tag iterator_category;
			typedef breadth_first_iterator self_type;
		private:
			std::queue<pointer> node_queue;
			pointer				root; // the logical root of the subtree to travel for this iterator
		public:
			breadth_first_iterator(void)
				: base_type(nullptr)
			{}

			// When ignore_root_sibling is set to True, the BFS-travel will ingore the siblings of current
			explicit breadth_first_iterator(const pointer ptr, bool ignore_root_sibling = false)
				: base_type(ptr), root(ptr)
			{
				node_queue.push(nullptr);
			}

			// Copy and bfs-iterator is expensive!
			breadth_first_iterator(const self_type& rhs)
				: base_type(ptr), node_queue(rhs), root(rhs.root)
			{}

			breadth_first_iterator(self_type&& rhs)
				: base_type(ptr), node_queue(std::move(rhs)), root(rhs.root)
			{}

			// Copy and bfs-iterator is expensive!
			self_type& operator=(const self_type& rhs)
			{
				node_queue = rhs;
				current = rhs.current;
			}

			self_type& operator=(self_type&& rhs)
			{
				node_queue = std::move(rhs);
				current = rhs.current;
			}

			// Copy and bfs-iterator is expensive!
			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}
			self_type& operator ++() {
				move_to_next();
				return *this;
			}

			void move_to_next()
			{
				if (!current) return;
				if (current == root)
				{
					current = current->_first_child;
				}
				else
				{
					if (current->_next_sibling)
					{
						if (current->_first_child)
							node_queue.push(current->_first_child);
						current = current->_next_sibling;
					}
					else if (!node_queue.empty())
					{
						current = node_queue.front();
						node_queue.pop();
					}
					else
						current = nullptr; // ended!
				}
			}

			void reset_to_begin()
			{
				current = root;
				node_queue.clear();
			}

			reference operator * () const {
				return *current;
			}
			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};

		// Bidirectional-Sibling-Iterator 
		template <typename _TBase>
		class sibling_iterator : public _TBase
		{
		public:
			typedef _TBase base_type;
			typedef sibling_iterator self_type;
			typedef std::bidirectional_iterator_tag iterator_category;
		public:
			sibling_iterator(void)
				: base_type(nullptr)
			{}

			explicit sibling_iterator(const pointer ptr)
				: base_type(ptr)
			{}

			self_type operator ++(int) {
				self_type other(current);
				++(*this);
				return other;
			}

			self_type& operator ++() {
				if (!current) return *this;
				current = current->_next_sibling;
				return *this;
			}
			self_type operator --(int) {
				self_type other(current);
				--(*this);
				return other;
			}

			self_type& operator --() {
				if (!current) return *this;
				current = current->_prev_sibling;
				return *this;
			}

			reference operator * () const {
				return *current;
			}

			pointer operator -> () const {
				return current;
			}

			template <class _TItr>
			bool operator == (const _TItr & rhs) const {
				return current == rhs.get();
			}

			template <class _TItr>
			bool operator != (const _TItr & rhs) const {
				return current != rhs.get();
			}
		};

		typedef depth_first_iterator<mutable_iterator_base>		mutable_depth_first_iterator;
		typedef breadth_first_iterator<mutable_iterator_base>	mutable_breadth_first_iterator;
		typedef sibling_iterator<mutable_iterator_base>			mutable_sibling_iterator;
		typedef depth_first_iterator<const_iterator_base>		const_depth_first_iterator;
		typedef breadth_first_iterator<const_iterator_base>		const_breadth_first_iterator;
		typedef sibling_iterator<const_iterator_base>			const_sibling_iterator;

		typedef const_depth_first_iterator const_iterator;
		typedef mutable_depth_first_iterator iterator;

		// Interators 
	private:
		// iterator throught this and all it's "next" siblings 
		const_sibling_iterator next_siblings_begin() const {
			return const_sibling_iterator(static_cast<const_pointer>(this));
		}

		const_sibling_iterator next_siblings_end() const {
			return const_sibling_iterator(nullptr);
		}
		// iterator throught this and all it's "next" siblings 
		const_sibling_iterator prev_siblings_begin() const {
			return const_sibling_iterator(static_cast<const_pointer>(this));
		}

		const_sibling_iterator prev_siblings_end() const {
			return const_sibling_iterator(nullptr);
		}
		// Iterator though all it's direct children
		const_sibling_iterator children_begin() const {
			return const_sibling_iterator(static_cast<const_pointer>(this)->_first_child);
		}

		const_sibling_iterator children_end() const {
			return const_sibling_iterator(nullptr);
		}
		// Iterator all descendants nodes in this sub-tree
		const_depth_first_iterator descendants_begin() const {
			return const_depth_first_iterator(static_cast<const_pointer>(this)->_first_child);
		}

		const_depth_first_iterator descendants_end() const {
			return const_depth_first_iterator::create_end(static_cast<const_pointer>(this));
		}
		// breadth_first_iterator can self determine if it has meet the end
		const_breadth_first_iterator descendants_breadth_first_begin() const {
			return const_breadth_first_iterator(static_cast<const_pointer>(this)->_first_child);
		}
		// just an null-iterator
		const_sibling_iterator descendants_breadth_first_end() const {
			return const_sibling_iterator(nullptr);
		}
		// Depth first begin iterator to all nodes inside this sub-tree
		const_depth_first_iterator nodes_begin() const {
			return const_depth_first_iterator(static_cast<const_pointer>(this));
		}
		// Depth first end iterator to all nodes inside this sub-tree
		const_depth_first_iterator nodes_end() const {
			return const_depth_first_iterator::create_end(static_cast<const_pointer>(this));
		}
		// breadth_first_iterator can self determine if it has meet the end, iterate through sub-tree
		const_breadth_first_iterator nodes_breadth_first_begin() const {
			return const_breadth_first_iterator(static_cast<const_pointer>(this), true);
		}
		// just an null-iterator
		const_sibling_iterator nodes_breadth_first_end() const {
			return const_sibling_iterator(nullptr);
		}

		// Mutable ranges

		// iterator throught this and all it's "next" siblings 
		mutable_sibling_iterator siblings_begin() {
			return mutable_sibling_iterator(static_cast<pointer>(this));
		}

		mutable_sibling_iterator siblings_end() {
			return mutable_sibling_iterator(nullptr);
		}
		// List-like iterator over children
		mutable_sibling_iterator children_begin() {
			return mutable_sibling_iterator(static_cast<pointer>(this)->_first_child);
		}
		// List-like iterator over children
		mutable_sibling_iterator children_end() {
			return mutable_sibling_iterator(nullptr);
		}
		// Depth first descendants begin iterator
		mutable_depth_first_iterator descendants_begin() {
			return mutable_depth_first_iterator(static_cast<pointer>(this)->_first_child);
		}
		// Depth first descendants end iterator
		mutable_depth_first_iterator descendants_end() {
			return mutable_depth_first_iterator::create_end(static_cast<pointer>(this));
		}
		// Breadth first descendants iterator can self determine if it has meet the end
		mutable_breadth_first_iterator descendants_breadth_first_begin() {
			return mutable_breadth_first_iterator(static_cast<pointer>(this)->_first_child);
		}
		// just an null-iterator
		mutable_sibling_iterator descendants_breadth_first_end() {
			return mutable_sibling_iterator(nullptr);
		}
		// begin iterator to all nodes inside this sub-tree
		mutable_depth_first_iterator nodes_begin() {
			return mutable_depth_first_iterator(static_cast<pointer>(this));
		}
		// end iterator to all nodes inside this sub-tree
		mutable_depth_first_iterator nodes_end() {
			return mutable_depth_first_iterator::create_end(static_cast<pointer>(this));
		}
		// breadth_first_iterator can self determine if it has meet the end
		mutable_breadth_first_iterator nodes_breadth_first_begin() {
			return const_breadth_first_iterator(static_cast<pointer>(this), true);
		}
		// just an null-iterator
		mutable_sibling_iterator nodes_breadth_first_end() {
			return mutable_sibling_iterator(nullptr);
		}

		// Ranges
	public:
		iterator_range<const_sibling_iterator>
			children() const
		{
			return iterator_range<const_sibling_iterator>(children_begin(), children_end());
		}
		iterator_range<const_depth_first_iterator>
			nodes() const
		{
			return iterator_range<const_depth_first_iterator>(nodes_begin(), nodes_end());
		}
		iterator_range<const_depth_first_iterator>
			descendants() const
		{
			return iterator_range<const_depth_first_iterator>(descendants_begin(), descendants_end());
		}
		iterator_range<const_depth_first_iterator>
			nodes_breadth_first() const
		{
			return iterator_range<const_depth_first_iterator>(nodes_breadth_first_begin(), nodes_breadth_first_end());
		}
		iterator_range<const_depth_first_iterator>
			descendants_breadth_first() const
		{
			return iterator_range<const_depth_first_iterator>(descendants_breadth_first_begin(), descendants_breadth_first_end());
		}
		iterator_range<mutable_sibling_iterator>
			children()
		{
			return iterator_range<mutable_sibling_iterator>(children_begin(), children_end());
		}
		iterator_range<mutable_depth_first_iterator>
			nodes()
		{
			return iterator_range<mutable_depth_first_iterator>(nodes_begin(), nodes_end());
		}
		iterator_range<mutable_depth_first_iterator>
			descendants()
		{
			return iterator_range<mutable_depth_first_iterator>(descendants_begin(), descendants_end());
		}
		iterator_range<mutable_depth_first_iterator>
			nodes_breadth_first()
		{
			return iterator_range<mutable_depth_first_iterator>(nodes_breadth_first_begin(), nodes_breadth_first_end());
		}
		iterator_range<mutable_depth_first_iterator>
			descendants_breadth_first()
		{
			return iterator_range<mutable_depth_first_iterator>(descendants_breadth_first_begin(), descendants_breadth_first_end());
		}
	};

	// 1 pointer + 1 std::vector<> overhaul for each node
	template<typename _Ty>
	class vector_tree_node
	{
	public:
		typedef _Ty value_type;
		typedef value_type& reference;
		typedef value_type* pointer;
		typedef value_type const & const_reference;
		typedef value_type const * const_pointer;

	protected:
		pointer	_parent;
		std::vector<value_type> _children;

	public:
		bool has_parent() const
		{
			return _parent != nullptr;
		}

		const_reference parent() const {
			return *_parent;
		}

		// Logical Parent for this node
		reference parent() {
			return *_parent;
		}

		const std::vector<value_type>& children() const
		{
			return _children;
		}

		std::vector<value_type>& children()
		{
			return _children;
		}

	};
}

#ifdef STDX_TREE_IMPORT_TO_STD
namespace std
{
	using stdx::tree_node;
	using stdx::foward_tree_node;
}
#endif
#endif