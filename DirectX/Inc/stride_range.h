#pragma once

#ifndef STDX_STRIDE_RANGE
#define STDX_STRIDE_RANGE

#include <iterator>
#include <cstdint>
#include <type_traits>
#include <boost\range.hpp>
#include <boost\range\adaptors.hpp>

namespace stdx {
	using boost::iterator_range;
	using boost::sub_range;

	template <class T>
	class stride_range;

	template <class _Ty>
	class stride_iterator : public std::iterator<std::random_access_iterator_tag, _Ty>
	{
	private:
		pointer data;
		// stride in byte
		size_t stride;

	protected:
		// these friend declarations are to allow access to the protected
		// "raw" constructor that starts from a raw pointer plus
		// stride+length info
		template <class U> friend class stride_iterator;
		template <class U> friend class stride_range;

		stride_iterator(pointer data, size_t stride) :
			data(data), stride(stride) {}

	public:
		typedef stride_iterator<_Ty> _Myiter;

		stride_iterator(const stride_iterator& other) :
			data(other.data), stride(other.stride) {}

		template <class U>
		explicit stride_iterator(const stride_iterator<U>& other) :
			data(other.data), stride(other.stride) {}


		bool has_more() const { return data < stop; }

		// Dereference

		reference operator*()
		{
			return *data;
		}
		reference operator[](int idx)
		{
			auto ptr = reinterpret_cast<pointer>(reinterpret_cast<char*>(data) + stride*idx);
			return *ptr;
		}

		// Comparison

		bool operator==(const _Myiter& other) const { return data == other.data; }

		bool operator!=(const _Myiter& other) const { return data != other.data; }

		bool operator<(const _Myiter& other) const { return data < other.data; }

		difference_type operator-(const _Myiter& other) const
		{
			return ((char*) data - (char*) other.data) / stride;
		}

		// Increment/Decrement

		_Myiter& operator++() { data = reinterpret_cast<pointer>(reinterpret_cast<char*>(data) + stride); return *this; }
		_Myiter& operator--() { data = reinterpret_cast<pointer>(reinterpret_cast<char*>(data) - stride); return *this; }

		_Myiter operator++(int) { _Myiter cpy(*this); data += stride; return cpy; }
		_Myiter operator--(int) { _Myiter cpy(*this); data -= stride; return cpy; }

		_Myiter& operator+=(int x) { data = &(*this)[x]; return *this; }
		_Myiter& operator-=(int x) { data = &(*this)[-x]; return *this; }

		_Myiter operator+(int x) const { _Myiter res(*this); res += x; return res; }
		_Myiter operator-(int x) const { _Myiter res(*this); res -= x; return res; }

		//#if _ITERATOR_DEBUG_LEVEL == 2
		//	void _Compat(const _Myiter& _Right) const
		//	{	// test for compatible iterator pair
		//		if (this->_Getcont() == 0
		//			|| this->_Getcont() != _Right._Getcont())
		//		{	// report error
		//			_DEBUG_ERROR("iterators incompatible");
		//			_SCL_SECURE_INVALID_ARGUMENT;
		//		}
		//	}
		//
		//#elif _ITERATOR_DEBUG_LEVEL == 1
		//	void _Compat(const _Myiter& _Right) const
		//	{	// test for compatible iterator pair
		//		_SCL_SECURE_VALIDATE(this->_Getcont() != 0);
		//		_SCL_SECURE_VALIDATE_RANGE(this->_Getcont() == _Right._Getcont());
		//	}
		//
		//#else /* _ITERATOR_DEBUG_LEVEL == 0 */
		//	void _Compat(const _Myiter&) const
		//	{	// test for compatible iterator pair
		//	}
		//#endif /* _ITERATOR_DEBUG_LEVEL */
		//
	};
}

namespace std {
	template<class _Container>
	struct _Is_checked_helper<stdx::stride_iterator<_Container> >
		: public true_type
	{	// mark back_insert_iterator as checked
	};
}

namespace stdx {
	// a stride_range is a wrapper of a contious container, which expose a random access range interface 
	template <class _Ty>
	class stride_range
	{
	public:
		typedef	std::random_access_iterator_tag	iterator_category;
		//typedef stride_range<T>							_SelfType;
		typedef std::remove_reference_t<_Ty>					value_type;
		typedef ptrdiff_t										difference_type;
		typedef value_type*										pointer;
		typedef value_type&										reference;

		typedef stride_iterator<value_type>						iterator_type;
		typedef stride_iterator<std::add_const_t<value_type>>	const_iterator_type;
	protected:
		pointer data;
		// stride in byte
		size_t	stride;
		pointer stop;
	public:
		stride_range()
			:data(nullptr), stride(1), stop(nullptr)
		{}

		stride_range(pointer data, size_t stride, size_t count)
			:data(data), stride(stride), stop(reinterpret_cast<pointer>(reinterpret_cast<char*>(data) + stride*count))
		{}

		template <class TContiniousContainer>
		explicit stride_range(const TContiniousContainer& container)
			: stride_range(container.data(),sizeof(container[0]), container.size())
		{}

		void reset(pointer data, size_t stride, size_t count)
		{
			this->data = data;
			this->stride = stride;
			this->stop = reinterpret_cast<pointer>(reinterpret_cast<char*>(data) + stride*count);
		}

		void reset()
		{
			data = nullptr;
			stride = 0;
			stop = nullptr;
		}

		bool empty() const
		{
			return stop - data == 0;
		}

		size_t size()
		{
			return ((char*) stop - (char*) data) / stride;;
		}

		iterator_type begin()
		{
			return iterator_type(data, stride);
		}

		iterator_type end()
		{
			return iterator_type(stop, stride);
		}

		const_iterator_type begin() const
		{
			return const_iterator_type(data, stride);
		}

		const_iterator_type end() const
		{
			return const_iterator_type(stop, stride);
		}

		const_iterator_type cbegin() const
		{
			return const_iterator_type(data, stride);
		}

		const_iterator_type cend() const
		{
			return const_iterator_type(stop, stride);
		}

		reference operator[](int idx)
		{
			auto ptr = reinterpret_cast<pointer>(reinterpret_cast<char*>(data) + stride*idx);
#if _ITERATOR_DEBUG_LEVEL == 2
			if (stop <= ptr)
			{	// report error
				_DEBUG_ERROR("vector subscript out of range");
				_SCL_SECURE_OUT_OF_RANGE;
			}

#elif _ITERATOR_DEBUG_LEVEL == 1
			_SCL_SECURE_VALIDATE_RANGE(ptr < stop);
#endif /* _ITERATOR_DEBUG_LEVEL */
			return *ptr;
		}

		const reference operator[](int idx) const
		{
			auto ptr = reinterpret_cast<const pointer>(reinterpret_cast<const char*>(data) + stride*idx);
#if _ITERATOR_DEBUG_LEVEL == 2
			if (stop <= ptr)
			{	// report error
				_DEBUG_ERROR("vector subscript out of range");
				_SCL_SECURE_OUT_OF_RANGE;
			}

#elif _ITERATOR_DEBUG_LEVEL == 1
			_SCL_SECURE_VALIDATE_RANGE(ptr < stop);
#endif /* _ITERATOR_DEBUG_LEVEL */
			return *ptr;
		}

	};

}
#endif