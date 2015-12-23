#pragma once
#include "DirectXMathExtend.h"
#include <type_traits>

#ifndef HAS_MEMBER_FUNCTION
#define HAS_MEMBER_FUNCTION(member_name, name)                                   \
    template<typename T, typename Signature>                            \
    struct name {                                                       \
        typedef char yes[1];                                            \
        typedef char no [2];                                            \
        template <typename U, U> struct type_check;                     \
        template <typename _1> static yes &chk(type_check<Signature, &_1::member_name > *); \
        template <typename   > static no  &chk(...);                    \
        static bool const value = sizeof(chk<T>(0)) == sizeof(yes);     \
    }
#endif // !HAS_MEM_FUNC

#ifndef HAS_MEMBER
#define HAS_MEMBER(member_name, name)                                   \
    template<typename T>												\
    struct name {                                                       \
        typedef char yes[1];                                            \
        typedef char no [2];                                            \
        template <typename> static yes &chk(decltype(T::member_name)*);	\
        template <typename> static no  &chk(...);						\
        static bool const value = sizeof(chk<T>(0)) == sizeof(yes);     \
    }
#endif // !HAS_MEM_FUNC

namespace DirectX
{
	namespace VertexTraits
	{
		HAS_MEMBER(position, has_position);
		HAS_MEMBER(normal, has_normal);
		HAS_MEMBER(textureCoordinate, has_tex);
		HAS_MEMBER(uv, has_uv);
		HAS_MEMBER(color, has_color);
		HAS_MEMBER(tanget, has_tanget);
		HAS_MEMBER(weights, has_weights);

		XMGLOBALCONST
			XMUINT4 XMWeightIndexZero = {0U,0U,0U,0U};

		template <class TVertex>
		void XM_CALLCONV SetVertex(
			std::enable_if_t<
			has_position<TVertex>::value &&
			!has_normal<TVertex>::value,
			TVertex>& vertex, FXMVECTOR position, FXMVECTOR normal = g_XMZero.v, FXMVECTOR uv = g_XMZero.v, GXMVECTOR tangent = g_XMZero.v, HXMVECTOR color = g_XMZero.v, HXMVECTOR weights = g_XMZero.v, const XMUINT4 &index = XMWeightIndexZero)
		{
			vertex.position = position;
		}

		template <class TVertex>
		void XM_CALLCONV SetVertex( 
			std::enable_if_t<
				has_position<TVertex>::value &&
				has_normal<TVertex>::value && 
				!has_uv<TVertex>::value && 
				!has_tex<TVertex>::value,
				TVertex>& vertex, FXMVECTOR position, FXMVECTOR normal, FXMVECTOR uv = g_XMZero.v, GXMVECTOR tangent = g_XMZero.v, HXMVECTOR color = g_XMZero.v, HXMVECTOR weights = g_XMZero.v, const XMUINT4 &index = XMWeightIndexZero)
		{
			((Vector3&)(vertex.position)) = position;
			((Vector3&)(vertex.normal)) = normal;
		}

		template <class TVertex>
		void XM_CALLCONV SetVertex(
			std::enable_if_t<
				has_position<TVertex>::value &&
				has_normal<TVertex>::value && 
				has_uv<TVertex>::value &&
				!has_tex<TVertex>::value &&
				!has_tanget<TVertex>::value,
				TVertex>& vertex, FXMVECTOR position, FXMVECTOR normal, FXMVECTOR uv, GXMVECTOR tangent = g_XMZero.v, HXMVECTOR color = g_XMZero.v, HXMVECTOR weights = g_XMZero.v, const XMUINT4 &index = XMWeightIndexZero)
		{
			((Vector3&)(vertex.position)) = position;
			((Vector3&)(vertex.normal)) = normal;
			((Vector2&)(vertex.uv)) = uv;
		}

		template <class TVertex>
		void XM_CALLCONV SetVertex(
			std::enable_if_t<
				has_position<TVertex>::value &&
				has_normal<TVertex>::value &&
				has_tex<TVertex>::value &&
				!has_uv<TVertex>::value &&
				!has_tanget<TVertex>::value,
				TVertex>& vertex, FXMVECTOR position, FXMVECTOR normal, FXMVECTOR uv, GXMVECTOR tangent = g_XMZero.v, HXMVECTOR color = g_XMZero.v, HXMVECTOR weights = g_XMZero.v, const XMUINT4 &index = XMWeightIndexZero)
		{
			((Vector3&)(vertex.position)) = position;
			((Vector3&)(vertex.normal)) = normal;
			((Vector2&)(vertex.textureCoordinate)) = uv;
		}

		template <class TVertex>
		void XM_CALLCONV SetVertex(
			std::enable_if_t<
			has_position<TVertex>::value &&
			has_normal<TVertex>::value &&
			has_tex<TVertex>::value &&
			has_tanget<TVertex>::value,
			TVertex>& vertex, FXMVECTOR position, FXMVECTOR normal, FXMVECTOR uv, GXMVECTOR tangent, HXMVECTOR color = g_XMZero.v, HXMVECTOR weights = g_XMZero.v, const XMUINT4 &index = XMWeightIndexZero)
		{
			((Vector3&)(vertex.position)) = position;
			((Vector3&)(vertex.normal)) = normal;
			((Vector2&)(vertex.textureCoordinate)) = uv;
			((Vector4&)(vertex.tagent)) = uv;
		}
	}
}