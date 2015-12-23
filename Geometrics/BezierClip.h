#pragma once

#include <array>
#include <algorithm>
#include <cassert>

namespace Geometrics
{
	namespace Bezier
	{
		namespace Internal
		{

			/// <summary>
			/// Helper class for accessing combination numbers , compute in static way
			/// </summary>
			template <size_t N>
			class Combine
			{
			protected:
				std::array<int,N+1> D;

			public:
				Combine()
				{
					D[0] = 1;
					for (int i = 0; i < N; i++)
					{
						D[i+1] = 1;
						for (int j = i; j >= 1 ; j--)
						{
							D[j] += D[j-1];
						}
					}
				}

				int operator[] (size_t K) const
				{
					return D[K];
				}
			};

			constexpr size_t Combination(size_t d, size_t n)
			{
				return (d < 0 || d > n) ? 0 : (((d == 0) || (n == d)) ? 1 : (Combination(d - 1, n - 1) + Combination(d, n - 1)));
			}

			// static_assert
			//const static size_t C62 = Combination(2, 6);
			//const static size_t C64 = Combination(4, 6);
			//const static size_t C63 = Combination(3, 6);
		}

		template <typename _Ty , size_t _Order>
		struct BezierClipping
			: public std::array< _Ty,_Order+1>
		{
		public:
			static const size_t Order = _Order;
			typedef _Ty ValueType;
			typedef std::array< _Ty,Order+1> Collection;
			typedef Internal::Combine<Order> CombinationType;
			typedef BezierClipping SelfType;
		protected:
			//const static Internal::Combine<Order> Combination;
			static constexpr size_t Combination(size_t d)
			{
				return Internal::Combination(d, Order);
			}

			//protected:
			//	/// <summary>
			//	/// Initializes a new instance of the <see cref="BezierClipping{_Ty, _N}"/> class with the composite of clip0 and clip1
			//	/// </summary>
			//	/// <param name="Clip0">The clip0.</param>
			//	/// <param name="Clip1">The clip1.</param>
			//	BezierClipping(const BezierClipping& Clip0 , const BezierClipping& Clip1)
			//	{
			//		for (size_t i = 0; i <= Order; i++)
			//		{
			//			(*this)[i] = Clip0[i] + Clip1[i];
			//		}
			//	}

		public:
			BezierClipping() = default;
			BezierClipping(const SelfType&) = default;
			BezierClipping(BezierClipping&& rhs)
			{
				*this = std::move(rhs);
			}
			BezierClipping(const Collection & ControlPoints)
			{
				*this = ControlPoints;
			}
			BezierClipping(Collection&& rhs)
			{
				*this = std::move(rhs);
			}
			using Collection::operator=;
			BezierClipping& operator=(const BezierClipping& rhs) = default;
			BezierClipping& operator=(BezierClipping&& rhs)
			{
				Collection::operator=(std::move(rhs));
				return *this;
			}

			/// <summary>
			/// Subdivides the Clip into 2 Clip with in the specified ratio r.
			/// </summary>
			/// <param name="r">The subdivide ratio r.</param>
			/// <param name="SubClip0">The SubClip 0.</param>
			/// <param name="SubClip1">The SubClip 1.</param>
			void divide(_In_ const float r , _Out_ BezierClipping& FrontClipping , _Out_ BezierClipping& BackClipping) const
			{
				assert(0<=r && r<=1);
				const float q = 1-r;
				BezierClipping& C = BackClipping;
				C = *this;

				FrontClipping[0] = C[0];
				for (size_t i = Order; i > 0; --i)
				{
					for (size_t j = 0; j < i; ++j)
					{
						C[j] = (q*C[j]) + (r*C[j+1]);
					}
					FrontClipping[Order-i] = C[0];
				}
			}

			/// <summary>
			/// Divides the clipping with specified ratio r , the front clipping will storage in (*this) , and the back clipping will output by Parameter "BackClipping".
			/// </summary>
			/// <param name="r">The r.</param>
			/// <param name="BackClipping">The back clipping.</param>
			void divide(_In_ const float r , _Out_ BezierClipping& BackClipping)
			{
				assert(0<=r && r<=1);
				const float q = 1-r;
				auto& Front = *this;

				BackClipping[Order] = Front[Order];
				for (size_t i = Order; i > 0; --i)
				{
					for (size_t j = Order; j > Order-i; --j)
					{
						Front[j] = (q*Front[j-1]) + (r*Front[j]);
					}
					BackClipping[i-1] = Front[Order];
				}
			}

			void crop_front(float r)
			{
				assert(0<=r && r<=1);
				const float q = 1-r;

				for (size_t i = Order; i > 0; --i)
				{
					for (size_t j = 0; j < i; ++j)
					{
						(*this)[j] = (q*(*this)[j]) + (r*(*this)[j+1]);
					}
				}
			}

			void crop_back(float r)
			{
				assert(0<=r && r<=1);
				const float q = 1-r;

				for (size_t i = Order; i > 0; --i)
				{
					for (size_t j = Order; j > Order-i; --j)
					{
						(*this)[j] = (q*(*this)[j-1]) + (r*(*this)[j]);
					}
				}
			}

			void crop(float s , float t)
			{
				if (s==t)
				{
					float value = (*this)(s);
					this->fill(value);
					return;
				}

				crop_front(s);
				t = (t-s)/(1-s);
				crop_back(t);
			}


			//void reverse()
			//{
			//	std::reverse(D.begin(),D.end());
			//}

			//void fill(const _Ty & value)
			//{
			//	D.fill(value);
			//}


			BezierClipping subclip(float s, float t) const
			{
				assert (s < t);
				BezierClipping clip;
				if (s == 0.0f)
				{
					clip.crop_back(t);
				}else if (t == 1.0f)
				{
					clip.crop_front(s);
				} else
				{
					clip.crop_front(s);
					clip.crop_back((t-s)/(1-s));
				}
				return clip;
			}

			ValueType eval(float t) const{	
				assert(0<=t && t<=1);

				float q = 1-t;
				std::array<float,Order+1> P,Q;
				P[0] = 1.0f; Q[0] = 1.0f;
				for (size_t i = 0; i < Order; i++)
				{
					P[i+1] = P[i] * t;
					Q[i+1] = Q[i] * q;
				}

				_Ty value = (*this)[0] * Q[Order];
				for (int i = 1; i <= Order; i++)
				{
					value += P[i] * Q[Order - i] * Combination(i) * (*this)[i];
				}

				return value;
			}

			ValueType tangent(float t) const
			{	
				assert(0<=t && t<=1);

				float q = 1-t;
				std::array<float,Order+1> P,Q;
				P[0] = 1.0f; Q[0] = 1.0f;
				for (size_t i = 0; i < Order; i++)
				{
					P[i+1] = P[i] * t;
					Q[i+1] = Q[i] * q;
				}

				_Ty value = (1 - Order) * (*this)[0] * Q[Order-1]; // i == 0
				value += (Order - 1) * (*this)[Order] *P[Order-1]; // i==Order
				for (int i = 1; i < Order; i++)
				{
					float cof = i - Order*t;
					value += cof * P[i-1] * Q[Order - i - 1] * Combination(i) * (*this)[i];
				}

				return value;
			}

			inline ValueType operator()(float t) const{	
				return eval(t);
			}

			//_Ty& operator[](size_t index){
			//	return D[index];
			//}

			//const _Ty& operator[](size_t index) const{
			//	return D[index];
			//}

			BezierClipping& compound(const BezierClipping &rhs)
			{
				for (size_t i = 0; i <= Order; i++)
				{
					(*this)[i] += rhs[i];
				}
				return *this;
			}


			//BezierClipping& operator+=(const BezierClipping& rhs)
			//{
			//	for (size_t i = 0; i <= Order; i++)
			//	{
			//		D[i] += Clip1.D[i];
			//	}
			//}

		protected:
			// the control points located in a uniform distribute in parameter space
			//Collection D;
		};

		template <typename _Ty , size_t _Order>
		struct BezierPatch
			: public std::array<std::array<_Ty,_Order+1>,_Order+1>
		{
		public:
			static const size_t Order = _Order;
			typedef _Ty ValueType;
			typedef std::array<std::array<_Ty,_Order+1>,_Order+1> BaseType;
			typedef std::array< _Ty,Order+1> Collection;
			typedef Internal::Combine<Order> CombinationType;
			typedef BezierClipping<ValueType,Order> ClippingType;

		protected:
			const static Internal::Combine<Order> Combination;
		public:
			const ValueType* data() const
			{
				return BaseType::front().data();
			}

			ValueType* data() 
			{
				return BaseType::front().data();
			}

			ClippingType row_clipping(float s) const
			{
				assert(0<=s && s<=1);
				ClippingType clipping;

				float q = 1-s;
				std::array<float,Order+1> P,Q;
				P[0] = 1.0f; Q[0] = 1.0f;
				for (size_t i = 0; i < Order; i++)
				{
					P[i+1] = P[i] * s;
					Q[i+1] = Q[i] * q;
				}

				for (size_t i = 0; i <= Order; i++)
				{
					_Ty value = (*this)[i][0] * Q[Order];
					for (size_t j = 1; j <= Order; j++)
					{
						value += P[j] * Q[Order - j] * Combination[j] * (*this)[i][j];
					}
					clipping[i] = value;
				}

				return clipping;
			}

			ClippingType col_clipping(float t) const
			{
				assert(0<=t && t<=1);
				ClippingType clipping;

				float q = 1-t;
				std::array<float,Order+1> P,Q;
				P[0] = 1.0f; Q[0] = 1.0f;
				for (size_t i = 0; i < Order; i++)
				{
					P[i+1] = P[i] * t;
					Q[i+1] = Q[i] * q;
				}

				for (size_t i = 0; i <= Order; i++)
				{
					_Ty value = (*this)[0][i] * Q[Order];
					for (size_t j = 1; j <= Order; j++)
					{
						value += P[j] * Q[Order - j] * Combination[j] * (*this)[j][i];
					}
					clipping[i] = value;
				}

				return clipping;
			}

			ValueType eval(float s , float t) const
			{
				assert(0<=s && s<=1);
				assert(0<=t && t<=1);
				ClippingType clipping = row_clipping(s);
				return clipping.eval(t);
			}

			inline ValueType operator()(float s , float t) const
			{
				return eval(s,t);
			}

			const ClippingType& top() const
			{
				return static_cast<const ClippingType&>((*this)[0]);
			}

			const ClippingType& bottom() const
			{
				return static_cast<const ClippingType&>((*this)[Order]);
			}

			ClippingType left() const
			{
				ClippingType clipping;
				for (size_t i = 0; i < Order; i++)
				{
					clipping[i] = (*this)[i][0];
				}
				return clipping;
			}

			ClippingType right() const
			{
				ClippingType clipping;
				for (size_t i = 0; i < Order; i++)
				{
					clipping[i] = (*this)[i][Order];
				}
				return clipping;
			}

			void crop_top(float s)
			{
				assert(0<=s && s<=1);
				ClippingType clipping;
				for (size_t i = 0; i < Order; i++)
				{
					for (size_t j = 0; j < Order; j++)
					{
						clipping[j] = (*this)[j][i];
					}
					clipping.crop_front(s);
					for (size_t j = 0; j < Order; j++)
					{
						(*this)[j][i] = clipping[j];
					}
				}
			}

			void crop_bottom(float s)
			{
				assert(0<=s && s<=1);
				ClippingType clipping;
				for (size_t i = 0; i < Order; i++)
				{
					for (size_t j = 0; j < Order; j++)
					{
						clipping[j] = (*this)[j][i];
					}
					clipping.crop_back(s);
					for (size_t j = 0; j < Order; j++)
					{
						(*this)[j][i] = clipping[j];
					}
				}
			}
			void crop_left(float t)
			{
				assert(0<=t && t<=1);
				for (size_t i = 0; i < Order; i++)
				{
					ClippingType& clipping = static_cast<ClippingType&>((*this)[i]);
					clipping.crop_front(t);
				}
			}

			void crop_right(float t)
			{
				assert(0<=t && t<=1);
				for (size_t i = 0; i < Order; i++)
				{
					ClippingType& clipping = static_cast<ClippingType&>((*this)[i]);
					clipping.crop_back(t);
				}
			}

			void crop(float top,float bottom,float left,float right)
			{
				assert(0<=top && top<=bottom && bottom<=1);
				assert(0<=left && left<=right && right<=1);

				ClippingType clipping;
				for (size_t i = 0; i < Order; i++)
				{
					for (size_t j = 0; j < Order; j++)
					{
						clipping[j] = (*this)[j][i];
					}
					clipping.crop(top,bottom);
					for (size_t j = 0; j < Order; j++)
					{
						(*this)[j][i] = clipping[j];
					}
				}

				for (size_t i = 0; i < Order; i++)
				{
					ClippingType& clipping = static_cast<ClippingType&>((*this)[i]);
					clipping.crop(left,right);
				}
			}

		};

		namespace Internal{	
			template <size_t Order>
			bool if_have_root(BezierClipping<float,Order> &clipping)
			{
				const unsigned int MaxIteration = 5;
				const float delta = 1.0f / (float)Order;
				auto& D = clipping;
				unsigned int C = 0;

				float begin = 0.0f , end = 1.0f;

				do
				{
					if (D[0] * D[Order] < 0) return true;

					int m = 0;
					for (int i = 0; i < Order; i++)
					{
						if (D[i] * D[i+1] < 0)
							++m;
					}

					if (!m) return false; // No root

					auto minmax = convex_hull_intersection(D);
					auto tmin = minmax.first;
					auto tmax = minmax.second;


					{
						float dis = end-begin;
						begin += dis*tmin;
						end -= dis*(1-tmax);
						D.crop(tmin,tmax);
						if (D[0] * D[Order] < 0) return true;
					}

					if (m > 1 || (tmax-tmin) > 0.5f)
					{
						BezierClipping<float,Order> back;
						D.divide(0.5f,back);
						return if_have_root(D) || if_have_root(back);
					}
				} while (C++ < MaxIteration);
				return false;
			}

			template <size_t Order>
			inline bool if_have_root(const BezierClipping<float,Order> &clipping)
			{
				auto D = clipping;
				return if_have_root(D);

			}

			template <size_t Order>
			inline bool if_have_root(BezierClipping<float,Order> &&clipping)
			{
				return if_have_root(clipping);
			}

			template <size_t Order>
			float min_value(const BezierClipping<float,Order> &clipping,float preciese,float known_min)
			{
				const auto& D = clipping;
				unsigned int C = 0;

				int m = 0; 
				float min_t = 1.0f; 
				float min_v = D[Order];
				for (int i = 0; i < Order; i++)
				{
					if (D[i] * D[i+1] < 0)
						++m;
					if (D[i] < min_v) 
					{
						min_v = D[i];
						min_t = static_cast<float>(i) / static_cast<float>(Order);
					}
				}

				if (!m || preciese > 1.0f) return min_v; // No root
				if (min_v > known_min ) return known_min;

				BezierClipping<float,Order> front,back;
				clipping.divide(0.5f,front,back);
				min_v = clipping.back();
				float pcs = preciese * 2;
				min_v = min_value(front,pcs,min_v);
				min_v = min_value(back,pcs,min_v);
				return min_v;
			}

		}

		/// <summary>
		/// Find the min and max intersection point of the t-axis with convex hull of the giving Bezier clipping.
		/// </summary>
		/// <param name="D">The D.</param>
		/// <returns>pair of (tmin,tmax)</returns>
		template <size_t Order>
		inline std::pair<float,float> convex_hull_intersection(const BezierClipping<float,Order> &D)
		{
			const float delta = 1.0f / (float)Order;
			float tmin , tmax;
			int l ,k ,r;

			if (D[0] == 0) return std::make_pair(.0f,.0f);
			if (D[Order] == 0) return std::make_pair(1.0f,1.0f);

			// Caculate tmin
			float dx = delta;
			float cSlope = 0;
			k = 0;
			for (int i = 1; i <= Order; ++i,dx+=delta)
			{
				if (D[0] * D[i] < 0)
				{
					float slope = (D[i] - D[0]) / dx;
					slope = std::abs(slope);
					if (slope > cSlope)
					{
						cSlope = slope;
						k = i;
					}
				}
			}

			//if (!k) return -1.0f;

			cSlope = 1e6f;
			dx = delta;
			l = k;

			for (int i = k-1; i >= 0; --i,dx+=delta)
			{
				if (D[i] * D[k] < 0)
				{
					float slope = (D[k] - D[i]) / dx;
					slope = std::abs(slope);
					if (slope < cSlope)
					{
						cSlope = slope;
						l = i;
					}
				}
			}

			tmin = std::abs(D[l]);
			tmin /= tmin + std::abs(D[k]);
			tmin = (l*(1-tmin) + k*tmin)*delta; 

			//Caculate tmax
			dx = delta;
			cSlope = 0;
			k = Order;
			for (int i = Order-1; i >= 0 ; --i,dx+=delta)
			{
				if (D[Order] * D[i] < 0)
				{
					float slope = (D[Order] - D[i]) / dx;
					slope = std::abs(slope);
					if (slope > cSlope)
					{
						cSlope = slope;
						k = i;
					}
				}
			}

			//if (!k) return -1.0f; // No possible for no root case
			cSlope = 1e6f;
			dx = delta;
			r = k;

			for (int i = k+1; i <= Order; i++,dx+=delta)
			{
				if (D[i] * D[k] < 0)
				{
					float slope = (D[i] - D[k]) / dx;
					slope = std::abs(slope);
					if (slope < cSlope)
					{
						cSlope = slope;
						r = i;
					}
				}
			}

			tmax = std::abs(D[k]);
			tmax /= tmax + std::abs(D[r]);
			tmax = (k*(1-tmax) + r*tmax)*delta; 
			// Finish Find the convex-hull t-axis intersection (tmin,tmax)

			assert(tmin<=tmax);
			return std::make_pair(tmin,tmax);
		}


		/// <summary>
		/// Soloves the specified clipping intersect with a value T. return -1.0f if there is not root
		/// Equation : B(root) = Value;
		/// </summary>
		/// <param name="clipping">The clipping.</param>
		/// <param name="T">The target value T.</param>
		/// <param name="preciese">The preciese.</param>
		/// <returns> The valid return value will always in [0,1]</returns>
		template <size_t Order>
		float solove_first_root(BezierClipping<float,Order> clipping,float T,float preciese)
		{
			unsigned int N = 0;
			const float delta = 1.0f / (float)Order;
			const auto& D = clipping;
			unsigned int C = 0;
			const unsigned int MaxIteration = 5;

			if (T!=0.0f){
				for (auto& obj : clipping)
				{
					obj -= T;
				}
			}

			float begin = 0.0f , end = 1.0f;

			do
			{
				int m = 0;
				for (int i = 0; i < Order; i++)
				{
					if (D[i] * D[i+1] < 0)
						++m;
				}

				if (!m) return -1.0f; // No root

				auto minmax = convex_hull_intersection(D);
				auto tmin = minmax.first;
				auto tmax = minmax.second;

				{
					float dis = end-begin;
					begin += dis*tmin;
					end -= dis*(1-tmax);
					clipping.crop(tmin,tmax);
				}
				if (m > 1 || (tmax-tmin) > 0.5f)
				{
					BezierClipping<float,Order> back;
					clipping.divide(0.5f,back);
					float pcs = preciese * 2 / (end-begin);
					float root = solove_first_root(clipping,0.0f,pcs);
					if (root >= 0.0f && root <= 1.0f)
						return root * (end - begin)*0.5f +begin;
					else
					{
						root = solove_first_root(back,0.0f,pcs);
						if (root >= 0.0f && root <= 1.0f)
							return (0.5f*root + 0.5f)*(end - begin) + begin;
						else return root;
					}
				}
			} while (end - begin > preciese && C++ < MaxIteration);
			if (end - begin <= preciese)
			{
				float root = std::abs(D[Order]);
				root /= root + std::abs(D[0]);
				root = (end-begin)*root + begin;
				return root;
			} else
			{
				return -1.0f;
			}

		}



		/// <summary>
		/// Soloves the specified clipping intersect with a value T. return -1.0f if there is not root
		/// Equation : B(root) = Value;
		/// </summary>
		/// <param name="clipping">The clipping.</param>
		/// <param name="T">The target value T.</param>
		/// <param name="preciese">The preciese.</param>
		/// <returns> The valid return value will always in [0,1]</returns>
		template <size_t Order>
		inline bool if_have_root(BezierClipping<float,Order> clipping,float T)
		{
			if (T!=0.0f){
				for (auto& obj : clipping)
				{
					obj -= T;
				}
			}
			return Internal::if_have_root(std::move(clipping));
		}

		template <size_t Order>
		float min_value(const BezierClipping<float,Order> &clipping,float preciese)
		{
			return Internal::min_value(clipping,preciese,std::numeric_limits<float>::infinity());
		}

	}
}