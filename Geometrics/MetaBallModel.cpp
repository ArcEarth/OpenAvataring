#include "MetaBallModel.h"
//#include <iostream>
#include <set>
#include <map>
#include <unordered_map>

#ifndef This
#define This (*this)
#endif
#ifdef _DEBUG
#define Polygonize_Precise 0.1f
#else
#define  Polygonize_Precise 0.03f
#endif // _DEBUG

#ifndef MODELING_ISO
#define MODELING_ISO 0.5f
#endif // !MODELING_ISO

using namespace DirectX;
using namespace Geometrics;

//const Metaball::ClipType::CombinationType Metaball::ClipType::Combination;

static void CreateBoundingBoxFromSpheres( BoundingBox& Out, size_t Count, const BoundingSphere* pSpheres, size_t Stride )
{
	assert( Count > 0 );
	assert( pSpheres );

	// Find the minimum and maximum x, y, and z
	XMVECTOR vMin, vMax;

	{
		XMVECTOR vCenter = XMLoadFloat3( &pSpheres->Center );
		XMVECTOR vRadius = XMVectorReplicate( pSpheres->Radius );
		vMin = vCenter - vRadius;
		vMax = vCenter + vRadius;
	}

	for( size_t i = 1; i < Count; ++i )
	{
		const BoundingSphere *pSphere = reinterpret_cast<const BoundingSphere*>( reinterpret_cast<const uint8_t*>(pSpheres) + i * Stride );
		XMVECTOR vCenter = XMLoadFloat3( &pSphere->Center );
		XMVECTOR vRadius = XMVectorReplicate( pSphere->Radius );
		XMVECTOR vPoint = vCenter - vRadius;
		vMin = XMVectorMin( vMin, vPoint );
		vPoint = vCenter + vRadius;
		vMax = XMVectorMax( vMax, vPoint );
	}

	// Store center and extents.
	XMStoreFloat3( &Out.Center, ( vMin + vMax ) * 0.5f );
	XMStoreFloat3( &Out.Extents, ( vMax - vMin ) * 0.5f );
}



Metaball::Metaball(const DirectX::Vector3 &_Position,float _Radius,unsigned int _BindingIndex)
	: Position(_Position) , Radius(_Radius) //, BindingIndex(_BindingIndex)// , Coefficient(Cof)
	//, Color((const float*)Colors::Azure)
{}

Metaball::Metaball(DirectX::FXMVECTOR _Position,float _Radius,unsigned int _BindingIndex)
	: Position(_Position) , Radius(_Radius) //, BindingIndex(_BindingIndex)// , Coefficient(Cof)
	//, Color((const float*)Colors::Azure)
{}

Metaball::Metaball( const DirectX::Vector4 & Sphere,unsigned int _BindingIndex /*= 0*/ )
	: Position(Sphere.x,Sphere.y,Sphere.z) , Radius(Sphere.w) //, BindingIndex(_BindingIndex)
	//, Color((const float*)Colors::Azure)
{}
//// r is the distance , where R is the radius of the metaball
//inline float R2Function(float r2, float R)
//{
//	float b = MODELING_FACTOR_BbyR*R;
//	float b2 = b*b;
//	if (r2 <= b2) {
//		if (r2 <= b2/9){
//			return (1-3*r2/b2);
//		}
//		else
//		{
//			float r = sqrtf(r2);
//			//float r = DirectX::sqrtEst2(r2);
//			float t = r / b;
//			t = 1.0f - t;  //This statement cost even more than sqrt......
//			t = t * t;
//			t = 1.5f * t;
//			return t;
//		}
//	} else
//	{
//		return 0.0f;
//	}
//}	

//// r is the distance , where R is the radius of the metaball
//inline float R2Function(float r2, float R)
//{
//	R *= R;
//	if (r2>=R) return 0.0f;
//	float t = r2 / R;
//	float t2 = t * t;
//	float t3 = t2 * t;
//	float value = 1.0f;
//	value -= (22.0f/9.0f) * t;
//	value += (17.0f/9.0f) * t2;
//	value -= ( 4.0f/9.0f) * t3;
//	return value;
//}	

inline float PolynomialKernel_6::DecayFunction(float t)
{
	assert (t >= 0 && t <= 1);
	float t2 = t * t;
	float t3 = t2 * t;
	float value = 1.0f;
	value -= (22.0f/9.0f) * t;
	value += (17.0f/9.0f) * t2;
	value -= ( 4.0f/9.0f) * t3;
	return value;
}

inline float PolynomialKernel_6::DecayDerivative(float t)
{
	assert (t >= 0 && t <= 1);
	float f = - 22.0f/9.0f;
	f += 34.0f/9.0f * t;
	t *= t;
	f -= 4.0f/3.0f * t;
	return f;
}

inline PolynomialKernel_6::SegmentFunctionType PolynomialKernel_6::SegementFunction(float a)
{
	SegmentFunctionType Clipping;
	Clipping[0]=Clipping[1]=Clipping[5]=Clipping[6]=0.0f;
	float a2 = a*a;
	Clipping[2] = Clipping[4] = 16.0f/27.0f*a2;
	Clipping[3] = 8.0f/45.0f*(8*a+5)*a2;
	return Clipping;
}

#if defined (_DEBUG)
#define assume assert
#else
#define assume __assume
#endif

inline float Metaball::DecayFunction(float t)
{
	assume(t >= 0 && t <= 1);
	float t2 = t * t;
	float t3 = t2 * t;
	float value = 1.0f;
	value -= (22.0f/9.0f) * t;
	value += (17.0f/9.0f) * t2;
	value -= ( 4.0f/9.0f) * t3;
	return value;
}

//inline float Metaball::DecayFunction(float r2, float R)
//{
//	R *= R;
//	if (r2>=R) return 0.0f;
//	float t = r2 / R;
//	return DecayFunction(t);
//}

inline float Metaball::DecayDerivative(float t)
{
	assume(t >= 0 && t <= 1);
	float f = - 22.0f/9.0f;
	f += 34.0f/9.0f * t;
	t *= t;
	f -= 4.0f/3.0f * t;
	return f;
}

float Metaball::EffectiveRadiusRatio(float ISO , float precise/* = 1e-6*/)
{
	float t = 1.0f-ISO;
	float dt,ft;
	do
	{
		ft = DecayFunction(t);
		dt = DecayDerivative(t);
		dt = (ISO - ft) / dt;
		t += dt;
	} while (std::abs(dt) > precise);

	assert(std::abs(DecayFunction(t) - ISO) < 0.001f);
	return std::sqrtf(t);
}

float XM_CALLCONV Metaball::eval(FXMVECTOR pos) const
{
	float r2 = DirectX::XMVectorGetX(DirectX::XMVector3LengthSq(pos - XMLoadA(Position)));
	float R2 = Radius*Radius;
	if (r2>=R2) return 0.0f;
	float t = r2 / R2;
	return DecayFunction(t);
}

XMVECTOR XM_CALLCONV Metaball::grad(FXMVECTOR pos) const
{
	XMVECTOR vtr = pos - XMLoadA(Position);
	float r2 = DirectX::XMVectorGetX(DirectX::XMVector3LengthSq(vtr));
	float R2 = Radius*Radius;
	if (r2 <= R2){
		R2 = 1/R2;
		float t = r2*R2;

		float f = DecayDerivative(t);
		f *= R2 * 2;
		return f*vtr;
	} else
	{
		return g_XMZero;
	}
}

XMVECTOR XM_CALLCONV Metaball::evalgrad(FXMVECTOR pos) const
{
	XMVECTOR vtr = pos - XMLoadA(Position);
	float r2 = DirectX::XMVectorGetX(DirectX::XMVector3LengthSq(vtr));
	float R2 = Radius*Radius;
	if (r2 <= R2) {
		R2 = 1 / R2;
		float t = r2*R2;
		float f = DecayDerivative(t);
		f *= R2 * 2;
		vtr *= f;
		f = DecayFunction(t);
		vtr = XMVectorSetW(vtr, f);
		return vtr;
	}
	else
	{
		return g_XMZero;
	}
}

MetaBallModel::AcceleratedContainer::AabbType getMetaballAabb(const Metaball& metaball)
{
	const Eigen::Vector3f& center = reinterpret_cast<const Eigen::Vector3f&>(metaball.Position);
	Eigen::Vector3f extent;
	extent.setConstant(metaball.Radius);
	return MetaBallModel::AcceleratedContainer::AabbType(center - extent, center + extent); // Eigen box
}

DirectX::BoundingBox getDxBox(const MetaBallModel::AcceleratedContainer::AabbType& ebox)
{
	DirectX::BoundingBox dbox;
	reinterpret_cast<Eigen::Vector3f&>(dbox.Center) = ebox.center();
	reinterpret_cast<Eigen::Vector3f&>(dbox.Extents) = ebox.max() - reinterpret_cast<Eigen::Vector3f&>(dbox.Center);
	return dbox;	
}

MetaBallModel::MetaBallModel(void)
	: Primitives(getMetaballAabb)
{
	//m_Polygonizer = nullptr;
	ISO = MODELING_ISO;
}

MetaBallModel::MetaBallModel(const PrimitveVectorType &primitives)
	: Primitives(getMetaballAabb)
{
	//m_Polygonizer = nullptr;
	Primitives.assign(primitives.begin(),primitives.end());
	ISO = MODELING_ISO;
	Update();
}

MetaBallModel::MetaBallModel(PrimitveVectorType &&primitives)
	: Primitives(getMetaballAabb)
{
	//m_Polygonizer = nullptr;
	ISO = MODELING_ISO;
	Primitives.assign(primitives.begin(), primitives.end());
	Update();
}

MetaBallModel& MetaBallModel::operator=(const MetaBallModel& rhs)
{
	Primitives = rhs.Primitives;
	//Connections = rhs.Connections;
	ISO = rhs.ISO;
	//BoundingBox = rhs.BoundingBox;
	//BoundingSphere = rhs.BoundingSphere;
	return *this;
}

MetaBallModel& MetaBallModel::operator=(MetaBallModel&& rhs)
{
	Primitives = std::move(rhs.Primitives);
	//Connections = std::move(rhs.Connections);
	ISO = rhs.ISO;
	//BoundingBox = rhs.BoundingBox;
	//BoundingSphere = rhs.BoundingSphere;
	return *this;
}


//
//void MetaBallModel::InitializePoygonizer(float Precise , unsigned int Boundry)
//{
//	m_Polygonizer = new Polygonizer::Polygonizer(this,Precise,Boundry);
//	if (!m_Polygonizer) 
//		throw "Can't Initialize polygonizer.";
//}


MetaBallModel::~MetaBallModel(void)
{
}

typedef MetaBallModel::AcceleratedContainer::AabbType AabbType;

//XM_ALIGNATTR
struct PointContainsOperator
{
	typedef AabbType::VectorType VectorType;
	VectorType pt;

	bool operator()(const Metaball& ball)
	{
		XMVECTOR dist = XMLoadA(ball.Position) - DirectX::XMLoadFloat3A(pt.data());
		dist = XMVector3LengthSq(dist);
		return XMVectorGetX(dist) < ball.Radius * ball.Radius;
	}

	bool operator()(const AabbType& aabb)
	{
		return aabb.contains(pt);
	}
};

//XM_ALIGNATTR
struct RayIntersectionOperator
{
	typedef AabbType::VectorType VectorType;
	Vector4 origin;
	Vector4 direction;

	bool operator()(const Metaball& ball)
	{
		float dist;
		return reinterpret_cast<const BoundingSphere&>(ball).Intersects(origin, direction, dist);
	}

	bool operator()(const AabbType& aabb)
	{
		float dist;
		DirectX::BoundingBox box = getDxBox(aabb);
		return box.Intersects(origin, direction, dist);
	}
};

float XM_CALLCONV MetaBallModel::eval(DirectX::FXMVECTOR vtr) const
{
	float sum  = - m_ISO;

	PointContainsOperator pred;
	XMStoreFloat3A(pred.pt.data(), vtr);

	auto range = BVFindAllIf(Primitives,pred);

	for (auto& ball : range)
	{
		sum += ball.eval(vtr);
	}

	//for (unsigned int i = 0; i < This.size(); i++)
	//	sum+=This[i].eval(vtr);

	return sum;
}

/// <summary>
/// Compute the negtive grad vector of the field.
/// </summary>
/// <param name="vtr">The VTR.</param>
/// <returns></returns>
XMVECTOR XM_CALLCONV MetaBallModel::grad(DirectX::FXMVECTOR vtr) const
{
	XMVECTOR sum = g_XMZero;

	PointContainsOperator pred;
	XMStoreFloat3A(pred.pt.data(), vtr);

	auto range = BVFindAllIf(Primitives, pred);

	for (auto& ball : range)
	{
		sum -= ball.grad(vtr);
	}

	return (Vector3)sum;
}

XMVECTOR XM_CALLCONV MetaBallModel::evalgrad(DirectX::FXMVECTOR vtr) const
{
	XMVECTOR sum = g_XMZero;

	PointContainsOperator pred;
	XMStoreFloat3A(pred.pt.data(), vtr);

	auto range = BVFindAllIf(Primitives, pred);

	for (auto& ball : range)
	{
		sum -= ball.evalgrad(vtr);
	}

	sum += g_XMIdentityR3 * m_ISO;
	sum *= g_XMNegateW;

	return sum;
}

// 	float MetaBallModel::EvalSphere(const Vector3 &SphereCentre,float Radius) const
// 	{
// 
// 		float sum = - 0.666666f;
// 		unsigned int n = This.size();
// 
// 		for (unsigned int i = 0; i <n; i++ )
// 		{
// 			float r = (This[i].Position - SphereCentre).Length();
// 			float r2;
// 			if (Radius>=r) r2 = 0;
// 			else r2 = (Radius-r)*(Radius-r);
// 			sum += R2Function(r2,This[i].Radius);
// 		}
// 		return sum ;
// 	}

//const bool MetaBallModel::FindLineSegmentIntersectionPointWithMesh(Vector3 &Output,FXMVECTOR LineEnd1,FXMVECTOR LineEnd2,float Precision) const
//{
//	XMVECTOR Start = LineEnd1,End = LineEnd2,Centre;
//	float Es , Ec , Ed;
//	int c=0;
//
//	//Binary search the surface point
//	Es = eval(Start);
//	Ed = eval(End);
//
//	if (!(Sgn(Es)^Sgn(Ed)))
//		return false;
//	c = 0;
//	do
//	{
//		Centre = 0.5*(Start + End);
//		Ec = eval(Centre);
//		if (Sgn(Ec)^Sgn(Es)) { 
//			End = Centre;
//			Ed = Ec;
//		}
//		else {
//			Start = Centre;
//			Es = Ec;
//		}
//	} while (abs(Ec)>Precision && ++c<30);
//
//	if (c>=30) {
//		return false;
//	}
//	Output = Centre;
//	return true;
//}

unsigned int Metaball::Intersects(_In_ FXMVECTOR Origin,_In_ FXMVECTOR Direction,_Out_ float* const d1/* = nullptr*/,_Out_ float* const d2/* = nullptr*/) const
{
	assert( DirectX::Internal::XMVector3IsUnit( Direction ) );

	XMVECTOR vCenter = XMLoadFloat3( &Position );
	XMVECTOR vRadius = XMVectorReplicatePtr( &Radius );

	// l is the vector from the ray origin to the center of the sphere.
	XMVECTOR l = vCenter - Origin;

	// s is the projection of the l onto the ray direction.
	XMVECTOR s = XMVector3Dot( l, Direction );

	XMVECTOR l2 = XMVector3Dot( l, l );

	XMVECTOR r2 = vRadius * vRadius;

	// m2 is squared distance from the center of the sphere to the projection.
	XMVECTOR m2 = l2 - s * s;

	XMVECTOR NoIntersection;

	// If the ray origin is outside the sphere and the center of the sphere is 
	// behind the ray origin there is no intersection.
	NoIntersection = XMVectorAndInt( XMVectorLess( s, XMVectorZero() ), XMVectorGreater( l2, r2 ) );

	// If the squared distance from the center of the sphere to the projection
	// is greater than the radius squared the ray will miss the sphere.
	NoIntersection = XMVectorOrInt( NoIntersection, XMVectorGreater( m2, r2 ) );


	if( XMVector4NotEqualInt( NoIntersection, XMVectorTrueInt() ) )
	{
		// The ray hits the sphere, compute the nearest intersection point.
		XMVECTOR q = XMVectorSqrt( r2 - m2 );
		XMVECTOR t1 = s - q;
		XMVECTOR t2 = s + q;

		XMVECTOR OriginInside = XMVectorLessOrEqual( l2, r2 );
		//XMVECTOR t = XMVectorSelect( t1, t2, OriginInside );
		// Store the x-component to *pDist.
		XMStoreFloat( d1, t1 );

		//t = XMVectorSelect( t2, g_XMZero,OriginInside);
		XMStoreFloat( d2, t2 );

		return XMVectorGetIntX(OriginInside) ? 1 : 2;
	}

	*d1 = *d2 = 0.f;
	return 0;
}

//Geometrics::BezierClip make_clip();

bool MetaBallModel::RayIntersection(DirectX::Vector3 &Output,DirectX::FXMVECTOR Origin,DirectX::FXMVECTOR Direction, float Precision/*=0.001f*/) const
{
	float distances[2];
	typedef std::pair<unsigned int,float> InteresctUint;
	std::vector<InteresctUint> Intersections;
	std::set<size_t> EffectiveSet;
	// A table for storage the the intersection distance for each intersected sphere
	//std::unordered_map<size_t,std::pair<float,float>> SphereMap(Primitives.size() / 10 + 1);
	std::map<size_t,std::pair<float,float>> SphereMap;
	//std::vector<std::pair<float,float>> SphereMap(this->size());

	XMVECTOR vDir= XMVector3Normalize(Direction);

	// Aabb Tree Ray Intersection test to cull un-intersected volums
	auto intersected = BVFindAllIf(Primitives, RayIntersectionOperator{Origin,vDir});

	for (auto itr = intersected.begin(); itr != intersected.end(); ++itr)
	{
		int i = itr.getIndex();
		const auto& sphere = Primitives[i];
		int count = sphere.Intersects(Origin,vDir,distances,distances+1);

		if (count > 0) 
		{
			if (count == 1) // if count == 1, than origin is inside this sphere, and distance[0] is in negative direction
				EffectiveSet.insert(i);
			else
				Intersections.emplace_back(i,distances[0]);
			Intersections.emplace_back(i,distances[1]);
			SphereMap[i] = std::make_pair(distances[0],distances[1]);
		}
	}

	std::sort(Intersections.begin(),Intersections.end(),
		[](const InteresctUint& lhs, const InteresctUint& rhs){ 
			return lhs.second < rhs.second;
	});

	Bezier::BezierClipping<float,6> IntervalClipping;

	const float EfficeRatio = this->EffictiveRadiusRatio();

	float start = 0.0f , end = 0.0f;
	for (auto itr = Intersections.begin(); itr != Intersections.end(); itr++)
	{
		end = itr->second;

		if (end > start) // For the case (end == begin) 
		{
			if (EffectiveSet.size() == 1) // Single metaball form
			{
				float d1,d2;
				Metaball isoSphere(Primitives[*EffectiveSet.cbegin()]);
				isoSphere.Radius *= EfficeRatio;
				auto count = isoSphere.Intersects(Origin,vDir,&d1,&d2);
				if (count > 0)
				{
					if (d1 >= start && d1 <= end)
					{
						Output = d1 * vDir + Origin;
						float eval = this->eval(Output);
						assert(eval<0.01f);
						return true;
					}
					if (d2 >= start && d2 <= end)
					{
						Output = d2 * vDir + Origin;
						float eval = this->eval(Output);
						assert(eval<0.01f);
						return true;
					}
				}
			} else if (EffectiveSet.size() > 1) // Multi metaball form
			{
				IntervalClipping.fill(0.0f);
				for (auto&& idx : EffectiveSet)
				{
					const auto& sphere = SphereMap[idx];
					float d = sphere.second - sphere.first;
					float Delta = d * 0.5f;
					Delta *= Delta;
					auto clipping = Primitives[idx].GetBezierFunctionInLine(Delta);
					float s = (start - sphere.first)/d;
					float t = (end - sphere.first)/d;
					clipping.crop(s,t);
					IntervalClipping.compound(clipping);
				}
#ifdef _DEBUG
				float root = Bezier::solove_first_root(IntervalClipping,m_ISO,Precision*0.3333f/(end-start));
#else
				float root = Bezier::solove_first_root(IntervalClipping,m_ISO,Precision/(end-start));
#endif
				if (root >= 0.0f && root <= 1.0f)
				{
					root *= end-start;
					root += start;
					Output = root * vDir + Origin;

					float eval = this->eval(Output);
					assert(eval<0.01f);

					return true;
				}

			}
		}

		auto pos = EffectiveSet.find(itr->first);
		if (pos == EffectiveSet.end())
			EffectiveSet.insert(itr->first);
		else
			EffectiveSet.erase(pos);

		start = end;
	}

	return false;
}

bool Metaball::Connected(const Metaball& lhs, const Metaball& rhs , float ISO)
{
	//static std::array<float,7> clip = {0.0f,0.0f,16.0f/27.0f,8.0f/45.0f*13.0f,16.0f/27.0f,0.0f,0.0f};
	XMVECTOR vS0 = lhs.Position;
	XMVECTOR vS1 = rhs.Position;
	float dis = Vector3::Distance(vS0,vS1);
	if (dis >= lhs.Radius + rhs.Radius) return false;
	ClipType lClip = lhs.GetBezierFunctionInLine(lhs.Radius*lhs.Radius);
	ClipType rClip = lClip;

	float st = ::std::max(0.0f,dis-rhs.Radius);
	float ed = ::std::min(dis,lhs.Radius);

	float s0 = ::std::min(0.5f*st/lhs.Radius + 0.5f,1.0f);
	float e0 = ::std::min(0.5f*ed/lhs.Radius + 0.5f,1.0f);
	lClip.crop(s0,e0);

	float s1 = 1 - ::std::min(0.5f*(dis - st)/rhs.Radius + 0.5f,1.0f);
	float e1 = 1 - ::std::min(0.5f*(dis - ed)/rhs.Radius + 0.5f,1.0f);
	rClip.crop(s1,e1);

	lClip.compound(rClip);
	if (lClip.front() < ISO ||  lClip.back() < ISO) 
		return false;
	else 
	{
		return !Bezier::if_have_root(lClip,ISO);
	}

}

float Metaball::ConnectionStrength(const Metaball& lhs, const Metaball& rhs , float ISO)
{
	XMVECTOR vS0 = lhs.Position;
	XMVECTOR vS1 = rhs.Position;
	float dis = Vector3::Distance(vS0,vS1);
	if (dis >= lhs.Radius + rhs.Radius) return false;
	ClipType lClip = lhs.GetBezierFunctionInLine(lhs.Radius*lhs.Radius);
	ClipType rClip = lClip;

	float st = ::std::max(0.0f,dis-rhs.Radius);
	float ed = ::std::min(dis,lhs.Radius);

	float s0 = ::std::min(0.5f*st/lhs.Radius + 0.5f,1.0f);
	float e0 = ::std::min(0.5f*ed/lhs.Radius + 0.5f,1.0f);
	lClip.crop(s0,e0);

	float s1 = 1 - ::std::min(0.5f*(dis - st)/rhs.Radius + 0.5f,1.0f);
	float e1 = 1 - ::std::min(0.5f*(dis - ed)/rhs.Radius + 0.5f,1.0f);
	rClip.crop(s1,e1);

	lClip.compound(rClip);

	//float value = std::numeric_limits<float>::infinity();
	//for (float u = 0.0f; u < 1.0f; u+=0.01f)
	//{
	//	float v = lClip(u);
	//	if (v < value) value = v;
	//}
	float value = Bezier::min_value(lClip,0.001f);
	return std::max(value - ISO,0.0f);
	//return lhs.eval(rhs.Position) + rhs.eval(lhs.Position);
}




//bool MetaBallModel::FindRayIntersectionPointWithMesh(Vector3 &Output,FXMVECTOR RaySource,FXMVECTOR RayDirection,float MaxDistance , float Precision) const{
//	XMVECTOR End;
//	float Es = eval(RaySource) , Ed;
//
//	End = RaySource + RayDirection;
//	Ed = eval(End);
//
//	if (Sgn(eval(End))^Sgn(Es))
//	{
//		return FindLineSegmentIntersectionPointWithMesh(Output,RaySource,End,Precision);
//	}
//
//	XMVECTOR Dirction = Vector3::Normalize(RayDirection);
//	//Step is set to 0.03, but seems too big?
//	Dirction *= 0.03f;
//
//	XMVECTOR Start = RaySource;
//	End = RaySource + Dirction;
//
//	int c=0;
//	int attemptLimit = static_cast<int>(MaxDistance / 0.03f) + 1;
//	//Find a point inside and a point outside
//	while ((++c<attemptLimit) && !(Sgn(eval(End))^Sgn(Es))){
//		Start = End;
//		End += Dirction;
//		//Dirction *= 1.25; // Get the step bigger and bigger
//	}
//
//	if (c>=attemptLimit) {
//		return false;
//	}
//
//	return FindLineSegmentIntersectionPointWithMesh(Output,Start,End,Precision);
//}

bool MetaBallModel::Contains(const DirectX::FXMVECTOR p) const
{
	return eval(p)>0;
}

unsigned int MetaBallModel::FindClosestMetballindex(DirectX::FXMVECTOR vPoint) const
{
	unsigned int Index = -1;
	float MinDis = 1e6;
	const float EffectiveRatio = EffictiveRadiusRatio();
	//std::min_element(Primitives.cbegin(),Primitives.cend(),
	for (unsigned int i = 0; i < This.size(); i++)
	{
		float Dis = Vector3::Distance(vPoint,This[i].Position);
		Dis -= This[i].Radius * EffectiveRatio; 
		if (Dis < MinDis){ 
			MinDis = Dis;
			Index = i;
		}
	}
	return Index;
}

DirectX::XMVECTOR MetaBallModel::FindClosestSurfacePoint(DirectX::FXMVECTOR vPoint) const
{

	int Index = FindClosestMetballindex(vPoint);

	Vector3 Sp;
	bool Hr;
	//if (Contains(vPoint))
	//{
	//	XMVECTOR vDirection = vPoint - This[Index].Position;
	//	vDirection = XMVector3Normalize(vDirection);
	//	Hr = RayIntersection(Sp,This[Index].Position,vDirection);
	//} else {
	//	Hr = FindLineSegmentIntersectionPointWithMesh(Sp,This[Index].Position,vPoint);
	//}

	XMVECTOR vDirection = vPoint - (XMVECTOR)This[Index].Position;
	vDirection = XMVector3Normalize(vDirection);
	Hr = RayIntersection(Sp,This[Index].Position,vDirection);

	if (!Hr) return g_XMQNaN;

	Vector3 MinP = Sp;
	float MinDis = Vector3::Distance(Sp,vPoint);

	for (int i = 0; i < 10; i++)
	{
		XMVECTOR vP = Sp;
		XMVECTOR vN = this->grad(vP);
		vP -= vPoint;
		vP = XMVector3Normalize(vP);
		vN = -vN;
		vN = XMVector3Normalize(vN);
		vP = (vP - vN) * 0.2f;
		XMVECTOR vD = vN;
		for (int j = 0; j < 5; j++)
		{
			//bool hr = FindRayIntersectionPointWithMesh(Sp,vPoint,vD);
			bool hr = RayIntersection(Sp,vPoint,vD);
			if (hr)
			{
				float Dis = Vector3::Distance(Sp,vPoint);
				if (Dis < MinDis)
				{
					MinP = Sp;
					MinDis = Dis;
				}
			}
		}
	}
	return MinP;
}
DirectX::XMVECTOR MetaBallModel::FindClosestSurfacePoint2(DirectX::FXMVECTOR vPoint) const
{
	const float precise = 0.01f;
	const unsigned int maxStep = 30;
	XMVECTOR vPos = vPoint;
	float field = this->eval(vPos);

	unsigned int step = 0;
	while (abs(field) > precise && step < maxStep )
	{
		XMVECTOR vnGrad = this->grad(vPoint);
		XMVECTOR vLen = XMVector3Length(vnGrad);
		if (XMVector4NearEqual(vLen,g_XMZero,g_XMEpsilon*10))
		{
			step = maxStep;
			break;
		}
		vLen = XMVectorReplicate(field) / vLen;
		vnGrad *= vLen;
		vnGrad *= 0.1f;
		vPos += vnGrad;
		field = this->eval(vPos);
		++step;
	}

	if (step >= maxStep)
	{
		return g_XMQNaN;
		//std::cout<<"Error  : Can't find closest point for "<<Vector3(vPoint)<<", field value = "<<field<<std::endl;
	}

	return vPos;
}


// 	const bool MetaBallModel::IsSphereIntersectMesh(const Vector3 &SphereCentre,float Radius) const
// 	{
// 		return EvalSphere(SphereCentre,Radius)>0;
// 	}

void MetaBallModel::OptimizeConnection( unsigned int BlockIndex )
{
	if (size()<=1) return;
	std::vector<bool> Arrived(this->size(),false);
	std::vector<bool> deleted_flags(this->size(),false);
	Travel(BlockIndex,Arrived,deleted_flags);
	//std::vector<Metaball> buffer;
	unsigned int k = 0;
	auto& prims = Primitives.getObjectList();
	prims.erase(std::remove_if(prims.begin(), prims.end(),[&Arrived,&k](const Metaball& ball)->bool{
		return !Arrived[k++];
	}), prims.end());

	Update();
}

size_t MetaBallModel::remove_if(const std::vector<bool>& remove_flags)
{
	if (size()<=1) return 0;
	if (remove_flags.size()!= Primitives.size()) return -1;
	auto itr = remove_flags.begin();
	for (; itr != remove_flags.end(); ++itr)
	{
		if (!*itr) break;
	}
	if (itr == remove_flags.end())
	{
			clear();
			return 0;
	}

	auto beginIndex = itr - remove_flags.begin();

	std::vector<bool> Arrived(Primitives.size(),false);

	Travel(beginIndex,Arrived,remove_flags);
	//std::vector<Metaball> buffer;
	unsigned int k = 0;

	auto& prims = Primitives.getObjectList();

	prims.erase(std::remove_if(prims.begin(), prims.end(),[&Arrived,&k](const Metaball& ball)->bool{
		return !Arrived[k++];
	}), prims.end());

	Update();

	return size();
}

std::vector<bool> MetaBallModel::flood_fill(size_t origin,const std::vector<bool>& deleted_flags)
{
	if (size()<=1) return std::vector<bool>();
	if (deleted_flags.size()!= Primitives.size()) 
		throw std::logic_error("falgs size not match.");

	std::vector<bool> Arrived(Primitives.size(),false);
	Travel(origin,Arrived,deleted_flags);
	return Arrived;
}




void MetaBallModel::Travel( unsigned int index , std::vector<bool>& Arrived , const std::vector<bool>& remove_flags) const
{
#ifdef _DEBUG
	if (index<0 || index>=this->size()) 
		throw std::exception("index over range.");
#endif
	Arrived[index]=true;
	for (unsigned int i = 0; i < this->size(); i++)
	{
		if (!Arrived[i] && !remove_flags[i] && IsTwoMetaballIntersect(This[index],This[i]))
			Travel(i,Arrived,remove_flags);
	}
}

bool MetaBallModel::IsTwoMetaballIntersect(const Metaball& lhs,const Metaball& rhs ) const
{
	//	float d = Vector3::Distance(A->Position , B->Position);
	//	float r1 = A->Radius,r2 = B->Radius;
	//	if (d < r1 + r2) return true;
	//	float x = (r1*r2*r2 + r1*r1*(d-r2)) / (r1*r1 + r2*r2); // x is the point minimum f among the intersection line
	//	float val = -MODELING_ISO;
	//	//	val += RFunction(x,r1) + RFunction(d-x,r2);
	//	float t1 = x/r1 , t2 = (d-x)/r2;
	//	t1 *= t1; t2 *= t2;
	//	val += Metaball::DecayFunction(t1) + Metaball::DecayFunction(t2);
	//
	//	return val > 0.0f;
	return Metaball::Connected(lhs,rhs,m_ISO);
}

DirectX::BoundingBox MetaBallModel::GetBoundingBox() const
{
	DirectX::BoundingBox boundingBox;
	boundingBox = getDxBox(Primitives.getVolume(Primitives.getRootIndex()));
	return boundingBox;
}

void MetaBallModel::Update()
{
	if (!Primitives.empty())
	{
		UpdatePrimtives();
	}
	else
	{
		Primitives.clear();
	}
}




namespace Geometrics
{
#ifdef LAPLACIAN_INTERFACE
	void CreateConnectionGraph(const MetaBallModel& volume, _Out_ ConnectionGraph& Graph) const
	{
		ConnectionGraph g(Primitives.size());
		for (size_t i = 0; i < Primitives.size(); i++)
		{
			for (size_t j = i + 1; j < Primitives.size(); j++)
			{
				if (IsTwoMetaballIntersect(Primitives[i], Primitives[j]))
				{
					float w = Metaball::ConnectionStrength(Primitives[i], Primitives[j], m_ISO);
					boost::add_edge(i, j, w, g);
				}
			}
		}
		Graph = g;
	}

	Eigen::SparseMatrix<float> CreateLaplacianMatrix(const MetaBallModel &Volume,const std::vector<float> &OuterWeights)
	{
		using namespace boost;
		const auto N = Volume.size();

		std::vector<Eigen::Triplet<float>> triplets;
		std::vector<float> weights_sum;
		if (OuterWeights.empty())
			weights_sum.assign(N,0.0f);
		else
			weights_sum = OuterWeights;
		assert(weights_sum.size() == N);

		graph_traits<ConnectionGraph>::edge_iterator eitr,eend;
		ConnectionGraph g_volume;
		Volume.CreateConnectionGraph(g_volume);
		const auto& g_w = get(edge_weight, g_volume);

		for (std::tie(eitr,eend) = edges(g_volume); eitr != eend; eitr++)
		{
			int x = source(*eitr,g_volume) , y = target(*eitr,g_volume);
			float w = g_w[*eitr];

			triplets.emplace_back(x,y,-w);
			triplets.emplace_back(y,x,-w);
			weights_sum[x] += w;
			weights_sum[y] += w;
		}

		for (size_t i = 0; i < N; i++)
		{
			assert(weights_sum[i] != .0f);
			triplets.emplace_back(i,i,weights_sum[i]);
		}

		Eigen::SparseMatrix<float> Laplace(N,N);
		Laplace.setFromTriplets(triplets.begin(),triplets.end());
		return Laplace;
	}
#endif

}