#pragma once

// Prevent Windows header's macro pollution
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <DirectXMathExtend.h>
#include "Polygonizer.h"
#include <DirectXCollision.h>
#include <vector>
#include <array>
#include "BezierClip.h"
#include "KdBVH.h"

#ifdef LAPLACIAN_INTERFACE
#include <Eigen\Dense>
#include <Eigen\Sparse>
#include <boost\graph\adjacency_list.hpp>
#endif

namespace Geometrics{
	class Primitive
	{
	public:
		virtual float distance(DirectX::FXMVECTOR Ptr) const = 0;
		virtual DirectX::XMVECTOR diplacement(DirectX::FXMVECTOR Ptr) const = 0;
	};

	struct PolynomialKernel_6
	{
		typedef Bezier::BezierClipping<float,6> SegmentFunctionType;
		/// <summary>
		/// Evaluate the filed function , where
		/// t = r2/R2 , t belongs [0,1] ,
		/// r is the point-center distance , and R is the radius of the metaball
		/// </summary>
		static float DecayFunction(float t);
		/// <summary>
		/// Compute the fild function's derivative , respect to t.
		/// where t = r2/R2 , belongs [0,1]
		/// r is the point-center distance , and R is the radius of the metaball
		/// </summary>
		static float DecayDerivative(float t);

		/// <summary>
		/// return the function define in Segement respect paramenter space t.
		/// </summary>
		/// <param name="a">a = sin^2(a).</param>
		/// <returns></returns>
		static SegmentFunctionType SegementFunction(float a);
	};

	XM_ALIGNATTR
	struct Metaball : public DirectX::AlignedNew<DirectX::XMVECTOR>
	{
	public:
		typedef Bezier::BezierClipping<float,6> ClipType;
	public:
		Metaball() = default;
		Metaball(DirectX::FXMVECTOR _Position,float _Radius,unsigned int _BindingIndex = 0);
		Metaball(const DirectX::Vector3 &_Position,float _Radius,unsigned int _BindingIndex = 0);
		Metaball(const DirectX::Vector4 & Sphere,unsigned int _BindingIndex = 0);

		/// <summary>
		/// Evaluate the filed function , where
		/// t = r2/R2 , t belongs [0,1] ,
		/// r is the point-center distance , and R is the radius of the metaball
		/// </summary>
		static float DecayFunction(float t);
		/// <summary>
		/// Compute the fild function's derivative , respect to t.
		/// where t = r2/R2 , belongs [0,1]
		/// r is the point-center distance , and R is the radius of the metaball
		/// </summary>
		static float DecayDerivative(float t);

		// A Newton method to evalate the ratio (single sphere's iso surface radius)/(Effictive ratio)
		static float EffectiveRadiusRatio(float ISO , float precise = 1e-6);
		static bool  Connected(const Metaball& lhs, const Metaball& rhs , float ISO);
		static float ConnectionStrength(const Metaball& lhs, const Metaball& rhs , float ISO);
	public:
		float XM_CALLCONV eval(DirectX::FXMVECTOR pos) const;
		DirectX::XMVECTOR XM_CALLCONV grad(DirectX::FXMVECTOR pos) const;
		// return gradiant and value at same time, retval.w == eval
		DirectX::XMVECTOR XM_CALLCONV evalgrad(DirectX::FXMVECTOR pos) const;

		inline ClipType GetBezierFunctionInLine(float Delta) const
		{
			float a = Delta/(Radius*Radius);
			float a2 = a*a;

			ClipType Clipping;
			Clipping[0]=Clipping[1]=Clipping[5]=Clipping[6]=0.0f;
			Clipping[2] = Clipping[4] = 16.0f/27.0f*a2;
			Clipping[3] = 8.0f/45.0f*(8*a+5)*a2;
			return Clipping;
		}

		/// <summary>
		/// Return the Ray intersection times to the bounding sphere of this metaball the specified origin.
		/// If return 0 , means there is no interesection point at all , d1 and d2 == 0.0f
		/// If return 1 , means there is only 1 intersection point , the distance storage in d2 (d1 will be negtive).
		/// If return 2 , d1 and d2 will be the interesection points' distance , and d1 < d2
		/// </summary>
		/// <param name="Origin">The ray origin.</param>
		/// <param name="Direction">The ray direction.</param>
		/// <param name="d1">Out : The distance to first intersection point.</param>
		/// <param name="d2">Out : The distance to second intersection point.</param>
		/// <returns></returns>
		unsigned int Intersects(_In_ DirectX::FXMVECTOR Origin,_In_ DirectX::FXMVECTOR Direction,_Out_ float* const d1 = nullptr,_Out_ float* const d2 = nullptr) const;
		//unsigned int ISOIntersects(_In_ float ISO ,_In_ DirectX::FXMVECTOR Origin,_In_ DirectX::FXMVECTOR Direction,_Out_ float* const d1 = nullptr,_Out_ float* const d2 = nullptr) const;
	public:
		XM_ALIGNATTR
		DirectX::Vector3 Position;
		float Radius;

		// that's a optional data for extend the capability of our model
		//unsigned int BindingIndex;
		//DirectX::SimpleMath::Color Color;
		//std::vector<float>	BlendWeights;
		//DirectX::XMUINT4 BlendIndex;
		//DirectX::XMFLOAT4 BlendWeights;
		//float Coefficient;
	};

	class MetaBallModel 
		: public Polygonizer::ImplicitFunction 
	{
	public:

		typedef std::vector<Metaball, DirectX::XMAllocator> PrimitveVectorType;
		MetaBallModel(void);
		explicit MetaBallModel(const PrimitveVectorType &primitives);
		explicit MetaBallModel(PrimitveVectorType &&primitives);
		MetaBallModel& operator=(const MetaBallModel& rhs);
		MetaBallModel& operator=(MetaBallModel&& rhs);
		virtual ~MetaBallModel(void);
	public:
		// Methods for polygonlizer to use
		float XM_CALLCONV eval(DirectX::FXMVECTOR vtr) const override;
		DirectX::XMVECTOR XM_CALLCONV grad(DirectX::FXMVECTOR vtr) const override;
		DirectX::XMVECTOR XM_CALLCONV evalgrad(DirectX::FXMVECTOR pos) const;

		float GetISO() const{
			return m_ISO;
		}

		void SetISO(float _Iso)
		{
			m_ISO = _Iso;
			m_EffectiveRatio = Metaball::EffectiveRadiusRatio(m_ISO);
		}

		__declspec(property(get = GetISO , put = SetISO))
		float ISO;

	public:
		// Using this method to tessellates the implicit function defined surface into a mesh
		// To use this , your vertex must have the member "float3 position" & "float3 normal"
		template <typename _Tvertex, typename _TIndex>
		void Triangulize(std::vector<_Tvertex> &Vertices,std::vector<_TIndex> &Indices,float precise);

		inline void UpdatePrimtives() {
			Primitives.rebuild();
		}

		// Travel form the main index to delete all not connected metaballs
		// Optimize the structure of metaballs
		void OptimizeConnection(unsigned int BlockIndex);
		size_t remove_if(const std::vector<bool>& deleted_flags);
		std::vector<bool> flood_fill(size_t origin,const std::vector<bool>& deleted_flags);

#pragma region Vector Interface
	public:
		inline Metaball& operator[](unsigned int index) { return Primitives[index]; }
		inline const Metaball& operator[](unsigned int index) const { return Primitives[index]; }
		inline Metaball& at(unsigned int index) { return Primitives.at(index); }
		inline const Metaball& at(unsigned int index) const { return Primitives.at(index); }
		inline size_t size() const {return Primitives.size();}
		inline void clear() {Primitives.clear();}
		inline bool empty() const {return Primitives.empty();}
		inline void push_back(const Metaball &element) { Primitives.push_back(element); }
		inline std::vector<Metaball>::iterator begin() { return Primitives.begin(); }
		inline std::vector<Metaball>::iterator end() { return Primitives.end(); }
		inline std::vector<Metaball>::const_iterator cbegin() const { return Primitives.cbegin(); }
		inline std::vector<Metaball>::const_iterator cend() const { return Primitives.cend(); }
		inline Metaball& back() {return Primitives.back(); }
#pragma endregion

	protected:
		void Travel(unsigned int index , std::vector<bool>& Arrived , const std::vector<bool>& remove_flags) const;
		//void InitializePoygonizer(float Precise , unsigned int Boundry);
	public:
		//void operator= (const std::vector<Metaball>& rhs);
		// Geometry Methods Set for Editing
		//float EvalSphere(const DirectX::Vector3 &SphereCentre,float Radius) const;

		bool Contains(const DirectX::FXMVECTOR p) const;
		//const bool IsSphereIntersectMesh(const DirectX::Vector3 &SphereCentre,float Radius) const;

		// This function return the most closest intersection point to the ray-source.
		// It's a stupid obstacle , and foolish ray-marching algorithm , never use it!
		//bool FindRayIntersectionPointWithMesh(DirectX::Vector3 &Output,DirectX::FXMVECTOR RaySource,DirectX::FXMVECTOR RayDirection,float MaxDistance = 3.0f, float Precision=0.001) const;

		// Find the first intersection point from a ray to this model.
		// it's stable and correct implemented
		bool RayIntersection(DirectX::Vector3 &Output,DirectX::FXMVECTOR Origin,DirectX::FXMVECTOR Direction, float Precision=0.001f) const;

		//This function return the intersection point of a line segment with mesh define by this class
		//But whatever , this function only use the basic binary search , as long as there is multiply intersection point , this function may not work...
		//const bool FindLineSegmentIntersectionPointWithMesh(DirectX::Vector3 &Output,DirectX::FXMVECTOR LineEnd1,DirectX::FXMVECTOR LineEnd2 ,float Precision=0.001) const;

		DirectX::XMVECTOR FindClosestSurfacePoint(DirectX::FXMVECTOR vPoint) const;
		DirectX::XMVECTOR FindClosestSurfacePoint2(DirectX::FXMVECTOR vPoint) const;
		unsigned int FindClosestMetballindex(DirectX::FXMVECTOR vPoint) const;

	public:
		DirectX::BoundingBox GetBoundingBox() const;

		void Update();
	public:
		// This function returns the connection judgment if there is only A & B in space
		bool IsTwoMetaballIntersect(const Metaball& lhs, const Metaball& rhs) const;
		float EffictiveRadiusRatio() const { return m_EffectiveRatio; };

	public:
		typedef KdAabbTree<float, 3, Metaball> AcceleratedContainer;

		AcceleratedContainer		Primitives;

		//PrimitveVectorType		Primitives;
		//ConnectionGraph			Connections;
	private:
		//Polygonizer::Polygonizer *m_Polygonizer;
		float					m_ISO;
		float					m_EffectiveRatio;
	};


	//This method is EXTEMELY COSTLY. pay attention.
	template <typename _Tvertex, typename _TIndex>
	void MetaBallModel::Triangulize(std::vector<_Tvertex> &Vertices,std::vector<_TIndex> &Indices,float precise)
	{
		if (this->size() == 0) 
			return;
		auto bounds = 2 * std::max(BoundingBox.Extents.x,std::max(BoundingBox.Extents.y,BoundingBox.Extents.z));
		bounds /= precise;
		Polygonizer::Polygonizer polygonizer(this,precise,static_cast<int>(bounds)+1);
		// Core statement
		DirectX::Vector3 SurfaceP;
		bool hr = RayIntersection(SurfaceP,Primitives[0].Position,g_XMNegIdentityR2);
		if (!hr) return;
		polygonizer.march(false,SurfaceP.x,SurfaceP.y,SurfaceP.z);
		//polygonizer.march(false,(*this)[0].Position.x + (*this)[0].Radius,(*this)[0].Position.y,(*this)[0].Position.z);

		//Output the data
		Vertices.resize(polygonizer.no_vertices());
		Indices.resize(polygonizer.no_triangles()*3);

		for (int i = 0; i < polygonizer.no_vertices(); i++)
		{
			Vertices[i].position = polygonizer.get_vertex(i);
			Vertices[i].normal = polygonizer.get_normal(i);
		}
		for (int i = 0; i < polygonizer.no_triangles(); i++)
		{
			Indices[i*3 + 0] = polygonizer.get_triangle(i).v2;
			Indices[i*3 + 1] = polygonizer.get_triangle(i).v1;
			Indices[i*3 + 2] = polygonizer.get_triangle(i).v0;
		}

		//const DirectX::XMFLOAT3* points = &polygonizer.get_VerticesList()[0];
		//DirectX::BoundingBox::CreateFromPoints(BoundingBox,Vertices.size(),points,sizeof(DirectX::XMFLOAT3));
		//DirectX::BoundingSphere::CreateFromPoints(BoundingSphere,Vertices.size(),points,sizeof(DirectX::XMFLOAT3));
	}

#ifdef LAPLACIAN_INTERFACE
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, float>> ConnectionGraph;

	void CreateConnectionGraph(const MetaBallModel &Volume, _Out_ ConnectionGraph& Graph) const;

	// helper function for creating Laplace matrix for a given metaball graph
	// the OuterWeights parameter is optional , to specify a all zero on ,
	Eigen::SparseMatrix<float> CreateLaplacianMatrix(const MetaBallModel &Volume,const std::vector<float> &OuterWeights);
	inline Eigen::SparseMatrix<float> CreateLaplacianMatrix(const MetaBallModel &Volume)
	{
		return CreateLaplacianMatrix(Volume,std::vector<float>());
	}
#endif

} // Namespace

#ifdef PARALLEL_UPDATE
#undef PARALLEL_UPDATE
#endif