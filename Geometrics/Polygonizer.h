/**********************************************************************

polygonizer.h

This is Jules Bloomenthal's implicit surface polygonizer from GRAPHICS 
GEMS IV. Bloomenthal's polygonizer is still used and the present code
is simply the original code morphed into C++.

J. Andreas Bærentzen 2003.

**********************************************************************/
#pragma once
#ifndef POLYGONIZER_H
#define POLYGONIZER_H

#include <vector>
#include "DirectXMathExtend.h"

namespace Polygonizer{

	enum ToTetraHedralize
		{
			TET = 0,  // use tetrahedral decomposition 
			NOTET = 1  // no tetrahedral decomposition  */
		};

	/** The implicit function class represents the implicit function we wish 
			to polygonize. Derive a class from this one and implement your 
			implicit primitive in the eval function. Eval takes x,y,z coordinates and
			returns a value. We assume that the surface is the zero level set 
			and that the negative values are outside. This an arbitrary choice
			which does not make the code less general. */
	class ImplicitFunction
	{
	public:
	  //virtual float eval(float,float,float) const = 0;
	  //virtual DirectX::Vector3 grad(const DirectX::Vector3 &p) const = 0;
	  virtual float XM_CALLCONV eval(DirectX::FXMVECTOR p) const = 0;
	  virtual DirectX::XMVECTOR XM_CALLCONV grad(DirectX::FXMVECTOR p) const = 0;
	};

	typedef DirectX::Vector3 Point3D;
	typedef DirectX::Vector4 Point4D;


	//struct POINT { float x, y, z;	};

	/*struct Point3D
	{
		float x, y, z;

		Point3D(){}
		Point3D(float _x, float _y, float _z):x(_x), y(_y),z(_z){}

	};*/

	struct Point3DINT
	{
		int x, y, z;
		Point3DINT(int _x, int _y, int _z):x(_x), y(_y), z(_z){}
	};

	typedef Point3D VERTEX;
	typedef Point3D NORMAL;

	/** TRIANGLE struct contains the indices of the vertices comprising 
			the triangle */
	struct TRIANGLE
	{
	  int v0,v1,v2;
	  TRIANGLE () {}
	  TRIANGLE (int V0,int V1,int V2){
		  v0 = V0;
		  v1 = V1;
		  v2 = V2;
	  }
	};

	/** Polygonizer is the class used to perform polygonization.*/
	class Polygonizer
	{
	  std::vector<NORMAL> gnormals;  
	  std::vector<VERTEX> gvertices;  
	  std::vector<TRIANGLE> gtriangles;
  
	  ImplicitFunction* func;
	  float size;
	  int bounds;

	  //set up a vector to save all the cubes location
	  //currently, dont need to care how they get the cubes
	  std::vector<Point3D> gcubes;

	 public:	
	
		 //get an empty constructor
		 Polygonizer()
		 {

		 }

		/** Constructor of Polygonizer. The first argument is the ImplicitFunction
				that we wish to polygonize. The second argument is the size of the 
				polygonizing cell. The final arg. is the limit to how far away we will
				look for components of the implicit surface. */
	  Polygonizer(ImplicitFunction* _func, float _size, int _bounds):
	  func(_func), size(_size), bounds(_bounds) {}

		/** March erases the triangles gathered so far and builds a new 
				polygonization. The first argument indicates whether the primitive
				cell is a tetrahedron (true) or a cube (false). The final x,y,z 
				arguments indicate a point near the surface. */
	  void march(bool, float x, float y, float z);

		/** Return number of triangles generated after the polygonization.
				Call this function only when march has been called. */
	  int no_triangles() const
	  {
		return gtriangles.size();
	  }

		/** Return number of vertices generated after the polygonization.
				Call this function only when march has been called. */
	  int no_vertices() const
	  {
		return gvertices.size();
	  }
	
		/** Return number of normals generated after the polygonization.
				Of course the result of calling this function is the same as
				no_vertices.
				Call this function only when march has been called. */
	  int no_normals() const
	  {
		return gnormals.size();
	  }

	  //return the number of gcubes
	  int no_gcubes() const
	  {
		  return gcubes.size();
	  }
	
		/// Return triangle with index i. 
		TRIANGLE& get_triangle(int i) 
	  {
		return gtriangles[i];
	  }

		/// Return vertex with index i. 
		VERTEX& get_vertex(int i) 
	  {
		return gvertices[i];
	  }

		const std::vector<TRIANGLE>& get_TrianglesList() const
		{
			return gtriangles;
		}

		const std::vector<VERTEX>& get_VerticesList() const
		{
			return gvertices;
		}
	
		/// Return normal with index i. 
		NORMAL& get_normal(int i) 
	  {
		return gnormals[i];
	  }

		Point3D& get_gcubes(int i)
		{
			return gcubes[i];
		}

	};
}

#endif
