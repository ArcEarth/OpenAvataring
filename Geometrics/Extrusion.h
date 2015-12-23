#pragma once
#include <vector>
#include <cassert>
#include "csg.h"
#include "SpaceCurve.h"

namespace Geometrics
{
	using DirectX::Vector3;
	using DirectX::Ray;
	using DirectX::FXMVECTOR;

	using Curve = SpaceCurve;

	// A Patch is a closed curve on a surface
	// And the region within
	class SurfacePatch
	{
		MeshType*	m_surface;
		Curve		m_uvCurve;

		// the face id of each uv points
		std::vector<int>	
					m_fids;

		Curve		m_boundry;

		MeshType	m_mesh;
		
		int			m_dirty;

	public:
		MeshType& surface() { return *m_surface; }
		const MeshType& surface() const { return *m_surface; }
		void setSurface(MeshType* surface) { m_surface = surface; m_dirty = 1; };

		//Curve& boundry();
		const Curve& boundry() const;

		//Curve& uvBoundry();
		const Curve& uvBoundry() const { return m_uvCurve; }

		MeshType& mesh();
		const MeshType& mesh() const;

		void clear();
		bool append(XMVECTOR position, int fid);
		void closeLoop();

		// convert uv to positon in world space
		XMVECTOR unproject(XMVECTOR uv, int& fid);

		// unproject uv curve to the spatial curve
		void unprojectBoundry(int startFid);

		// seperate and re-triangluate the patch from exist surface
		MeshType& triangulate(size_t subdiv);
	};


	using std::vector;
	class Extrusion
	{
		SurfacePatch *m_top, *m_bottom;
		Curve *m_path;
		MeshType m_mesh;
		int		m_dirty;

	public:
		Extrusion();

		Extrusion(SurfacePatch *top, SurfacePatch *bottom, Curve *axis);

		~Extrusion();

		bool valiad() const { return m_top != nullptr && m_bottom != nullptr && m_path && !m_dirty; }

		MeshType& mesh();
		const MeshType& mesh() const;

		SurfacePatch& top() { return *m_top; }
		const SurfacePatch& top() const { return *m_top; }
		void setTop(SurfacePatch *path) { m_dirty = (m_top == path); m_top = path; }

		SurfacePatch& bottom() { return *m_bottom; }
		const SurfacePatch& bottom() const { return *m_bottom; }
		void setBottom(SurfacePatch *path) { m_dirty = (m_bottom == path); m_bottom = path; }

		Curve& axis() { return *m_path; }
		const Curve& axis() const { return *m_path; }
		void setAxis(Curve *axis) { m_dirty = (m_path == axis); m_path = axis; }

		// triangulate top and bottom with path into m_mesh
		MeshType& triangulate(int axisSubdiv, int polarSubdiv);

	};
}