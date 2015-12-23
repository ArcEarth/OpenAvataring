#pragma once
#include "SceneObject.h"
#include "StepTimer.h"
#include "RenderSystemDecl.h"
#include <Textures.h>
#include <mutex>
#include <vector>

namespace DirectX
{
	class StepTimer;
}

namespace Causality
{
	class Frame;
	class ICamera;
	class IVisual;
	class ILight;
	class AssetDictionary;

	using DirectX::StepTimer;

	interface IScene abstract
	{
	public:
		virtual ~IScene() {};

		virtual void Update() = 0;
		virtual void Render(IRenderContext* context) = 0;
		virtual void Load() = 0;
		virtual void Release() = 0;

		virtual bool IsLoading() const = 0;
		virtual bool IsReleasing() const = 0;

		virtual bool IsLoaded() const = 0;

		virtual void OnNavigatedTo() = 0;
		virtual void OnNavigatedFrom() = 0;

		virtual SceneObject* Content() = 0;
	};

	enum SceneTimeLineType
	{
		PhysicalTimeInterval, // The time of scene is binded to physical time
		FixedTimeInterval,	  // The time of scene is binded to frame count, every two adjacant frame have same time delta
	};

	using std::vector;
	// A Scene is a collection of it's contents, interactive logic, 
	class Scene : public IScene
	{
	public:
		Scene();
		~Scene();

		static uptr<Scene> LoadSceneFromXML(const string& xml_file);

		void LoadFromFile(const string& xml_file);

		static Scene& GetSceneForCurrentView();

		// Inherit from IScene
		virtual void Update() override;
		virtual void Render(IRenderContext* context) override;
		virtual void Load() override;
		virtual void Release() override;

		virtual bool IsLoading() const override;
		virtual bool IsReleasing() const override;

		virtual bool IsLoaded() const override;

		virtual void OnNavigatedTo() override;
		virtual void OnNavigatedFrom() override;

		// Timeline operations
		bool	IsPaused() const { return is_paused; }
		void	Pause() { is_paused = true; }
		void	Resume() { is_paused = false; }

		time_seconds	GetLocalTime() const
		{
			return time_seconds(m_timer.GetTotalSeconds());
		}

		double	GetTimeScale() const { return time_scale; }
		void	SetTimeScale(double time_scale) { this->time_scale = time_scale; }

		// Contens operations
		SceneObject*	Content() override;
		SceneObject*	SetContent(SceneObject* sceneRoot);

		HUDCanvas*		GetHudCanvas();
		void			SetHudCanvas(HUDCanvas* canvas);

		AssetDictionary& Assets() { return *m_assets; }
		const AssetDictionary& Assets() const { return *m_assets; }

		const ParamArchive* GetSceneSettings() const { return m_settings; }

		ICamera *PrimaryCamera();

		bool SetAsPrimaryCamera(ICamera* camera);

		IRenderDevice*			GetRenderDevice() { return m_device.Get(); }
		IRenderContext*			GetRenderContext() { return m_context.Get(); }
		I2DContext*				Get2DContext() { return m_2dContext.Get(); }
		I2DFactory*				Get2DFactory() { return m_2dFactory.Get(); }
		ITextFactory*			GetTextFactory() { return m_textFactory.Get(); }

		void SetRenderDeviceAndContext(IRenderDevice* device, IRenderContext* context);
		void SetHudRenderDevice(I2DFactory* pD2dFactory,I2DContext* pD2dContext, ITextFactory* pTextFactory);

		RenderTarget&		Canvas() { return m_canvas; }
		const RenderTarget&	Canvas() const { return m_canvas; }
		void							SetCanvas(RenderTarget& canvas);

		vector<ICamera*>&				GetCameras() { return m_cameras; }
		vector<ILight*>&				GetLights() { return m_lights; }
		const vector<const ICamera*>&	GetCameras() const { return reinterpret_cast<const vector<const ICamera*>&>(m_cameras); }
		const vector<const ILight*>&	GetLights() const { return reinterpret_cast<const vector<const ILight*>&>(m_lights); }
		vector<IEffect*>&		GetEffects();

		void SetupEffectsViewProject(IEffect* pEffect, const DirectX::XMMATRIX &v, const DirectX::XMMATRIX &p);
		void SetupEffectsLights(IEffect* pEffect);

		void UpdateRenderViewCache();

		void SignalCameraCache();

		std::mutex&	ContentMutex();

	private:
		SceneTimeLineType			timeline_type;
		double						time_scale;
		StepTimer					m_timer;

		cptr<IRenderDevice>			m_device;
		cptr<IRenderContext>		m_context;
		cptr<I2DFactory>			m_2dFactory;
		cptr<ITextFactory>			m_textFactory;
		cptr<I2DContext>			m_2dContext;

		uptr<AssetDictionary>
									m_assets;

		uptr<SceneObject>			m_sceneRoot;
		uptr<HUDCanvas>				m_hudRoot;


		ParamArchive*				m_settings;
		uptr<ParamDocument>			m_sourceDoc;

		RenderTarget				m_canvas;
		//RenderableTexture2D			back_buffer;

		bool						is_paused;
		bool						is_loaded;
		int							loading_count;

		// Caches
		ICamera						*m_primCamera;
		vector<ICamera*>			m_cameras;
		vector<ILight*>				m_lights;
		vector<IVisual*>			m_renderables;
		vector<IEffect*>			m_effects;

		std::mutex					content_mutex;

		bool						camera_dirty;
		bool						object_dirty;
	};
}
