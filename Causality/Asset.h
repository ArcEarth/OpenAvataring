#pragma once
#include <typeindex>
#include <unordered_map>
#include <functional>
#include <mutex>
#include "RenderSystemDecl.h"
#include "String.h"
#include "Serialization.h"
#include <ppltasks.h>

namespace Causality
{
	using path = std::wstring;
	using concurrency::task;
	using std::type_index;
	using std::unordered_map;

	class asset_dictionary;

	// asset control block
	class asset_control_base
	{
	private:
		friend asset_dictionary;
	public:
		ptrdiff_t					_ref_count;
		asset_dictionary*			_dictionary;
		path						_ref_path;
		bool						_loaded;
		task<void*>					_load_task;
		void						(*_destructor)(void*);

	public:
		asset_control_base()
			: _ref_count(0), _dictionary(nullptr), _loaded(false), _destructor(nullptr)
		{
		}

		template<class T>
		void _set_desctructor()
		{
			this->_destructor = reinterpret_cast<void(*)(void*)>(&T::~T);
		}

		asset_dictionary*			dictionary() { return _dictionary; }
		const asset_dictionary*		dictionary() const { return _dictionary; }

		const path&					source_path() const { return _ref_path; }
		bool						is_loaded() { return _loaded; }

		ptrdiff_t					ref_count() const { return _ref_count; }

		ptrdiff_t					_inc_ref() { return ++_ref_count; }
		ptrdiff_t					_dec_ref() { return --_ref_count; }
	};

	template <class T>
	class asset;

	template <class T>
	class asset : public asset_control_base
	{
	private:
		union
		{
			T	_Tdata;
			char _Bytes[sizeof(T)];
		};

		friend asset_dictionary;

	public:
		virtual ~asset() override
		{
			_Tdata.~T();
		}

		virtual bool				is_unified_chunck() const override
		{
			return true;
		}


		T&						get() { return _Tdata; }
		T*						get_ptr() { return &_Tdata; }

		template<class... _Types> inline
			T*						internal_construct(_Types&&... _Args)
		{	// make a unique_ptr
			return new (&_Tdata) T(_STD forward<_Types>(_Args)...);
		}

	};

	template <class T>
	class asset_ptr
	{
	private:
		asset_control_base* _control;
		T*					_ptr;
	public:
		T* get() { return  _ptr; }
		const T* get() const { return  _ptr; }
		operator T*() { return _ptr; }
		operator const T*() const { return _ptr; }

		explicit asset_ptr(asset_dictionary* dict, const path& source, T* ptr)
			: _ptr(ptr)
		{
			_control = new asset_control_base();
			_control->_dictionary = dict;
			_control->_ref_path = source;
			_control->_loaded = true;
			_control->_inc_ref();
			_control->_set_desctructor<T>();
		}

		explicit asset_ptr(asset_dictionary* dict, const path& source, task<T*>&& loader)
			: _ptr(nullptr)
		{
			_control = new asset_control_base();
			_control->_dictionary = dict;
			_control->_ref_path = source;
			_control->_loaded = false;
			_control->_inc_ref();
			_control->_set_desctructor<T>();
			_control->_load_task = loader.then([this](T* result) {
				_ptr = result;
				_control->_loaded = true;
			});
		}

		asset_ptr() : _ptr(nullptr), _control(nullptr) {}
		~asset_ptr()
		{
			reset();
		}

		void _internal_destruct()
		{
			(*this->_control->_destructor)(this->_ptr);
		}

		asset_ptr& operator=(const asset_ptr& rhs)
		{
			this->_ptr = rhs._ptr;
			this->_control = rhs._control;
			if (this->_ptr && this->_control)
				this->_control->_inc_ref();
			return *this;
		}

		asset_ptr& operator=(asset_ptr&& rhs)
		{
			swap(rhs);
			return *this;
		}

		asset_ptr(const asset_ptr& rhs) { *this = rhs; }
		asset_ptr(asset_ptr&& rhs) { *this = std::move(rhs); }

		void reset() {
			if (_ptr && !_control->_dec_ref())
			{
				bool is_same_block = _control->is_unified_chunck();

				delete _control;
				_control = nullptr;

				if (!is_same_block)
				{
					delete _ptr;
					_ptr = nullptr;
				}
			}
		}

		void attach(asset<T>* asset)
		{
			reset();
			this->_ptr = asset->get_ptr();
			this->_control = asset;
		}

		void swap(asset_ptr& rhs)
		{
			auto ptr = this->_ptr;
			this->_ptr = rhs._ptr;
			rhs._ptr = this->_ptr;
			auto control = this->_control;
			this->_control = rhs._control;
			rhs._control = this->_control;
		}
	public:
		asset_dictionary*		dictionary() { return !_control ? nullptr : _control->dictionary(); }
		const asset_dictionary*	dictionary() const { return !_control ? nullptr : _control->dictionary(); }

		bool					experied() const { return _control ? _control->ref_count() == 0 : true; }
		ptrdiff_t				ref_count() const { return !_control ? 0 : _control->ref_count(); }

		const path&				source_path() const { return _control->source_path(); }
		bool					is_loaded() { return !_control ? false : _control->is_loaded(); }
		task<T*>&				load_task() { return _control->_load_task; }
	};

	struct ArmatureFrameAnimation;
	class BehavierSpace;
	class StaticArmature;

	class asset_dictionary
	{
	public:
		using mesh_type = IModelNode;
		using texture_type = Texture;
		using audio_clip_type = int;
		using animation_clip_type = ArmatureFrameAnimation;
		using behavier_type = BehavierSpace;
		using armature_type = StaticArmature;
		using effect_type = IEffect;
		using material_type = IMaterial;

		typedef std::function<bool(void*, const char*)> creation_function_type;

		path	asset_directory;
		path	texture_directory;
		path	mesh_directory;
		path	animation_directory;

	private:
		unordered_map<type_index, creation_function_type> creators;
		unordered_map<string, asset_ptr<void>> m_assets;
	public:
		~asset_dictionary()
		{

		}
		// Helper for lazily creating a D3D resource.
		template<typename T, typename TCreateFunc>
		static T* demand_create(asset_ptr<T>& assetPtr, std::mutex& mutex, TCreateFunc createFunc)
		{
			T* result = assetPtr.get();

			// Double-checked lock pattern.
			//MemoryBarrier();

			if (!result)
			{
				std::lock_guard<std::mutex> lock(mutex);

				result = comPtr.Get();

				if (!result)
				{
					// Create the new object.
					createFunc(&assetPtr);

					assetPtr.attach(result);
				}
			}

			return result;
		}

		template<class _Ty, class... _Types> inline
		asset_ptr<_Ty> make_asset(const string& key, _Types&&... _Args)
		{
			auto ptr = new asset<_Ty>(std::forward(_Args)...);
		}

		template<class _Ty>
		asset_ptr<_Ty> get_asset(const string& key) const;

	};
}
