#pragma once
#include <memory>
#include <wrl\client.h>

namespace Causality
{
	template <class T>
	using sptr = std::shared_ptr<T>;
	template <class T>
	using uptr = std::unique_ptr<T>;
	template <class T>
	using wptr = std::weak_ptr<T>;
	template <class T>
	using cptr = Microsoft::WRL::ComPtr<T>;

	using std::weak_ptr;
	using std::unique_ptr;
	using std::shared_ptr;
	using Microsoft::WRL::ComPtr;
	using std::enable_shared_from_this;
	using std::make_unique;
	using std::make_shared;
}