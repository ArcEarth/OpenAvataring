#pragma once
#include <boost\signals2.hpp>
#include <functional>

namespace Causality
{
	template <class... TArgs>
	using Event = boost::signals2::signal<void(TArgs...)>;

	template <class TSender, class... TArgs>
	using TypedEvent = boost::signals2::signal<void(TSender*, TArgs...)>;

	using EventConnection = boost::signals2::connection;

	template <class TSender, class TCallback>
	auto MakeEventHandler(TCallback memberFuncPointer, TSender* sender)
	{
		return std::bind(memberFuncPointer, sender, std::placeholders::_1);
	}

	template <class TArg, class TCallback>
	inline auto operator+=(Event<TArg>& signal, TCallback &&callback)
	{
		return signal.connect(std::move(callback));
	}

	template <class TArg, class TCallback>
	inline auto operator+=(Event<TArg>& signal, TCallback &callback)
	{
		return signal.connect(callback);
	}

	template <class TArg, class TCallback>
	inline void operator-=(Event<TArg>& signal, TCallback &&callback)
	{
		signal.disconnect(std::move(callback));
	}
}