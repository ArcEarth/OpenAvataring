#pragma once
#include <Common\signal.hpp>

namespace Causality
{
	using stdx::signal;
	using stdx::connection;
	using stdx::scoped_connection;

	template <class... TArgs>
	using Event = stdx::signal<void(TArgs...)>;

	template <class TSender, class... TArgs>
	using TypedEvent = stdx::signal<void(TSender*, TArgs...)>;

	using EventConnection = stdx::connection;

	template <class TSender, class TCallback>
	auto MakeEventHandler(TCallback memberFuncPointer, TSender* sender)
	{
		return std::bind(memberFuncPointer, sender, std::placeholders::_1);
	}
}