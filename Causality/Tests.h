#pragma once
#include <functional>
#include <string>

namespace Causality
{
	class TestManager
	{
	public:
		using test_function_t = std::function<bool(void)>;

		static void RegisterTest(const char* name, std::function<bool(void)> func);
		static bool RunTest();
	};

#define REGISTER_TEST_METHOD(test_name,function) \
	struct _test_register_##test_name##_ \
	{ \
		_test_register_##test_name##_ () {\
		::Causality::TestManager::RegisterTest((#test_name),(function)); }\
	} _test_register_##test_name##_instance;
}