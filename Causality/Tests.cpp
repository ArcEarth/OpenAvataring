#include "pch_bcl.h"
#include "Tests.h"
#include <map>
#include <iostream>

namespace Causality
{
	using test_function_t = TestManager::test_function_t;

	using test_function_collection = std::map<std::string, test_function_t>;

	static test_function_collection& GetTestMethods()
	{
		static test_function_collection collection;
		return collection;
	}

	void TestManager::RegisterTest(const char* name, std::function<bool(void)> func)
	{
		GetTestMethods()[name] = std::move(func);
	}

	bool TestManager::RunTest()
	{
		for (auto& test : GetTestMethods())
		{
			bool result = test.second();
			if (!result)
			{
				std::cout << "[Test Failed] : Test name == " << test.first << std::endl;
				assert(result && "Unit test failed...");
				return false;
			}
		}
		return true;
	}
}