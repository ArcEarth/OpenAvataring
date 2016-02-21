/*----------------------------------------------------------------------------
* Copyright © Microsoft Corp.
*
* Licensed under the Apache License, Version 2.0 (the "License"); you may not 
* use this file except in compliance with the License.  You may obtain a copy 
* of the License at http://www.apache.org/licenses/LICENSE-2.0  
* 
* THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED 
* WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, 
* MERCHANTABLITY OR NON-INFRINGEMENT. 
*
* See the Apache Version 2.0 License for specific language governing 
* permissions and limitations under the License.
*---------------------------------------------------------------------------
* 
* C++ AMP standard algorithm library.
*
* This file contains the unit tests.
*---------------------------------------------------------------------------*/

#include "stdafx.h"
#include "amp_stl_algorithms.h"
#include "testtools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace concurrency;
using namespace amp_stl_algorithms;
using namespace testtools;

namespace testtools_tests
{
    TEST_CLASS(testtools_equality_tests)
    {
        TEST_METHOD_CATEGORY(testtools_array_view_equality, "testtools")
        {
            const int size = 10;
            std::vector<int> a(size, 0);
            std::vector<int> b(size, 0);
            array_view<int> b_av(size, b);

            Assert::IsTrue(are_equal(a, b_av));
            Assert::IsTrue(are_equal(b_av, a));

            b_av[6] = 2;

            Assert::IsFalse(are_equal(a, b_av));
            Assert::IsFalse(are_equal(b_av, a));

            b_av = b_av.section(0, 5);

            Assert::IsFalse(are_equal(a, b_av));
            Assert::IsFalse(are_equal(b_av, a));

            a.resize(5);

            Assert::IsTrue(are_equal(a, b_av));
            Assert::IsTrue(are_equal(b_av, a));
        }
    };

    std::wstring Msg(std::vector<int>& expected, std::vector<int>& actual, size_t width = 32)
    {
        std::wostringstream msg;
        msg << container_width(50) << L"[" << expected << L"] != [" << actual << L"]" << std::endl;
        return msg.str();
    }

    TEST_CLASS(testtools_sequential_scan_tests)
    {
        TEST_METHOD_CATEGORY(testtools_sequential_exclusive_scan, "testtools")
        {
            std::array<int, 16> input = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            std::vector<int> result(input.size(), -1);
            std::array<int, 16> expected = { 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120 };

            scan_sequential_exclusive(begin(input), end(input), result.begin());

            std::vector<int> exp(begin(expected), end(expected));
            Assert::IsTrue(exp == result, Msg(exp, result, 16).c_str());
        }

        TEST_METHOD_CATEGORY(testtools_sequential_inclusive_scan, "testtools")
        {
            std::array<int, 16> input = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            std::vector<int> result(input.size(), -1);
            std::array<int, 16> expected = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136 };

            scan_sequential_inclusive(begin(input), end(input), result.begin());

            std::vector<int> exp(begin(expected), end(expected));
            Assert::IsTrue(exp == result, Msg(exp, result, 16).c_str());
        }
    };

    TEST_CLASS(testtools_ostream_tests)
    {
        TEST_METHOD_CATEGORY(testtools_ostream_insertion_std_vector, "testtools")
        {
            auto data = std::vector<int>(5);
            std::iota(begin(data), end(data), 1);
            std::ostringstream stream;
            stream << data;
            Assert::AreEqual("1,2,3,4,", stream.str().c_str());
        }

        TEST_METHOD_CATEGORY(testtools_ostream_insertion_std_array, "testtools")
        {
            auto data = std::array<int, 5>();
            std::iota(begin(data), end(data), 1);
            std::ostringstream stream;
            stream << data;
            Assert::AreEqual("1,2,3,4,", stream.str().c_str());
        }

        TEST_METHOD_CATEGORY(testtools_ostream_insertion_amp_array_view, "testtools")
        {
            auto data = std::vector<int>(5);
            std::iota(begin(data), end(data), 1);
            auto data_vw = concurrency::array_view<int, 1>(5, data.data());
            std::ostringstream stream;
            stream << data_vw;
            Assert::AreEqual("1,2,3,4,", stream.str().c_str());
        }

        TEST_METHOD_CATEGORY(testtools_ostream_insertion_amp_array, "testtools")
        {
            auto data = std::vector<int>(5);
            std::iota(begin(data), end(data), 1);
            auto data_arr = concurrency::array<int, 1>(5);
            copy(begin(data), end(data), data_arr);
            std::ostringstream stream;
            stream << data_arr;
            Assert::AreEqual("1,2,3,4,", stream.str().c_str());
        }

        TEST_METHOD_CATEGORY(testtools_ostream_insertion_container_width_limited, "testtools")
        {
            auto data = std::vector<int>(5);
            std::iota(begin(data), end(data), 1);
            std::ostringstream stream;
            stream << container_width(2) << data << container_width(3) << " " << data;
            Assert::AreEqual("1,2, 1,2,3,", stream.str().c_str());
        }
    };
};
