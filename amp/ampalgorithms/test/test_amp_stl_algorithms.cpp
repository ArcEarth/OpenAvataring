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

#include <amp_stl_algorithms.h>
#include "testtools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace concurrency;
using namespace amp_stl_algorithms;
using namespace testtools;

namespace Microsoft {
    namespace VisualStudio {
        namespace CppUnitTestFramework
        {
            // TODO: Are all these overloads required? Should be able to just have std::pair and rely on casting from amp_stl_algorithms::pair.
            // TODO: Might want to  move ToString overloads into testtools and templatize them.
            template<> 
            static std::wstring ToString<std::pair<const int&, const int&>>(const std::pair<const int&, const int&>& v)
            { 
                std::wstringstream str;
                str << v.first << ", " << v.second;
                return str.str();
            }

            template<> 
            static std::wstring ToString<std::pair<int, int>>(const std::pair<int, int>& v)
            {
                std::wstringstream str;
                str << v.first << ", " << v.second;
                return str.str();
            }

            template<>
            static std::wstring ToString<std::pair<const int, const int>>(const std::pair<const int, const int>& v)
            {
                std::wstringstream str;
                str << v.first << ", " << v.second;
                return str.str();
            }

            template<> 
            static std::wstring ToString<amp_stl_algorithms::pair<int, int>>(const amp_stl_algorithms::pair<int, int>& v)
            {
                std::wstringstream str;
                str << v.first << ", " << v.second;
                return str.str();
            }
        }
    }
}

namespace amp_stl_algorithms_tests
{
    // This isn't a test, it's just a convenient way to determine which accelerator tests ran on.
    TEST_CLASS(testtools_configuration)
    {
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            set_default_accelerator(L"amp_stl_algorithms_tests_configuration");
        }
    };
    
    // TODO: Get the tests, header and internal implementations into the same logical order.

    TEST_CLASS(stl_pair_tests)
    {
        // TODO: Add more tests for pair<T, T>, casting etc.
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            set_default_accelerator(L"stl_pair_tests");
        }

        TEST_METHOD_INITIALIZE(initialize_test)
        {
            accelerator().default_view.wait();
        }

        TEST_METHOD_CATEGORY(stl_pair_property_accessors, "stl")
        {
            amp_stl_algorithms::pair<int, int> dat(1, 2);
            array_view<amp_stl_algorithms::pair<int, int>> dat_vw(1, &dat); 
 
            concurrency::parallel_for_each(dat_vw.extent, [=](concurrency::index<1> idx) restrict(amp)
            {
                amp_stl_algorithms::swap(dat_vw[idx].first, dat_vw[idx].second);
            });

            Assert::AreEqual(2, dat_vw[0].first);
            Assert::AreEqual(1, dat_vw[0].second);
        }

        TEST_METHOD_CATEGORY(stl_pair_copy, "stl")
        {
            amp_stl_algorithms::pair<int, int> dat(1, 2);
            auto dat_vw = array_view<amp_stl_algorithms::pair<int, int>>(1, &dat);

            concurrency::parallel_for_each(dat_vw.extent, [=](concurrency::index<1> idx) restrict(amp)
            {
                amp_stl_algorithms::pair<int, int> x(3, 4);
                dat_vw[0] = x;
            });

            Assert::AreEqual(3, dat_vw[0].first);
            Assert::AreEqual(4, dat_vw[0].second);
        }

        TEST_METHOD_CATEGORY(stl_pair_conversion_from_std_pair, "stl")
        {
            std::pair<int, int> y(1, 2);

            amp_stl_algorithms::pair<int, int> x = y;

            Assert::AreEqual(1, x.first);
            Assert::AreEqual(2, x.second);
        }

        TEST_METHOD_CATEGORY(stl_pair_conversion_to_std_pair, "stl")
        {
            amp_stl_algorithms::pair<int, int> y(1, 2);

            std::pair<int, int> x = y;

            Assert::AreEqual(1, x.first);
            Assert::AreEqual(2, x.second);
        }
    };

    TEST_CLASS(stl_algorithms_tests)
    {
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            set_default_accelerator(L"stl_algorithms_tests");
        }

        TEST_METHOD_INITIALIZE(initialize_test)
        {
            accelerator().default_view.wait();
        }

        //----------------------------------------------------------------------------
        // adjacent_difference
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_adjacent_difference, "stl")
        {
            const int size = 10;
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
            std::vector<int> result(size, 0);
            array_view<int> result_av(size, result);
            std::array<int, size> expected; 
            std::fill(begin(expected), end(expected), -1);

            // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

            std::adjacent_difference(begin(vec), end(vec), begin(expected));

            auto result_last = amp_stl_algorithms::adjacent_difference(begin(av), end(av), begin(result_av));

            Assert::AreEqual(size, std::distance(begin(result_av), result_last));
            result_av.synchronize();
            Assert::IsTrue(are_equal(expected, result));

            // 0, 1, 2, 3, 5, 5, 6, 7, 8, 9

            vec[4] = 5; 
            av.refresh();
            std::adjacent_difference(begin(vec), end(vec), begin(expected));

            amp_stl_algorithms::adjacent_difference(begin(av), end(av), begin(result_av));

            Assert::IsTrue(are_equal(expected, result_av));

            // 1, 1, 2, 3, 5, 5, 6, 7, 8, 9

            vec[0] = 1; 
            av.refresh();
            std::adjacent_difference(begin(vec), end(vec), begin(expected));

            amp_stl_algorithms::adjacent_difference(begin(av), end(av), begin(result_av));

            Assert::IsTrue(are_equal(expected, result_av));

            // 1, 1, 2, 3, 5, 5, 6, 3, 8, 9

            vec[7] = 3;  
            av.refresh();
            std::adjacent_difference(begin(vec), end(vec), begin(expected), std::plus<int>());

            amp_stl_algorithms::adjacent_difference(begin(av), end(av), begin(result_av), plus<int>());

            Assert::IsTrue(are_equal(expected, result_av));

            // Empty

            result_last = amp_stl_algorithms::adjacent_difference(begin(av), begin(av), begin(result_av));

            Assert::AreEqual(0, std::distance(begin(result_av), result_last));

            // 1

            std::adjacent_difference(begin(vec), begin(vec) + 1, begin(expected));

            result_last = amp_stl_algorithms::adjacent_difference(begin(av), begin(av) + 1, begin(result_av));

            Assert::AreEqual(expected[0], result_av[0]);
        }

        TEST_METHOD_CATEGORY(stl_adjacent_difference_multi_tile, "stl")
        {
            const int size = 1024;
            std::vector<int> vec(size);
            generate_data(vec);
            array_view<int> av(size, vec);
            std::vector<int> result(size, 0);
            array_view<int> result_av(size, result);
            std::array<int, size> expected; 
            std::fill(begin(expected), end(expected), -1);

            std::adjacent_difference(begin(vec), end(vec), begin(expected));

            auto result_last = amp_stl_algorithms::adjacent_difference(begin(av), end(av), begin(result_av));

            Assert::AreEqual(size, std::distance(begin(result_av), result_last));
            Assert::IsTrue(are_equal(expected, result_av));
        }

        //----------------------------------------------------------------------------
        // all_of, any_of, none_of
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_none_of, "stl")
        {
            static const int numbers[] = { 1, 3, 6, 3, 2, 2 };
            static const int n = sizeof(numbers)/sizeof(numbers[0]);

            array_view<const int> av(concurrency::extent<1>(n), numbers);
            bool r1 = amp_stl_algorithms::none_of(begin(av), end(av), [] (int v) restrict(amp) -> bool { return v>10; });
            Assert::IsTrue(r1);
            bool r2 = amp_stl_algorithms::none_of(begin(av), end(av), [] (int v) restrict(amp) -> bool { return v>5; });
            Assert::IsFalse(r2);
        }

        TEST_METHOD_CATEGORY(stl_any_of, "stl")
        {
            static const int numbers[] = { 1, 3, 6, 3, 2, 2 };
            static const int n = sizeof(numbers)/sizeof(numbers[0]);

            array_view<const int> av(concurrency::extent<1>(n), numbers);
            bool r1 = amp_stl_algorithms::any_of(begin(av), end(av), [] (int v) restrict(amp) -> bool { return v > 10; });
            Assert::IsFalse(r1);
            bool r2 = amp_stl_algorithms::any_of(begin(av), end(av), [] (int v) restrict(amp) -> bool { return v > 5; });
            Assert::IsTrue(r2);
        }

        TEST_METHOD_CATEGORY(stl_all_of, "stl")
        {
            static const int numbers[] = { 1, 3, 6, 3, 2, 2 };
            static const int n = sizeof(numbers)/sizeof(numbers[0]);

            array_view<const int> av(concurrency::extent<1>(n), numbers);

            bool r1 = amp_stl_algorithms::all_of(begin(av), end(av), [] (int v) restrict(amp) -> bool { return v > 10; });
            Assert::IsFalse(r1);
            bool r2 = amp_stl_algorithms::all_of(begin(av), end(av), [] (int v) restrict(amp) -> bool { return v > 5; });
            Assert::IsFalse(r2);
            bool r3 = amp_stl_algorithms::all_of(begin(av), end(av), [](int v) restrict(amp) -> bool { return v < 10; });
            Assert::IsTrue(r3);
        }

        //----------------------------------------------------------------------------
        // copy, copy_if, copy_n, copy_backward
        //----------------------------------------------------------------------------
        TEST_METHOD_CATEGORY(stl_copy, "stl")
        {
            test_copy(1024);
        }

        void test_copy(const int size)
        {
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 1);
            array_view<int> av(size, vec);

            std::vector<int> result(size, 0);
            array_view<int> result_av(size, result);

            auto result_end = amp_stl_algorithms::copy(begin(av), end(av), begin(result));

            Assert::IsTrue(are_equal(av, result));
            Assert::AreEqual(result_av[size - 1], *--result_end);
        }

        TEST_METHOD_CATEGORY(stl_copy_if_no_values_to_copy, "stl")
        {
            const std::array<int, 5> numbers = { 0, 0, 0, 0, 0 };
            test_copy_if(begin(numbers), end(numbers));
        }

        TEST_METHOD_CATEGORY(stl_copy_if_first_value_matches, "stl")
        {
            const std::array<int, 5> numbers = { 2, 0, 0, 0, 0 };
            test_copy_if(begin(numbers), end(numbers));
        }

        TEST_METHOD_CATEGORY(stl_copy_if_last_value_matches, "stl")
        {
            const std::array<float, 5> numbers = { 0.0f, 0.0f, 0.0f, 0.0f, 3.0f };
            test_copy_if(begin(numbers), end(numbers));
        }

        TEST_METHOD_CATEGORY(stl_copy_if, "stl")
        {

            const std::array<int, 12> numbers2 = { -1, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7 };
            //  predicate result:                   1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1
            //  Exclusive scan result:              0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8
            //  Final result:                      -1, 1, 2, 3, 4, 5, 6, 7
            test_copy_if(begin(numbers2), end(numbers2));

            const std::array<int, 5> numbers3 = { 0, 0, 0, 0, 0 };
            test_copy_if(begin(numbers3), end(numbers3));

            std::vector<int> numbers4(test_array_size<int>());
            generate_data(numbers4);
            test_copy_if(begin(numbers4), end(numbers4));
        }

        TEST_METHOD_CATEGORY(stl_copy_if_multi_tile, "stl")
        {
            std::vector<int> numbers4(test_array_size<int>());
            generate_data(numbers4);
            test_copy_if(begin(numbers4), end(numbers4));
        }

        template <typename InIt>
        void test_copy_if(InIt first, InIt last)
        {
            typedef typename std::iterator_traits<InIt>::value_type T;
            int size = int(std::distance(first, last));

            // Calculate expected result for copy all non-zeros.

            std::vector<T> expected(size, -42);
            auto expected_end = std::copy_if(first, last, begin(expected), [=] (const T i)
            { 
                return (i != 0) ? 1 : 0; 
            });
            expected.resize(distance(begin(expected), expected_end));

            // Calculate actual result

            array_view<const T> input_av(concurrency::extent<1>(size), &first[0]);
            std::vector<T> result(size, -42);
            array_view<T> result_av(concurrency::extent<1>(size), result);
            auto dest_end = amp_stl_algorithms::copy_if(begin(input_av), 
                end(input_av), begin(result_av), [=] (const T i) restrict(amp) 
            { 
                return (i != 0) ? 1 : 0; 
            });
            result_av.synchronize();

            Assert::AreEqual(expected.size(), size_t(std::distance(begin(result_av), dest_end)));
            for (size_t i = 0; i < expected.size(); ++i)
            {
                Assert::AreEqual(expected[i], result[i]);
            }
        }

        TEST_METHOD_CATEGORY(stl_copy_n, "stl")
        {
            const int size = 1023;
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 1);
            array_view<int> av(size, vec);

            std::vector<int> result(size, 0);
            array_view<int> result_av(size, result);

            auto result_end = amp_stl_algorithms::copy_n(begin(av), 512, begin(result));

            Assert::IsTrue(are_equal(av.section(0, 512), result_av.section(0, 512)));
            Assert::IsFalse(are_equal(av.section(512, 1), result_av.section(512, 2)));
            Assert::AreEqual(512, int(std::distance(begin(result), result_end)));
        }

        //----------------------------------------------------------------------------
        // count, count_if
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_count, "stl")
        {
            static const int numbers[] = {1, 3, 6, 3, 2, 2, 7, 8, 2, 9, 2, 19, 2};
            static const int n = sizeof(numbers)/sizeof(numbers[0]);
            array_view<const int> av(concurrency::extent<1>(n), numbers);
            auto r1 = amp_stl_algorithms::count(begin(av), end(av), 2);
            Assert::AreEqual(5, r1);
            auto r2 = amp_stl_algorithms::count(begin(av), end(av), 17);
            Assert::AreEqual(0, r2);
        }

        TEST_METHOD_CATEGORY(stl_count_if, "stl")
        {
            static const int numbers[] = {1, 3, 6, 3, 2, 2, 7, 8, 2, 9, 2, 19, 2};
            static const int n = sizeof(numbers)/sizeof(numbers[0]);
            array_view<const int> av(concurrency::extent<1>(n), numbers);
            auto r1 = amp_stl_algorithms::count_if(begin(av), end(av), [=](const int& v) restrict(amp) { return (v == 2); });
            Assert::AreEqual(5, r1);
            auto r2 = amp_stl_algorithms::count_if(begin(av), end(av), [=](const int& v) restrict(amp) { return (v == 17); });
            Assert::AreEqual(0, r2);
        }

        //----------------------------------------------------------------------------
        // equal
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_equal, "stl")
        {
            std::vector<int> vec1(1024);
            std::iota(begin(vec1), end(vec1), 1);
            array_view<int> av1(1024, vec1);
            std::vector<int> vec2(1024);
            std::iota(begin(vec2), end(vec2), 1);
            array_view<int> av2(1024, vec2);

            Assert::IsTrue(amp_stl_algorithms::equal(begin(av1), end(av1), begin(av2)));

            av2[512] = 1;

            Assert::IsFalse(amp_stl_algorithms::equal(begin(av1), end(av1), begin(av2)));

            av1 = av1.section(0, 512);

            Assert::IsTrue(amp_stl_algorithms::equal(begin(av1), end(av1), begin(av2)));
        }

        TEST_METHOD_CATEGORY(stl_equal_pred, "stl")
        {
            std::vector<int> vec1(1024);
            std::iota(begin(vec1), end(vec1), 1);
            array_view<int> av1(1024, vec1);
            std::vector<int> vec2(1024);
            std::iota(begin(vec2), end(vec2), 1);
            array_view<int> av2(1024, vec2);

            auto pred = [=](int& v1, int& v2) restrict(amp) { return ((v1 + 1) == v2); };
            Assert::IsFalse(amp_stl_algorithms::equal(begin(av1), end(av1), begin(av2), pred));

            std::iota(begin(vec2), end(vec2), 2);
            av2.refresh();

            Assert::IsTrue(amp_stl_algorithms::equal(begin(av1), end(av1), begin(av2), pred));
        }

        // TODO: These iterator tests are in the wrong place, move them.


        //----------------------------------------------------------------------------
        // fill, fill_n
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_fill, "stl")
        {
            std::vector<int> vec(1024);

            // Fill using an array_view iterator
            array_view<int> av(1024, vec);
            av.discard_data();

            amp_stl_algorithms::fill(begin(av), end(av), 7);
            av.synchronize();

            for (auto element : vec)
            {
                Assert::AreEqual(7, element);
            }
        }

        TEST_METHOD_CATEGORY(stl_fill_n, "stl")
        {
            std::vector<int> vec(1024);
            array_view<int> av(1024, vec);
            av.discard_data();

            amp_stl_algorithms::fill_n(begin(av), av.extent.size(), 616);
            av.synchronize();

            for (auto element : vec)
            {
                Assert::AreEqual(616, element);
            }
        }

        //----------------------------------------------------------------------------
        // find, find_if, find_if_not, find_end, find_first_of, adjacent_find
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_find, "stl")
        {
            static const int numbers[] = { 1, 3, 6, 3, 2, 2 };
            static const int n = sizeof(numbers)/sizeof(numbers[0]);
            array_view<const int> av(concurrency::extent<1>(n), numbers);

            auto iter = amp_stl_algorithms::find(begin(av), end(av), 3);
            int position = std::distance(begin(av), iter);
            Assert::AreEqual(1, position);

            iter = amp_stl_algorithms::find(begin(av), end(av), 17);
            Assert::IsTrue(end(av) == iter);
        }

        TEST_METHOD_CATEGORY(stl_find_if, "stl")
        {
            static const int numbers[] = { 1, 3, 6, 3, 2, 2 };
            static const int n = sizeof(numbers)/sizeof(numbers[0]);

            array_view<const int> av(concurrency::extent<1>(n), numbers);
            auto iter = amp_stl_algorithms::find_if(begin(av), end(av), [=](int v) restrict(amp) { return v == 3; });
            int position = std::distance(begin(av), iter);
            Assert::AreEqual(1, position);

            iter = amp_stl_algorithms::find_if(begin(av), end(av), [=](int v) restrict(amp) { return v == 17; });
            Assert::IsTrue(end(av) == iter);
        }

        TEST_METHOD_CATEGORY(stl_find_if_not, "stl")
        {
            static const int numbers[] = { 1, 3, 6, 3, 2, 2 };
            static const int n = sizeof(numbers)/sizeof(numbers[0]);

            array_view<const int> av(concurrency::extent<1>(n), numbers);
            auto iter = amp_stl_algorithms::find_if_not(begin(av), end(av), [=](int v) restrict(amp) { return v != 3; });
            int position = std::distance(begin(av), iter);
            Assert::AreEqual(1, position);

            iter = amp_stl_algorithms::find_if_not(begin(av), end(av), [=](int v) restrict(amp) { return v != 17; });
            Assert::IsTrue(end(av) == iter);
        }

        TEST_METHOD_CATEGORY(stl_adjacent_find, "stl")
        {
            const int size = 10;
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            Assert::AreEqual(0, std::distance(begin(av), amp_stl_algorithms::adjacent_find(begin(av), begin(av))));
            Assert::AreEqual(1, std::distance(begin(av), amp_stl_algorithms::adjacent_find(begin(av), begin(av)+1)));

            Assert::AreEqual(10, std::distance(begin(av), amp_stl_algorithms::adjacent_find(begin(av), end(av))));

            av[5] = 4;   // 0, 1, 2, 3, 4, 4, 6, 7, 8, 9

            Assert::AreEqual(4, std::distance(begin(av), amp_stl_algorithms::adjacent_find(begin(av), end(av))));

            av[0] = 1;   // 1, 1, 2, 3, 4, 4, 6, 7, 8, 9

            Assert::AreEqual(0, std::distance(begin(av), amp_stl_algorithms::adjacent_find(begin(av), end(av))));
        }

        //----------------------------------------------------------------------------
        // for_each, for_each_no_return
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_for_each_no_return, "stl")
        {
            std::vector<int> vec(1024);
            std::fill(vec.begin(), vec.end(), 2);
            array_view<const int> av(1024, vec);
            int sum = 0;
            array_view<int> av_sum(1, &sum);
            amp_stl_algorithms::for_each_no_return(begin(av), end(av), [av_sum] (int val) restrict(amp) {
                concurrency::atomic_fetch_add(&av_sum(0), val);
            });
            av_sum.synchronize();
            Assert::AreEqual(1024 * 2, sum);
        }

        //----------------------------------------------------------------------------
        // generate, generate_n
        //----------------------------------------------------------------------------
        
        TEST_METHOD_CATEGORY(stl_generate, "stl")
        {
            std::vector<int> vec(1024);

            // Generate using an array_view over the vector. Requires explicit synchronize.
            array_view<int> av(1024, vec);
            av.discard_data();

            amp_stl_algorithms::generate(begin(av), end(av), [] () restrict(amp) 
            {
                return 7;
            });
            av.synchronize();

            for (auto element : vec)
            {
                Assert::AreEqual(7, element);
            }
        }

        TEST_METHOD_CATEGORY(stl_generate_n, "stl")
        {
            std::vector<int> vec(1024);
            array_view<int> av(1024, vec);
            av.discard_data();

            amp_stl_algorithms::generate_n(begin(av), av.extent.size(), [] () restrict(amp) 
            {
                return 616;
            });
            av.synchronize();

            for (auto element : vec)
            {
                Assert::AreEqual(616, element);
            }
        }

        //----------------------------------------------------------------------------
        // inner_product
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_inner_product, "stl")
        {
            std::vector<int> vec1(1024);
            std::fill(begin(vec1), end(vec1), 1);
            array_view<int> av1(1024, vec1);
            std::vector<int> vec2(1024);
            std::fill(begin(vec2), end(vec2), 2);
            array_view<int> av2(1024, vec2);
            int expected = std::inner_product(begin(vec1), end(vec1), begin(vec2), 2);

            int result = amp_stl_algorithms::inner_product(begin(av1), end(av1), begin(av2), 2);

            Assert::AreEqual(expected, result);
        }

        TEST_METHOD_CATEGORY(stl_inner_product_pred, "stl")
        {
            std::vector<int> vec1(1024);
            std::fill(begin(vec1), end(vec1), 1);
            array_view<int> av1(1024, vec1);
            std::vector<int> vec2(1024);
            std::fill(begin(vec2), end(vec2), 2);
            array_view<int> av2(1024, vec2);
            int expected = std::inner_product(begin(vec1), end(vec1), begin(vec2), 2, std::plus<int>(), std::plus<int>());

            int result = amp_stl_algorithms::inner_product(begin(av1), end(av1), begin(av2), 2, amp_algorithms::plus<int>(), amp_algorithms::plus<int>());

            Assert::AreEqual(expected, result);
        }

        //----------------------------------------------------------------------------
        // iota
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_iota, "stl")
        {
            const int size = 1024;
            std::vector<int> vec(size);
            array_view<int> av(size, vec);
            av.discard_data();

            amp_stl_algorithms::iota(begin(av), end(av), 0);

            for (unsigned i = 0; i < av.extent.size(); ++i)
            {
                Assert::AreEqual(int(i), av[i]);
            }
        }

        //----------------------------------------------------------------------------
        // minmax, max_element, min_element, minmax_element
        //----------------------------------------------------------------------------
        // TODO: Should be able to make these tests a bit tidier with better casting support for pair<T, T>
        TEST_METHOD_CATEGORY(stl_minmax, "stl")
        {
            compare_operators(
                [=](int a, int b)->std::pair<const int, const int> { return std::minmax(a, b); },
                [=](int a, int b)->std::pair<const int, const int> 
            { 
                return amp_stl_algorithms::minmax(a, b); 
            });
        }

        TEST_METHOD_CATEGORY(stl_minmax_pred, "stl")
        {
            //std::pair<const int&, const int&>(*minmax) (const int&, const int&) = std::minmax<int>;

            compare_operators(
                [=](int& a, int& b)->std::pair<const int, const int> { return std::minmax(a, b, std::greater_equal<int>()); },
                [=](int& a, int& b)->std::pair<const int, const int>
            { 
                return amp_stl_algorithms::minmax(a, b, amp_algorithms::greater_equal<int>()); 
            });
        }

        //----------------------------------------------------------------------------
        // reduce
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_reduce_sum, "stl")
        {
            static const int numbers[] = {1, 3, 6, 3, 2, 2, 7, 8, 2, 9, 2, 19, 2};
            static const int n = sizeof(numbers)/sizeof(numbers[0]);
            array_view<const int> av(concurrency::extent<1>(n), numbers);
            auto result = amp_stl_algorithms::reduce(begin(av), end(av), 0);
            Assert::AreEqual(66, result);
        }

        TEST_METHOD_CATEGORY(stl_reduce_max, "stl")
        {
            static const int numbers[] = {1, 3, 6, 3, 2, 2, 7, 8, 2, 9, 2, 19, 2};
            static const int n = sizeof(numbers)/sizeof(numbers[0]);
            array_view<const int> av(concurrency::extent<1>(n), numbers);
            auto result = amp_stl_algorithms::reduce(begin(av), end(av), 0, [](int a, int b) restrict(cpu, amp) {
                return (a < b) ? b : a;
            });
            Assert::AreEqual(19, result);
        }

        //----------------------------------------------------------------------------
        // remove, remove_if, remove_copy, remove_copy_if
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_remove, "stl")
        {
            const int size = 10;
            std::array<int, (size - 1)> expected = { 0, 2, 3, 4, 5, 6, 7, 8, 9 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
 
            auto expected_end = amp_stl_algorithms::remove(begin(av), end(av), 1);
            av = av.section(0, std::distance(begin(av), expected_end));

            Assert::AreEqual(unsigned(size - 1), av.extent.size());
            Assert::IsTrue(are_equal(expected, av));
            Assert::AreEqual(9, *--expected_end);
        }

        TEST_METHOD_CATEGORY(stl_remove_if, "stl")
        {
            // These tests remove all the non-zero elements from the numbers array.

            std::array<float, 5> numbers0 = { 1, 1, 1, 1, 1 };
            test_remove_if(begin(numbers0), end(numbers0));

            std::array<float, 5> numbers1 = { 3, 0, 0, 0, 0 };
            test_remove_if(begin(numbers1), end(numbers1));

            std::array<float, 5> numbers2 = { 0, 0, 0, 0, 3 };
            test_remove_if(begin(numbers2), end(numbers2));
            
            std::array<float, 12> numbers3 = { -1, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7 };
            //  Predicate result:               0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0
            //  Scan result:                    0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4
            //  Final result:                  -1, 1, 2, 3, 4, 5, 6, 7
            test_remove_if(begin(numbers3), end(numbers3));
            
            std::vector<int> numbers4(test_array_size<int>());
            test_remove_if(begin(numbers4), end(numbers4));
        }

        // Customer reported bug. 
        // See: http://social.msdn.stl_remove_if_performance.com/Forums/vstudio/en-US/d959e3f3-2a85-4646-9c54-cae69c534b64
        BEGIN_TEST_METHOD_ATTRIBUTE(stl_remove_if_performance)
            TEST_CATEGORY("stl")
#if (defined(USE_REF) || defined(_DEBUG))
            TEST_IGNORE()
#endif
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(stl_remove_if_performance)
        {
            accelerator device(accelerator::default_accelerator);
            accelerator_view view = device.default_view;
            char buff[100];

            int size = 2048 * 1088 * 5;
            std::vector<float> h_vec(size);

            // Generate random numbers
            std::generate(h_vec.begin(), h_vec.end(), []() { return (static_cast<float>(rand()) / RAND_MAX); });

            // Copy to device
            concurrency::array<float, 1> d_vec(size);
            double tCopy1 = time_func(view, [&]() 
            {
                concurrency::copy(h_vec.begin(), h_vec.end(), d_vec);
            });
            array_view<float, 1> d_view(d_vec);

            sprintf_s<100>(buff, "Copied to GPU in %.1f ms", tCopy1);
            Logger::WriteMessage(buff);

            array_view_iterator<float> last;
            double tRemove = time_func(view, [&]() 
            {
                last = amp_stl_algorithms::remove_if(begin(d_view), end(d_view), [](const float v) restrict(amp)
                {
                    return ((v > 0.1) && (v < 0.9));
                });
            });

            sprintf_s<100>(buff, "amp_stl_algorithms::remove_if completed in %.1f ms", tRemove);
            Logger::WriteMessage(buff);

            int new_size = last - begin(d_view);
            sprintf_s<100>(buff, "Before: %d elements - After: %d elements", h_vec.size(), new_size);
            Logger::WriteMessage(buff);

            h_vec.resize(new_size);

            double tCopy2 = time_func(view, [&]() 
            {
                amp_stl_algorithms::copy(begin(d_view), last, h_vec.begin());
            });
            sprintf_s<100>(buff, "Copied to host in %.1f ms - Total: %.1f ms", tCopy2, tCopy1 + tRemove + tCopy2);
            Logger::WriteMessage(buff);
        }

        template <typename InIt>
        void test_remove_if(InIt first, InIt last)
        {
            typedef typename std::iterator_traits<InIt>::value_type T;
            const int size = int(std::distance(first, last));

            // Calculate expected result, for remove all zeros.

            std::vector<T> expected(size);
            std::copy(first, last, begin(expected));
            auto last_elem = std::remove_if(begin(expected), end(expected), [=] (const T i)
            {
                return (i > 0) ? 1 : 0;
            });
            expected.resize(std::distance(begin(expected), last_elem));

            // Calculate actual result

            array_view<T> av(concurrency::extent<1>(size), &first[0]);
            auto result_end = amp_stl_algorithms::remove_if(begin(av), 
                end(av), [=] (const T i) restrict(amp) 
            { 
                return (i > 0) ? 1 : 0;
            });
            av.synchronize();
            auto actual_size = distance(begin(av), result_end);

            Assert::IsTrue((actual_size != 0) || ((expected.size() == 0) && (actual_size == 0)));
            if (actual_size == 0)
            {
                return;
            }
            Assert::IsTrue(are_equal(expected, av.section(0, actual_size))); 
        }

        TEST_METHOD_CATEGORY(stl_remove_if_nop, "stl")
        {
            const int size = 10;
            std::array<int, 10> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            std::vector<int> vec(begin(expected), end(expected));
            array_view<int> av(size, vec);

            amp_stl_algorithms::remove_if(begin(av), begin(av), [=](int v) restrict(amp) { return (v % 2 != 0); });

            Assert::IsTrue(are_equal(expected, av));
        }

        TEST_METHOD_CATEGORY(stl_remove_copy, "stl")
        {
            const int size = 10;
            std::array<int, 9> expected = { 0, 2, 3, 4, 5, 6, 7, 8, 9 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
            std::vector<int> result(size, 0);
            array_view<int> result_av(size, result);

            auto expected_end = amp_stl_algorithms::remove_copy(begin(av), end(av), begin(result_av), 1);

            Assert::AreEqual(int(expected.size()), std::distance(begin(result_av), expected_end));
            Assert::IsTrue(are_equal(expected, result_av.section(0, int(expected.size()))));
        }

        TEST_METHOD_CATEGORY(stl_remove_copy_if, "stl")
        {
            const int size = 10;
            std::array<int, (6)> expected = { 0, 1, 2, 3, 4, 5 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
            std::vector<int> result(size, 0);
            array_view<int> result_av(size, result);

            auto expected_end = amp_stl_algorithms::remove_copy_if(begin(av), end(av), begin(result_av), [=](int& v) restrict(amp) { return (v > 5); });

            Assert::AreEqual(int(expected.size()), std::distance(begin(result_av), expected_end));
            Assert::IsTrue(are_equal(expected, result_av.section(0, int(expected.size()))));
        }

        //----------------------------------------------------------------------------
        // replace, replace_if, replace_copy, replace_copy_if
        //----------------------------------------------------------------------------
        
        TEST_METHOD_CATEGORY(stl_replace, "stl")
        {
            const int size = 10;
            std::array<int, size> expected = { 0, -1, 2, 3, 4, 5, 6, 7, 8, 9 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            amp_stl_algorithms::replace(begin(av), end(av), 1, -1);

            Assert::IsTrue(are_equal(expected, av));
        }

        TEST_METHOD_CATEGORY(stl_replace_if, "stl")
        {
            const int size = 10;
            std::array<int, size> expected = { 0, -1, 2, -1, 4, -1, 6, -1, 8, -1 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            amp_stl_algorithms::replace_if(begin(av), end(av), [=](int v) restrict(amp) { return (v % 2 != 0); }, -1);

            Assert::IsTrue(are_equal(expected, av));
        }

        TEST_METHOD_CATEGORY(stl_replace_if_nop, "stl")
        {
            const int size = 10;
            std::array<int, 10> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            std::vector<int> vec(begin(expected), end(expected));
            array_view<int> av(size, vec);

            amp_stl_algorithms::replace_if(begin(av), begin(av), [=](int v) restrict(amp) { return (v % 2 != 0); }, -1);

            Assert::IsTrue(are_equal(expected, av));
        }

        TEST_METHOD_CATEGORY(stl_replace_copy, "stl")
        {
            const int size = 10;
            std::array<int, size> expected = { 0, -1, 2, 3, 4, 5, 6, 7, 8, 9 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
            std::vector<int> result(size, -2);
            array_view<int> result_av(size, result);

            auto result_end = amp_stl_algorithms::replace_copy(begin(av), end(av), begin(result_av), 1, -1);

            result_av.synchronize();
            Assert::IsTrue(are_equal(expected, result_av));
            Assert::AreEqual(2, std::distance(begin(result_av), result_end));
        }

        TEST_METHOD_CATEGORY(stl_replace_copy_if_odd_numbers, "stl")
        {
            const int size = 10;
            std::array<int, size> expected = { 0, -1, 2, -1, 4, -1, 6, -1, 8, -1 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
            std::vector<int> result(size, -2);
            array_view<int> result_av(size, result);

            auto result_end = amp_stl_algorithms::replace_copy_if(begin(av), end(av), begin(result_av), 
                [=](int v) restrict(amp) { return (v % 2 != 0); }, -1);

            Assert::IsTrue(are_equal(expected, result_av));
            Assert::AreEqual(10, std::distance(begin(result_av), result_end));
        }

        TEST_METHOD_CATEGORY(stl_replace_copy_if_lower_half, "stl")
        {
            const int size = 10;
            std::array<int, size> expected = { -1, -1, -1, -1, -1, 5, 6, 7, 8, 9 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);
            std::vector<int> result(size, -2);
            array_view<int> result_av(size, result);

            auto result_end = amp_stl_algorithms::replace_copy_if(begin(av), end(av), begin(result_av), 
                [=](int v) restrict(amp) { return (v < 5); }, -1);

            Assert::IsTrue(are_equal(expected, result_av));
            Assert::AreEqual(5, std::distance(begin(result_av), result_end));
        }

        //----------------------------------------------------------------------------
        // reverse, reverse_copy
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_reverse, "stl")
        {
            test_reverse(1);
            test_reverse(1023);
            test_reverse(1024);
        }

        void test_reverse(const int size)
        {
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            amp_stl_algorithms::reverse(begin(av), end(av));

            for (int i = 0; i < int(vec.size()); ++i)
            {
                Assert::AreEqual((size - 1 - i), av[i]);
            }
        }

        TEST_METHOD_CATEGORY(stl_reverse_copy, "stl")
        {
            test_reverse_copy(1);
            test_reverse_copy(1023);
            test_reverse_copy(1024);
        }

        void test_reverse_copy(const int size)
        {
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            std::vector<int> result(size, 0);
            concurrency::array_view<int> result_av(size, result);

            auto result_end = amp_stl_algorithms::reverse_copy(begin(av), end(av), begin(result_av));

            for (int i = 0; i < int(vec.size()); ++i)
            {
                Assert::AreEqual((size - 1 - i), result_av[i]);
            }

            Assert::AreEqual(result_av[size - 1], *--result_end);
        }

        //----------------------------------------------------------------------------
        // rotate_copy
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_rotate_copy, "stl")
        {
            test_rotate_copy(1023, 200);
            test_rotate_copy(1, 0);
            test_rotate_copy(1024, 713);
        }

        void test_rotate_copy(const int size, const int middle_offset)
        {
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            std::vector<int> result(size, 0);
            concurrency::array_view<int> result_av(size, result);
            std::vector<int> expected_result(size, 0);
            auto expected_end = std::rotate_copy(begin(vec), begin(vec) + middle_offset, end(vec), begin(expected_result));

            auto result_end = amp_stl_algorithms::rotate_copy(begin(av), begin(av) + middle_offset, end(av), begin(result_av));

            Assert::IsTrue(are_equal(expected_result, result_av));
            Assert::AreEqual((size_t)std::distance(begin(expected_result), expected_end), (size_t)std::distance(begin(av), result_end));
        }

        //----------------------------------------------------------------------------
        // sort, partial_sort, partial_sort_copy, stable_sort, is_sorted, is_sorted_until
        //----------------------------------------------------------------------------

        // Special comparison operator for testing purposes.

        template <typename T>
        class abs_less_equal
        {
        public:
            T operator()(const T &a, const T &b) const restrict(cpu, amp)
            {
                return (abs(a) <= abs(b));
            }

        private:
            static T abs(const T& a)  restrict(cpu, amp) 
            { 
                return (a < 0) ? -a : a; 
            }
        };

        TEST_METHOD_CATEGORY(stl_is_sorted, "stl")
        {
            const int size = 10;
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            Assert::AreEqual(0, std::distance(begin(av), amp_stl_algorithms::is_sorted_until(begin(av), begin(av))));
            Assert::AreEqual(1, std::distance(begin(av), amp_stl_algorithms::is_sorted_until(begin(av), begin(av)+1)));

            Assert::AreEqual(9, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av), amp_algorithms::less<int>()));
            Assert::AreEqual(9, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av)));
            Assert::IsTrue(amp_stl_algorithms::is_sorted(begin(av), end(av), amp_algorithms::less<int>()));
            Assert::IsTrue(amp_stl_algorithms::is_sorted(begin(av), end(av)));

            av[5] = -4;  // 0, 1, 2, 3, 4, -4, 6, 7, 8, 9

            Assert::AreEqual(9, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av), abs_less_equal<int>()));
            Assert::AreEqual(4, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av)));
            Assert::IsTrue(amp_stl_algorithms::is_sorted(begin(av), end(av), abs_less_equal<int>()));
            Assert::IsFalse(amp_stl_algorithms::is_sorted(begin(av), end(av)));
            Assert::IsTrue(amp_stl_algorithms::is_sorted(begin(av), begin(av) + 5));

            av[5] = 4;   // 0, 1, 2, 3, 4, 4, 6, 7, 8, 9

            Assert::AreEqual(9, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av), amp_algorithms::less_equal<int>()));
            Assert::AreEqual(9, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av)));
            Assert::IsFalse(amp_stl_algorithms::is_sorted(begin(av), end(av), amp_algorithms::less<int>()));
            Assert::IsTrue(amp_stl_algorithms::is_sorted(begin(av), end(av)));

            av[1] = -1;  // 0, -1, 2, 3, 4, 4, 6, 7, 8, 9

            Assert::AreEqual(9, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av), abs_less_equal<int>()));
            Assert::AreEqual(0, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av), amp_algorithms::less_equal<int>()));
            Assert::AreEqual(0, *--amp_stl_algorithms::is_sorted_until(begin(av), end(av)));
            Assert::IsFalse(amp_stl_algorithms::is_sorted(begin(av), end(av), amp_algorithms::less_equal<int>()));
            Assert::IsFalse(amp_stl_algorithms::is_sorted(begin(av), end(av)));
            Assert::IsTrue(amp_stl_algorithms::is_sorted(begin(av), begin(av) + 1));
        }

        //----------------------------------------------------------------------------
        // swap, swap<T, N>, swap_ranges, iter_swap
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_swap_cpu, "stl")
        {
            int a = 1;
            int b = 2;

            amp_stl_algorithms::swap(a, b);

            Assert::AreEqual(2, a);
            Assert::AreEqual(1, b);
        }

        TEST_METHOD_CATEGORY(stl_swap_amp, "stl")
        {
            const int size = 2;
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 1);
            array_view<int> av(2, vec);

            parallel_for_each(concurrency::extent<1>(1), [=](concurrency::index<1> idx) restrict(amp)
            {
               amp_stl_algorithms::swap(av[idx], av[idx + 1]);
            });

            Assert::AreEqual(2, av[0]);
            Assert::AreEqual(1, av[1]);
        }

        TEST_METHOD_CATEGORY(stl_swap_n_cpu, "stl")
        {
            const int size = 10;
            int arr1[size];
            int arr2[size];
            std::iota(arr1, arr1 + size, 0);
            std::iota(arr2, arr2 + size, -9);

            amp_stl_algorithms::swap<int, 10>(arr1, arr2);

            for (int i = 0; i < size; ++i)
            {
                Assert::AreEqual(i, arr2[i]);
                Assert::AreEqual((-9 + i), arr1[i]);
            }
        }
      
        TEST_METHOD_CATEGORY(stl_swap_n_amp, "stl")
        {
            std::array<int, 10> expected = { 6, 7, 8, 9, 10, 1, 2, 3, 4, 5 };

            std::vector<int> vec(10);
            std::iota(begin(vec), end(vec), 1);
            array_view<int> av(10, vec);

            parallel_for_each(concurrency::tiled_extent<5>(concurrency::extent<1>(5)), 
                [=](concurrency::tiled_index<5> tidx) restrict(amp)
            {
                tile_static int arr1[5];
                tile_static int arr2[5];

                int idx = tidx.global[0];
                int i = tidx.local[0];
            
                arr1[i] = av[i];
                arr2[i] = av[i + 5];

                tidx.barrier.wait();

                if (i == 0)
                {
                    amp_stl_algorithms::swap<int, 5>(arr1, arr2);
                }

                tidx.barrier.wait();

                av[i] = arr1[i];
                av[i + 5] = arr2[i];

                tidx.barrier.wait();
            });

            av.synchronize();
            Assert::IsTrue(are_equal(expected, av));
        }
        
        TEST_METHOD_CATEGORY(stl_swap_ranges, "stl")
        {
            const int size = 10;
            std::array<int, size> expected = { 0, 6, 7, 8, 4, 5, 1, 2, 3, 9 };
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 0);
            array_view<int> av(size, vec);

            auto expected_end = amp_stl_algorithms::swap_ranges(begin(av) + 1, begin(av) + 4, begin(av) + 6);

            Assert::IsTrue(are_equal(expected, av));
            Assert::AreEqual(0, std::distance(begin(av) + 6 + 3, expected_end));
            Assert::AreEqual(3, *--expected_end);
        }

        TEST_METHOD_CATEGORY(stl_swap_iter, "stl")
        {
            const int size = 2;
            std::vector<int> vec(size);
            std::iota(begin(vec), end(vec), 1);
            array_view<int> av(2, vec);

            parallel_for_each(concurrency::extent<1>(1), [=](concurrency::index<1> idx) restrict(amp)
            {
                amp_stl_algorithms::iter_swap(begin(av), begin(av) + 1);
            });

            Assert::AreEqual(2, av[0]);
            Assert::AreEqual(1, av[1]);
        }

        //----------------------------------------------------------------------------
        // transform
        //----------------------------------------------------------------------------

        TEST_METHOD_CATEGORY(stl_unary_transform, "stl")
        {
            const int size = 1024;
            std::vector<int> vec_in(size);
            std::fill(begin(vec_in), end(vec_in), 7);
            array_view<const int> av_in(size, vec_in);

            std::vector<int> vec_out(size);
            array_view<int> av_out(size, vec_out);

            // Test "transform" by doubling the input elements

            amp_stl_algorithms::transform(begin(av_in), end(av_in), begin(av_out), [] (int x) restrict(amp) 
            {
                return 2 * x;
            });
            av_out.synchronize();

            for (auto element : vec_out)
            {
                Assert::AreEqual(2 * 7, element);
            }
        }

        TEST_METHOD_CATEGORY(stl_binary_transform, "stl")
        {
            const int size = 1024;

            std::vector<int> vec_in1(size);
            std::fill(begin(vec_in1), end(vec_in1), 343);
            array_view<const int> av_in1(size, vec_in1);

            std::vector<int> vec_in2(size);
            std::fill(begin(vec_in2), end(vec_in2), 323);
            array_view<const int> av_in2(size, vec_in2);

            std::vector<int> vec_out(size);
            array_view<int> av_out(size, vec_out);

            // Test "transform" by adding the two input elements

            amp_stl_algorithms::transform(begin(av_in1), end(av_in1), begin(av_in2), begin(av_out), [] (int x1, int x2) restrict(amp) {
                return x1 + x2;
            });
            av_out.synchronize();

            for (auto element : vec_out)
            {
                Assert::AreEqual(343 + 323, element);
            }
        }
    };
};// namespace tests
