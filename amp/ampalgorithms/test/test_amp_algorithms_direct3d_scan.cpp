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
* This file contains the unit tests for scan.
*---------------------------------------------------------------------------*/

#include "stdafx.h"

#include <amp_algorithms_direct3d.h>
#include "testtools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace concurrency;
using namespace amp_algorithms::direct3d;
using namespace testtools;

namespace amp_algorithms_direct3d_details_tests
{
    // This tests an inlined function so don't exclude this from coverage.
    TEST_CLASS(amp_direct3d_details_tests)
    {
        TEST_METHOD_CATEGORY(amp_details_check_hresult, "amp::direct3d")
        {
            try
            {
                amp_algorithms::direct3d::_details::_check_hresult(E_FAIL, "Failed!");
            }
            catch (runtime_exception& ex)
            {
                Assert::AreEqual(E_FAIL, ex.get_error_code());
                Assert::AreEqual("Failed! 0x80004005.", ex.what());
            }
        }
    };
};

namespace amp_algorithms_direct3d_tests
{
    enum class scan_type
    {
        scan,
        multiscan,
        segmented
    };

    struct bitvector;

    TEST_CLASS_CATEGORY(amp_direct3d_scan_tests, "amp::direct3d")
    // {
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            set_default_accelerator(L"amp_direct3d_scan_tests");
        }

        TEST_METHOD_INITIALIZE(flush)
        {
            accelerator().default_view.wait();
        }

        TEST_METHOD(amp_dx_scan_backwards)
        {
            test_scan<float>(amp_algorithms::scan_direction::backward);
            test_scan<unsigned int>(amp_algorithms::scan_direction::backward);
            test_scan<int>(amp_algorithms::scan_direction::backward);
            test_scan_bitwise_op<int>(amp_algorithms::scan_direction::backward);
        }

        TEST_METHOD(amp_dx_scan_forwards)
        {
            test_scan<float>(amp_algorithms::scan_direction::forward);
            test_scan<unsigned int>(amp_algorithms::scan_direction::forward);
            test_scan<int>(amp_algorithms::scan_direction::forward);
            test_scan_bitwise_op<int>(amp_algorithms::scan_direction::forward);
        }

        TEST_METHOD(amp_dx_multiscan_backwards)
        {
            test_multiscan<int>(amp_algorithms::scan_direction::backward);
            test_multiscan<unsigned int>(amp_algorithms::scan_direction::backward);
            test_multiscan<float>(amp_algorithms::scan_direction::backward);
            test_multiscan_bitwise_op<int>(amp_algorithms::scan_direction::backward);
        }

        TEST_METHOD(amp_dx_multiscan_forwards)
        {
            test_multiscan<int>(amp_algorithms::scan_direction::forward);
            test_multiscan<unsigned int>(amp_algorithms::scan_direction::forward);
            test_multiscan<float>(amp_algorithms::scan_direction::forward);
            test_multiscan_bitwise_op<int>(amp_algorithms::scan_direction::forward);
        }

        TEST_METHOD(amp_dx_segmented_scan_backwards)
        {
            test_segmented<int>(amp_algorithms::scan_direction::backward);
            test_segmented<unsigned int>(amp_algorithms::scan_direction::backward);
            test_segmented<float>(amp_algorithms::scan_direction::backward);
            test_segmented_bitwise_op<int>(amp_algorithms::scan_direction::backward);
        }

        TEST_METHOD(amp_dx_segmented_scan_forwards)
        {
            test_segmented<int>(amp_algorithms::scan_direction::forward);
            test_segmented<unsigned int>(amp_algorithms::scan_direction::forward);
            test_segmented<float>(amp_algorithms::scan_direction::forward);
            test_segmented_bitwise_op<int>(amp_algorithms::scan_direction::forward);
        }

        TEST_METHOD(amp_dx_scan_other)
        {
            const int elem_count = 10;
            std::vector<unsigned int> in(elem_count, 1);

            concurrency::array<unsigned int> input(concurrency::extent<1>(elem_count), in.begin());
            // use max_scan_size and max_scan_count greater than actual usage
            amp_algorithms::direct3d::scan s(2 * elem_count, elem_count);

            // 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ->
            s.scan_exclusive(input, input);
            // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ->
            s.scan_exclusive(input, input, amp_algorithms::scan_direction::forward, amp_algorithms::plus<unsigned int>());
            // 0, 0, 1, 3, 6, 10, 15, 21, 28, 36 ->

            unsigned int flg = 8; // 001000 in binary, so our segment is in here: 0, 0, 1, | 3, 6, 10, 15, 21, 28, 36
            concurrency::array<unsigned int> flags(1, &flg);
            s.segmented_scan_exclusive(input, input, flags, amp_algorithms::scan_direction::backward, amp_algorithms::plus<unsigned int>());
            // 1, 1, 0, 116, 110, 100, 85, 64, 36, 0

            // Copy out
            in = input;

            unsigned int expected_results[elem_count] = {1, 1, 0, 116, 110, 100, 85, 64, 36, 0};
            for (unsigned int i = 0; i < in.size(); ++i)
            {
                Assert::AreEqual(expected_results[i], in[i]);
            }
        }

        TEST_METHOD(amp_dx_scan_error_handling)
        {
            accelerator ref(accelerator::direct3d_ref);
            accelerator_view ref_view = ref.create_view();

            const int elem_count = 10;
            std::vector<unsigned int> in(elem_count, 1);

            concurrency::array<unsigned int> input(concurrency::extent<1>(elem_count), in.begin(), ref_view);

            Assert::ExpectException<runtime_exception>([&]() {
                amp_algorithms::direct3d::scan s2(2 * elem_count, elem_count, accelerator().default_view);
                s2.scan_exclusive(input, input);
            }, 
                L"Expected exception for non-matching accelerator_view in scan object");

            Assert::ExpectException<runtime_exception>([&]() {
                amp_algorithms::direct3d::scan s2(2 * elem_count, elem_count, ref_view);
                concurrency::array<unsigned int> output(elem_count, ref.create_view());
                s2.scan_exclusive(input, output);
            },
                L"Expected exception for non-matching accelerator_view in output");

            Assert::ExpectException<runtime_exception>([&]() {
                amp_algorithms::direct3d::scan s2(elem_count - 1, ref_view);
                s2.scan_exclusive(input, input);
            },
                L"Expected exception for scan object with max_scan_size < scan_size");

            Assert::ExpectException<runtime_exception>([&]() {
                amp_algorithms::direct3d::scan s2(elem_count, 0, ref_view);
            },
                L"Expected exception for scan object with max_scan_count == 0");

            Assert::ExpectException<runtime_exception>([&]() {
                amp_algorithms::direct3d::scan s2(elem_count, 1, ref_view);
                concurrency::array<unsigned int, 2> in2(10, 10);
                s2.multi_scan_exclusive(in2, in2, amp_algorithms::scan_direction::forward, amp_algorithms::plus<unsigned int>());
            },
                L"Expected exception for scan object with max_scan_count < scan_count");

            // Check scan binding cleanup
            array_view<const unsigned int> view(input);
            concurrency::array<unsigned int> output(input.extent, input.accelerator_view);
            parallel_for_each(output.extent, [view, &output] (concurrency::index<1> idx) restrict(amp) 
            {
                output[idx] = view[idx];
            });
            output.accelerator_view.wait();
        }

    private:
        template<typename T, typename BinaryFunction>
        void test_scan_internal(int column_count, BinaryFunction op, std::string test_name, 
            amp_algorithms::scan_direction direction, bool inplace, scan_type test_type = scan_type::scan, unsigned int row_count = 1)
        {
#ifdef _DEBUG
            column_count = std::min(200, column_count);
            row_count = std::min(200u, row_count);
#endif
            accelerator().default_view.wait();
            Logger::WriteMessage(get_extended_test_name<T, BinaryFunction>(test_name, direction, inplace).c_str());

            std::vector<T> in(row_count * column_count);
            generate_data(in);
            std::vector<T> out(row_count * column_count);
            amp_algorithms::direct3d::bitvector flags(column_count);

            // Construct scan object
            amp_algorithms::direct3d::scan s(column_count, row_count);

            // Run scan
            if (test_type == scan_type::multiscan)
            {
                concurrency::extent<2> e2(row_count, column_count);
                concurrency::array<T, 2> input(e2, in.begin());
                concurrency::array<T, 2> output(e2);

                s.multi_scan_exclusive(input, inplace ? input : output, direction, op);
                copy(inplace ? input : output, out.begin());
            }
            else
            {
                concurrency::extent<1> e(column_count);
                concurrency::array<T> input(e, in.begin());
                concurrency::array<T> output(e);

                if (test_type == scan_type::scan)
                {
                    s.scan_exclusive(input, inplace ? input : output, direction, op);
                }
                else
                {
                    flags.initialize(random_segments<int>());
                    concurrency::array<unsigned int> input_flags(static_cast<unsigned int>(flags.data.size()), flags.data.begin());
                    s.segmented_scan_exclusive(input, inplace ? input : output, input_flags, direction, op);
                }

                copy(inplace ? input : output, out.begin());
            }

            // Now time to verify the results with host-side computation
            verify_scan_results(direction, /*exclusive=*/true, op, in, out, column_count, column_count, row_count, flags);
        }

        template<typename T>
        void test_scan(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<T>(10, amp_algorithms::plus<T>(), "Test scan", direction, /*in_place=*/ false);
            test_scan_internal<T>(11, amp_algorithms::max<T>(), "Test scan", direction, /*in_place=*/ true);
            test_scan_internal<T>(2 * 1024, amp_algorithms::min<T>(), "Test scan", direction, /*in_place=*/ false);
            test_scan_internal<T>(2 * 1024 + 1, amp_algorithms::multiplies<T>(), "Test scan", direction, /*in_place=*/ false);
        }

        template<>
        void test_scan<unsigned int>(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<unsigned int>(10, amp_algorithms::plus<unsigned int>(), "Test scan", direction, /*in_place=*/ true);
            test_scan_internal<unsigned int>(777, amp_algorithms::multiplies<unsigned int>(), "Test scan", direction, /*in_place=*/ false);
        }

        template<typename T>
        void test_scan_bitwise_op(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<T>(77, amp_algorithms::bit_xor<T>(), "Test scan", direction, /*in_place=*/ true);
            test_scan_internal<T>(66, amp_algorithms::bit_and<T>(), "Test scan", direction, /*in_place=*/ false);
            test_scan_internal<T>(12345, amp_algorithms::bit_or<T>(), "Test scan", direction, /*in_place=*/ true);
        }

        template<typename T>
        void test_multiscan(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<T>(144, amp_algorithms::plus<T>(), "Test multiscan", direction, /*in_place=*/ false, scan_type::multiscan, 12);
            test_scan_internal<T>(2048, amp_algorithms::plus<T>(), "Test multiscan", direction, /*in_place=*/ true, scan_type::multiscan, 1024);
            test_scan_internal<T>(3333, amp_algorithms::max<T>(), "Test multiscan", direction, /*in_place=*/ false, scan_type::multiscan, 3);
            test_scan_internal<T>(128, amp_algorithms::multiplies<T>(), "Test multiscan", direction, /*in_place=*/ true, scan_type::multiscan, 8);
            test_scan_internal<T>(127, amp_algorithms::min<T>(), "Test multiscan", direction, /*in_place=*/ false, scan_type::multiscan, 3);
        }

        template<>
        void test_multiscan<unsigned int>(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<unsigned int>(144, amp_algorithms::plus<unsigned int>(), "Test multiscan", direction, /*in_place=*/ false, scan_type::multiscan, 12);
            test_scan_internal<unsigned int>(2048, amp_algorithms::plus<unsigned int>(), "Test multiscan", direction, /*in_place=*/ true, scan_type::multiscan, 64);
            test_scan_internal<unsigned int>(128, amp_algorithms::multiplies<unsigned int>(), "Test multiscan", direction, /*in_place=*/ false, scan_type::multiscan, 8);
        }

        template<typename T>
        void test_multiscan_bitwise_op(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<T>(8, amp_algorithms::bit_and<T>(), "Test multiscan", direction, /*in_place=*/ true, scan_type::multiscan, 2);
            test_scan_internal<T>(2048, amp_algorithms::bit_or<T>(), "Test multiscan", direction, /*in_place=*/ false, scan_type::multiscan, 2);
            test_scan_internal<T>(3072, amp_algorithms::bit_xor<T>(), "Test multiscan",  direction, /*in_place=*/ true, scan_type::multiscan, 3);
        }

        template<typename T>
        void test_segmented(amp_algorithms::scan_direction direction)
        {
            //test_scan_internal<T>(7123127, amp_algorithms::plus<T>(), "Test segmented scan", direction, /*inplace=*/false, scan_type::segmented);
            test_scan_internal<T>(31, amp_algorithms::multiplies<T>(), "Test segmented scan", direction, /*inplace=*/true, scan_type::segmented);
            test_scan_internal<T>(222, amp_algorithms::min<T>(), "Test segmented scan", direction, /*inplace=*/false, scan_type::segmented);
            test_scan_internal<T>(333, amp_algorithms::max<T>(), "Test segmented scan", direction, /*inplace=*/true, scan_type::segmented);
        }

        template<>
        void test_segmented<unsigned int>(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<unsigned int>(7123127, amp_algorithms::plus<unsigned int>(), "Test segmented scan", direction, /*inplace=*/false, scan_type::segmented);
            test_scan_internal<unsigned int>(111, amp_algorithms::multiplies<unsigned int>(), "Test segmented scan", direction, /*inplace=*/true, scan_type::segmented);
        }

        template<typename T>
        void test_segmented_bitwise_op(amp_algorithms::scan_direction direction)
        {
            test_scan_internal<T>(234, amp_algorithms::bit_and<T>(), "Test segmented scan", direction, /*inplace=*/false, scan_type::segmented);
            test_scan_internal<T>(432, amp_algorithms::bit_or<T>(), "Test segmented scan", direction, /*inplace=*/true, scan_type::segmented);
            test_scan_internal<T>(444, amp_algorithms::bit_xor<T>(), "Test segmented scan", direction, /*inplace=*/true, scan_type::segmented);
        }

        // A host side verification for scan, multiscan and segmented scan
        template <typename T, typename BinaryFunction>
        void verify_scan_results(amp_algorithms::scan_direction direction, bool exclusive, BinaryFunction op, std::vector<T> &in, std::vector<T> &out, 
            unsigned int scan_size, unsigned int scan_pitch, unsigned int scan_count, amp_algorithms::direct3d::bitvector &flags)
        {
            // For each sub-scan
            for (unsigned int current_scan_num = 0; current_scan_num < scan_count; ++current_scan_num)
            {
                T expected_scan_result = T();
                for (unsigned int i=current_scan_num * scan_pitch; i < scan_size; ++i)
                {
                    int pos = i; // pos is used to reference into output and input arrays depending on the direction we go from the front or the back
                    if (direction == amp_algorithms::scan_direction::backward)
                    {
                        pos = current_scan_num * scan_pitch + scan_size - 1 - i;
                    }

                    if (i == current_scan_num * scan_pitch || flags.is_bit_set(pos, direction))
                    {
                        // Establish first result, either it is identity (for exclusive) or first/last element depending on scan direction for inclusive scan
                        if (exclusive)
                        {
                            expected_scan_result = get_identity<T>(op);
                        }
                        else
                        {
                            expected_scan_result = in[pos];
                        }
                    }

                    // Inclusive is computed as pre-fix op
                    if (!exclusive && i != current_scan_num * scan_pitch)
                    {
                        expected_scan_result = op(expected_scan_result, in[pos]);
                    }

                    // Compare results
                    Assert::IsTrue(compare(out[pos], expected_scan_result));

                    // Exclusive is computed as post-fix op
                    if (exclusive)
                    {
                        expected_scan_result = op(expected_scan_result, in[pos]);
                    }
                }
            }
        }

        template<typename T, typename BinaryFunction>
        std::string get_extended_test_name(const std::string& test_name, const amp_algorithms::scan_direction direction, const bool inplace)
        {
            std::stringstream postfix;
            postfix << test_name << " ";
            postfix << typeid(T).name() << " ";
            postfix << get_binary_function_info<BinaryFunction>().name();

            postfix << (direction == amp_algorithms::scan_direction::backward ? " backward" : " forward");
            postfix << (inplace ? " in-place" : " not-in-place");

            return postfix.str();
        }

        template<typename BinaryFunction>
        struct get_binary_function_info
        {
            std::string name() { return "Unrecognized function"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::plus<T>>
        {
            std::string name() { return "plus"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::min<T>>
        {
            std::string name() { return "min"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::max<T>>
        {
            std::string name() { return "max"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::multiplies<T>>
        {
            std::string name() { return "multiplies"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::bit_and<T>>
        {
            std::string name() { return "bitwise and"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::bit_or<T>>
        {
            std::string name() { return "bitwise or"; }
        };

        template<typename T>
        struct get_binary_function_info<amp_algorithms::bit_xor<T>>
        {
            std::string name() { return "bitwise xor"; }
        };
    };

    template<typename value_type, typename functor>
    value_type get_identity(functor op)
    {
        if (typeid(op) == typeid(amp_algorithms::max<value_type>))
        {
            return std::numeric_limits<value_type>::lowest();
        }
        if (typeid(op) == typeid(amp_algorithms::min<value_type>))
        {
            return std::numeric_limits<value_type>::max();
        }
        if (typeid(op) == typeid(amp_algorithms::multiplies<value_type>))
        {
            return 1;
        }
        if (typeid(op) == typeid(amp_algorithms::bit_and<value_type>))
        {
            return (value_type)(0xFFFFFFFF);
        }
        else
        {
            return 0;
        }
    }

    template <typename T>
    class random_segments
    {
    public:
        random_segments()
        {
            srand(2012);    // Set random number seed so tests are reproducible.
        }

        bool operator()(const T &i) const
        {
            return (rand() % 13 == 0) ? 1 : 0;
        }
    };
};
