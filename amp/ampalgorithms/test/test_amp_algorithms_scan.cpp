/*----------------------------------------------------------------------------
* Copyright (c) Microsoft Corp.
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
* C++ AMP standard algorithms library.
*
* This file contains the helpers classes in amp_algorithms::_details namespace
*---------------------------------------------------------------------------*/

#include "stdafx.h"

#include "testtools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace amp_algorithms;
using namespace testtools;

namespace amp_algorithms_tests
{
    std::wstring Msg(std::vector<int>& expected, std::vector<int>& actual, size_t width = 32)
    {
        std::wostringstream msg;
        msg << container_width(50) << L"[" << expected << L"] != [" << actual << L"]" << std::endl;
        return msg.str();
    }

    TEST_CLASS_CATEGORY(amp_scan_tests, "amp")
    // {

    private:

    static const int test_tile_size = 256;

    public:
        TEST_CLASS_INITIALIZE(initialize_tests)
        { 
            set_default_accelerator(L"amp_scan_tests");
        }

        TEST_METHOD_INITIALIZE(initialize_test)
        {
            accelerator().default_view.wait();
        }

        TEST_METHOD(amp_details_scan_tile_exclusive)
        {
            static const int tile_size = 4;

            std::array<unsigned, 16> input =        {  3,  2,  1,  6,   10, 11, 13,  1,   15, 10,  5, 14,    4, 12,  9,  8 };
            std::array<unsigned, 16> reduce =       {  3,  5,  1, 12,   10, 21, 13, 35,   15, 25,  5, 44,    4, 16,  9, 33 };
            std::array<unsigned, 16> zero_top =     {  3,  5,  1,  0,   10, 21, 13,  0,   15, 25,  5,  0,    4, 16,  9,  0 };
            std::array<unsigned, 16> expected =     {  0,  3,  5,  6,    0, 10, 21, 34,    0, 15, 25, 30,    0,  4, 16, 25 };

            array_view<unsigned> input_av(int(input.size()), input);
            std::array<unsigned, 16> output;
            array_view<unsigned> output_av(int(output.size()), output);
            amp_algorithms::fill(output_av, 0);
            concurrency::tiled_extent<tile_size> compute_domain = input_av.get_extent().tile<4>().pad();

            concurrency::parallel_for_each(compute_domain, [=](concurrency::tiled_index<tile_size> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int idx = tidx.local[0];
                tile_static int tile_data[tile_size];
                tile_data[idx] = input_av[gidx];

                amp_algorithms::_details::scan_tile_exclusive<tile_size>(tile_data, tidx, amp_algorithms::plus<int>(), tile_size);

                tidx.barrier.wait();
                output_av[gidx] = tile_data[idx];
            });

            Assert::IsTrue(are_equal(expected, output_av));
        }

        TEST_METHOD(amp_details_segment_scan_width_2)
        {
            std::array<unsigned, 16> input =    { 3, 2,   1, 6,   10, 11,   13,  1,   15, 10,    5, 14,     4,  12,     9,   8 };
            // exclusive scan                     0, 3,   5, 6,   12, 22,   33, 47,   62, 77,   87, 92,   106, 118,   127, 135
            std::array<unsigned, 16> expected = { 0, 3,   0, 1,    0, 10,    0, 13,    0, 15,    0,  5,     0,   4,    0,    9 };

            scan_sequential_exclusive(begin(input), end(input), begin(input));
            array_view<unsigned> input_av(int(input.size()), input);
            std::array<unsigned, 16> output;

            for (int i = 0; i < int(output.size()); ++i)
            {
                output[i] = amp_algorithms::_details::segment_exclusive_scan(input_av, 2, i);
            }

            Assert::IsTrue(are_equal(expected, output));
        }

        TEST_METHOD(amp_details_segment_scan_width_8)
        {
            std::array<unsigned, 16> input =    { 3, 2, 1, 6, 10, 11, 13,  1,   15, 10,  5, 14,   4,  12,   9,   8 };
            // exclusive scan                     0, 3, 5, 6, 12, 22, 33, 46,   62, 77, 87, 92, 106, 118, 127, 135
            std::array<unsigned, 16> expected = { 0, 3, 5, 6, 12, 22, 33, 46,    0, 15, 25, 30,  44,  48,  60,  69 };

            scan_sequential_exclusive(begin(input), end(input), begin(input));
            array_view<unsigned> input_av(int(input.size()), input);
            std::array<unsigned, 16> output;

            for (int i = 0; i < int(output.size()); ++i)
            {
                output[i] = amp_algorithms::_details::segment_exclusive_scan(input_av, 8, i);
            }

            Assert::IsTrue(are_equal(expected, output));
        }

        TEST_METHOD(amp_details_scan_tile_exclusive_partial)
        {
            std::array<unsigned, 7> input =    { 3, 2, 1, 6, 10, 11,  5 };
            std::array<unsigned, 7> expected = { 0, 3, 5, 6, 12, 22, 33 };

            array_view<unsigned> input_av(int(input.size()), input);
            std::array<unsigned, 7> output;
            array_view<unsigned> output_av(int(output.size()), output);
            amp_algorithms::fill(output_av, 0);
            concurrency::tiled_extent<16> compute_domain = input_av.get_extent().tile<16>().pad();

            concurrency::parallel_for_each(compute_domain, [=](concurrency::tiled_index<16> tidx) restrict(amp)
            {
                const int gidx = tidx.global[0];
                const int idx = tidx.local[0];
                tile_static int tile_data[16];
                tile_data[idx] = input_av[gidx];

                amp_algorithms::_details::scan_tile_exclusive<16>(tile_data, tidx, amp_algorithms::plus<int>(), 7);

                output_av[gidx] = tile_data[idx];
            });

            Assert::IsTrue(are_equal(expected, output_av));
        }

        TEST_METHOD(amp_scan_exclusive_multi_tile)
        {
            std::vector<int> input(test_tile_size * 4);
            //generate_data(input);
            std::iota(begin(input), end(input), 1);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_exclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::exclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            Assert::IsTrue(are_equal(expected, input_vw));
        }

        TEST_METHOD(amp_scan_exclusive_multi_tile_partial)
        {
            std::vector<int> input(test_tile_size * 4 + 4);
            generate_data(input);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_exclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::exclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            input_vw.synchronize();
            Assert::IsTrue(expected == input, Msg(expected, input).c_str());
        }

        TEST_METHOD(amp_scan_exclusive_recursive_scan)
        {
            std::vector<int> input(test_tile_size * (test_tile_size + 2));
            generate_data(input);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_exclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::exclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            input_vw.synchronize();
            Assert::IsTrue(expected == input, Msg(expected, input).c_str());
        }

        TEST_METHOD(amp_scan_exclusive)
        {
            std::vector<int> input(test_tile_size * (test_tile_size + 10));
            generate_data(input);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_exclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::exclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            input_vw.synchronize();
            Assert::IsTrue(expected == input, Msg(expected, input).c_str());
        }

        TEST_METHOD(amp_scan_inclusive_multi_tile)
        {
            std::vector<int> input(test_tile_size * 4);
            //generate_data(input);
            std::iota(begin(input), end(input), 1);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_inclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::inclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            Assert::IsTrue(are_equal(expected, input_vw));
        }

        TEST_METHOD(amp_scan_inclusive_multi_tile_partial)
        {
            std::vector<int> input(test_tile_size * 4 + 4, 1);
            generate_data(input);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_inclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::inclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            input_vw.synchronize();
            Assert::IsTrue(expected == input, Msg(expected, input).c_str());
        }
        
        TEST_METHOD(amp_scan_inclusive_recursive_scan)
        {
            std::vector<int> input(test_tile_size * (test_tile_size + 2));
            generate_data(input);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> expected(input.size());
            scan_sequential_inclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::inclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            input_vw.synchronize();
            Assert::IsTrue(expected == input, Msg(expected, input).c_str());
        }

        TEST_METHOD(amp_scan_inclusive)
        {
            std::vector<int> input(test_tile_size * (test_tile_size + 10), 1);
            generate_data(input);
            concurrency::array_view<int, 1> input_vw(int(input.size()), input);
            std::vector<int> result(input.size(), -1);
            std::vector<int> expected(input.size());
            scan_sequential_inclusive(begin(input), end(input), begin(expected));

            scan<test_tile_size, scan_mode::inclusive>(input_vw, input_vw, amp_algorithms::plus<int>());

            input_vw.synchronize();
            Assert::IsTrue(are_equal(expected, input_vw));
        }
    };
}
