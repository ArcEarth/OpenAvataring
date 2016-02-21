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

#include <amp_algorithms.h>
#include <xx_amp_algorithms_impl_inl.h>
#include <amp_stl_algorithms.h>

#include "testtools.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

using namespace concurrency;
using namespace amp_stl_algorithms;

namespace examples
{
    TEST_CLASS_CATEGORY(amp_examples, "examples")
    // {
        TEST_CLASS_INITIALIZE(initialize_tests)
        {
            testtools::set_default_accelerator(L"stl_examples");
        }

        TEST_METHOD_INITIALIZE(initialize_test)
        {
            accelerator().default_view.wait();
        }

        // SAXPY functor (for both C++ and C++ AMP.

        struct saxpy_functor 
        {
            const float a;

            saxpy_functor(float _a) : a(_a) {}

            float operator()(const float& x, const float& y) const restrict(amp, cpu)
            {
                return a * x + y;
            }
        };

        // Calculate SAXPY, y = a * x + y, for all elements in an array.

        BEGIN_TEST_METHOD_ATTRIBUTE(stl_example_saxpy)
            TEST_CATEGORY("stl")
#if (defined(USE_REF) || defined(_DEBUG))
            TEST_IGNORE()
#endif
            END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(stl_example_saxpy)
        {
            auto size = testtools::test_array_size<int>();

            {
                const float a = 2.0f;
                concurrency::array<float> x(size);
                array_view<float> x_av(x);
                amp_stl_algorithms::iota(begin(x_av), end(x_av), 1.0f);
                concurrency::array<float> y(size);
                array_view<float> y_av(y);
                amp_stl_algorithms::iota(begin(y_av), end(y_av), 1.0f);

                amp_stl_algorithms::transform(begin(x_av), end(x_av), begin(y_av), begin(y_av), saxpy_functor(a));
            }

            // STL Equivalent code.
            {
                const float a = 2.0f;
                std::vector<float> x(size);
                std::iota(std::begin(x), std::end(x), 1.0f);
                std::vector<float> y(size);
                std::iota(std::begin(y), std::end(y), 1.0f);

                std::transform(begin(x), end(x), begin(y), begin(y), saxpy_functor(a));
            }
        }

        // Sum of the number of even numbers in an array.

        BEGIN_TEST_METHOD_ATTRIBUTE(stl_example_hello_world)
            TEST_CATEGORY("stl")
#if (defined(USE_REF) || defined(_DEBUG))
            TEST_IGNORE()
#endif
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(stl_example_hello_world)
        {
            auto size = testtools::test_array_size<int>();

            {
                concurrency::array<float> data(size);
                array_view<float> data_av(data);
                amp_stl_algorithms::iota(begin(data_av), end(data_av), 1.0f);

                auto last = amp_stl_algorithms::remove_if(begin(data_av), end(data_av), 
                    [=](const float& v) restrict(amp) { return int(v) % 2 == 1; });
                float total = amp_stl_algorithms::reduce(begin(data_av), last, 0.0f);

                std::stringstream str;
                str << "AAL: Sum of all even numbers in the input array = " << total;
                Logger::WriteMessage(str.str().c_str());
            }

            // STL Equivalent code.
            {
                std::vector<float> data(size);
                std::iota(std::begin(data), std::end(data), 1.0f);

                auto last = std::remove_if(std::begin(data), std::end(data), 
                    [=](const float& v) { return int(v) % 2 == 1; });
                float total = std::accumulate(begin(data), last, 0.0f);

                std::stringstream str;
                str << "STD: Sum of all even numbers in the input array = " << total;
                Logger::WriteMessage(str.str().c_str());
            }
        }

        // Calculate the volume of a set of randomly generated tetrahedrons each with one vertex at (0, 0, 0).

        BEGIN_TEST_METHOD_ATTRIBUTE(stl_example_map_reduce)
            TEST_CATEGORY("stl")
#if (defined(USE_REF) || defined(_DEBUG))
            TEST_IGNORE()
#endif
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(stl_example_map_reduce)
        {
            struct vertices
            {
                float x1, y1, z1;
                float x2, y2, z2;
                float x3, y3, z3;

                vertices() { };

                vertices(std::uniform_real_distribution<float> dist, std::mt19937 gen) : 
                    x1(dist(gen)), x2(dist(gen)), x3(dist(gen)), 
                    y1(dist(gen)), y2(dist(gen)), y3(dist(gen)),
                    z1(dist(gen)), z2(dist(gen)), z3(dist(gen))
                { };
            };

            std::random_device rnd_dev;
            std::mt19937 rnd_gen(rnd_dev());
            std::uniform_real_distribution<float> rnd_dist(0.5f, 1.5f);
            std::vector<vertices> tetrahedrons(1000);
            std::generate(begin(tetrahedrons), end(tetrahedrons), [&rnd_dist, &rnd_gen] { return vertices(rnd_dist, rnd_gen); });

            array_view<const vertices, 1> tetrahedrons_av(static_cast<int>(tetrahedrons.size()), tetrahedrons.data());
            concurrency::array<float, 1> volumes_arr(static_cast<int>(tetrahedrons.size()));
            array_view<float, 1> volumes_vw(volumes_arr);
            amp_stl_algorithms::transform(begin(tetrahedrons_av), end(tetrahedrons_av), begin(volumes_vw), 
                [=](const vertices& t) restrict(amp)
            {
                return t.x1 * (t.y2 * t.z3 - t.y3 * t.z2)
                     + t.y1 * (t.z2 * t.x3 - t.x2 * t.z3)
                     + t.z1 * (t.x2 * t.y3 - t.x3 * t.y2);
            });
            float sum = amp_stl_algorithms::reduce(begin(volumes_vw), end(volumes_vw), 0.0f);

            std::stringstream str;
            str << "STD: Total volume of tetrahedrons = " << sum;
            Logger::WriteMessage(str.str().c_str());
        }
    };

};// namespace examples
