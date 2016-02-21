//--------------------------------------------------------------------------------------
// File: main.cpp
//
// Demonstrates how to use C++ AMP FFT library
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <cmath>
#include "amp_fft.h"
#include <iostream>

using namespace concurrency;

const float ALLOWED_ERROR_RATIO = 0.00001f;
bool compare_with_error_margin(const float &actual, const float &expected) 
{
    float actual_error_ratio = fabs((actual - expected)/expected);
    if (actual_error_ratio > ALLOWED_ERROR_RATIO) {
        return false;
    }

    return true;
}

bool compare_with_error_margin(const std::complex<float> &actual, const std::complex<float> &expected) 
{
    float actual_real_error_ratio = fabs((actual.real() - expected.real())/expected.real());
    float actual_imag_error_ratio = fabs((actual.imag() - expected.imag())/expected.imag());

    if ((actual_real_error_ratio > ALLOWED_ERROR_RATIO) || (actual_imag_error_ratio > ALLOWED_ERROR_RATIO)) {
        return false;
    }

    return true;
}

template <typename value_type>
void verify_results(const std::vector<value_type> &actual, const std::vector<value_type> &expected)
{
    bool passed = true;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        if (!compare_with_error_margin(actual[i], expected[i]))
        {
            if (passed) {
                std::cout << "Incorrect result for FFT!\n";
            }
            passed = false;
            std::cout << "Expected: " << expected[i] << ", Actual: " << actual[i] << "\n";
        }
    }
}

template <int dims>
void test_fft_real(bool inPlace = false)
{
    extent<dims> e;
    if (dims == 1) { e[0] = 10000; }
    if (dims == 2) { e[0] = 100; e[1] = 100; }
    if (dims == 3) { e[0] = 10; e[1] = 10; e[2] = 100; }

    // Create the FFT transformation object
    fft<float, dims> transform(e);

    // Initialize some input
    std::vector<float> input_vec(10000);
    for (int y = 0; y < 100; y++)
    {
        for (int x = 0; x < 100; x++)
        {
            input_vec[y*100 + x] = static_cast<float>(10.0f * sin(x*0.05f) * cos(y*0.1f) + 10.0f * log(2.0f + x + y) + cos(x));
        }
    }

    // Copy the input to the accelerator
    array<float, dims> in_array(e, input_vec.begin());

    std::vector<float> output_vec;

    if (inPlace)
    {
        // Apply the forward transformation
        array<std::complex<float>, dims> transformed_array(e);
        transform.forward_transform(in_array, transformed_array);

        // Now calculate the inverse transform, this should get us
        // back to the original input
        transform.inverse_transform(transformed_array, in_array);

        // Copy back to the CPU
        output_vec = in_array;
    }
    else
    {
        // Apply the forward transformation
        array<std::complex<float>, dims> transformed_array(e);
        transform.forward_transform(in_array, transformed_array);

        // Now calculate the inverse transform, this should get us
        // back to the original input
        array<float, dims> inverse_array(e);
        transform.inverse_transform(transformed_array, inverse_array);

        // Copy back to the CPU
        output_vec = inverse_array;
    }

    // Verify if the inverse transform results are same as the original input
    verify_results(output_vec, input_vec);
}

template <int dims>
void test_fft_complex(bool inPlace = false)
{
    extent<dims> e;
    if (dims == 1) { e[0] = 10000; }
    if (dims == 2) { e[0] = 100; e[1] = 100; }
    if (dims == 3) { e[0] = 10; e[1] = 10; e[2] = 100; }

    // Create the FFT transformation object
    fft<std::complex<float>, dims> transform(e);

    // Initialize some input
    std::vector<std::complex<float>> input_vec(10000);
    for (int y = 0; y < 100; y++)
    {
        for (int x = 0; x < 100; x++)
        {
            float value = static_cast<float>(10.0f * sin(x*0.05f) *cos(y*0.1f) + 10.0f * log(2.0f + x + y) + cos(x));
            input_vec[y*100 + x] = std::complex<float>(value, value/2.0f);
        }
    }

    // Copy the input to the GPU
    array<std::complex<float>,dims> in_array(e, input_vec.begin());

    std::vector<std::complex<float>> output_vec;
    if (inPlace)
    {
        // Apply the forward transformation
        transform.forward_transform(in_array, in_array);

        // Now calculate the inverse transform, this should get us
        // back to the original input
        transform.inverse_transform(in_array, in_array);

        // Copy back to the CPU
        output_vec = in_array;
    }
    else
    {
        // Apply the forward transformation
        array<std::complex<float>, dims> transformed_array(e);
        transform.forward_transform(in_array, transformed_array);

        // Now calculate the inverse transform, this should get us
        // back to the original input
        array<std::complex<float>, dims> inverse_array(e);
        transform.inverse_transform(transformed_array, inverse_array);

        // Copy back to the CPU
        output_vec = inverse_array;
    }

    // Verify if the inverse transform results are same as the original input
    verify_results(output_vec, input_vec);
}

void test_fft()
{
    // Test 1D real data
    test_fft_real<1>();

    // Test 2D real data in-place
    test_fft_real<2>(true);

    // Test 3D real data
    test_fft_real<3>();

    // Test 1D complex data in-place
    test_fft_complex<1>(true);

    // Test 2D complex data
    test_fft_complex<2>();

    // Test 3D complex data in-place
    test_fft_complex<3>(true);
}

template <int dims>
void fft_sample()
{
    extent<dims> e;
    if (dims == 1) { e[0] = 10000; }
    if (dims == 2) { e[0] = 100; e[1] = 100; }
    if (dims == 3) { e[0] = 10; e[1] = 10; e[2] = 100; }

    // Create the FFT transformation object
    fft<float,dims> transform(e);

    FILE * fdata = NULL;
	if (fopen_s(&fdata, "data.txt", "wt") != 0)
    {
        printf("open file data.txt file\n");
    }
    // Initialize some input
    std::vector<float> input_vec(10000);
    for (int y=0; y<100; y++)
    {
        for (int x=0; x<100; x++)
        {
            input_vec[y*100 + x] = static_cast<float>(10 * sin(x*0.05) *cos(y*0.1) + 10 * log(2 + x + y) + cos(x));
            fprintf_s(fdata, "%lf ", input_vec[y*100 + x]);
        }
        fprintf_s(fdata, "\n");
    }

    // Copy the input to the GPU
    array<float,dims> in_array(e, input_vec.begin());

    // apply the transformation
    array<std::complex<float>,dims> transformed_array(e);
    transform.forward_transform(in_array, transformed_array);

    // Copy the results back and print them
    std::vector<std::complex<float>> transformed_vec = transformed_array;
    for (int y=0; y<100; y++)
    {
        for (int x=0; x<100; x++)
        {
            fprintf_s(fdata, "%lf ", transformed_vec[y*100 + x].real());
            fprintf_s(fdata, "%lf ", transformed_vec[y*100 + x].imag());
        }
        fprintf_s(fdata, "\n");
    }

    // Now calculate the inverse transform, this should get us
    // back to the original input
    array<float,dims> inverse_array(e);
    transform.inverse_transform(transformed_array, inverse_array);

    // Copy back to the CPU and output
    std::vector<float> inverse_vec = inverse_array;
    for (int y=0; y<100; y++)
    {
        for (int x=0; x<100; x++)
        {
            fprintf_s(fdata, "%lf ", inverse_vec[y*100 + x]);
        }
        fprintf_s(fdata, "\n");
    }
    fclose(fdata);
}

int main(int argc, char **argv)
{
    test_fft();

    int number_of_dims = 1;
    if (argc == 2)
    {
        number_of_dims = atoi(argv[1]);
        if (number_of_dims < 1 ||  number_of_dims > 3)
        {
            printf("amp_fft [<number of dimensions=1,2 or 3>]\n");
            return 1;
        }
    }

    switch (number_of_dims)
    {
    case 1: fft_sample<1>(); break;
    case 2: fft_sample<2>(); break;
    case 3: fft_sample<3>(); break;
    }

    system("pause");
    return 0;
}
