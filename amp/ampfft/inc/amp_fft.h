//--------------------------------------------------------------------------------------
// File: amp_fft.h
//
// Header file for the C++ AMP wrapper over the Direct3D FFT API's.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include <d3d11.h>
#include <complex>
#include <amp.h>
#include <wrl\client.h>
#include <exception>

#include <d3dcsx.h>

#pragma comment(lib, "d3dcsx")

//----------------------------------------------------------------------------
// DLL export/import specifiers
//
// The export/import mechanism used here is the __declspec(export) method 
// supported by Microsoft Visual Studio, but any other export method supported
// by your development environment may be substituted.
//----------------------------------------------------------------------------
#ifndef AMP_FFT_DLL 
#define AMP_FFT_DLL __declspec(dllimport)
#else
#undef AMP_FFT_DLL
#define AMP_FFT_DLL __declspec(dllexport)
#endif

//--------------------------------------------------------------------------------------
// This exception type should be expected from all public methods in this header, 
// except for destructors.
//--------------------------------------------------------------------------------------
class fft_exception : public std::exception
{
public:
    explicit fft_exception(HRESULT error_code) throw() 
        : err_code(error_code) {}

    fft_exception(const char *const& msg, HRESULT error_code) throw()
        : err_msg(msg), err_code(error_code) {}

    fft_exception(const std::string& msg, HRESULT error_code) throw()
        : err_msg(msg), err_code(error_code) {}

    fft_exception(const fft_exception &other) throw()
        : std::exception(other), err_msg(other.err_msg), err_code(other.err_code) {}

    virtual ~fft_exception() throw() {}

    HRESULT get_error_code() const throw()
    {
        return err_code;
    }

    virtual const char *what() const throw()
    {
        return  err_msg.data();
    }

private:
    fft_exception &operator=(const fft_exception &);
    std::string err_msg;
    HRESULT err_code;
};
//--------------------------------------------------------------------------------------
// Implementation details, amp_fft.cpp contains the implementation for the functions
// declared in this namespace. 
//--------------------------------------------------------------------------------------
namespace _details
{
    template <typename _Type>
    struct dx_fft_type_helper
    {
        static const bool is_type_supported = false;
    };

    template <>
    struct dx_fft_type_helper<float>
    {
        static const bool is_type_supported = true;
        typedef float precision_type;
        static const D3DX11_FFT_DATA_TYPE dx_type = D3DX11_FFT_DATA_TYPE_REAL;
    };

    template <>
    struct dx_fft_type_helper<std::complex<float>>
    {
        static const bool is_type_supported = true;
        typedef float precision_type;
        static const D3DX11_FFT_DATA_TYPE dx_type = D3DX11_FFT_DATA_TYPE_COMPLEX;
    };

    class fft_base
    {
    public:
        AMP_FFT_DLL void set_forward_scale(float scale);
        AMP_FFT_DLL float get_forward_scale() const;
        AMP_FFT_DLL void set_inverse_scale(float scale);
        AMP_FFT_DLL float get_inverse_scale() const;

    protected:
        AMP_FFT_DLL fft_base(D3DX11_FFT_DATA_TYPE _Dx_type, int _Dim, const int* _Transform_extent, const concurrency::accelerator_view& _Av, float _Forward_scale, float _Inverse_scale);
        AMP_FFT_DLL HRESULT base_transform(bool _Forward, ID3D11Buffer *pBufferIn, ID3D11Buffer *pBufferOut) const;

    private:
        HRESULT create_raw_buffer(UINT floatSize, ID3D11Buffer **ppBuffer) const;
        HRESULT create_uav(UINT floatSize, ID3D11Buffer *ppBuffer, ID3D11UnorderedAccessView **ppUAV) const;
        HRESULT create_raw_buffer_and_uav(UINT floatSize, ID3D11Buffer **ppBuffer, ID3D11UnorderedAccessView **ppUAV) const;

        UINT _M_Total_float_size;
        D3DX11_FFT_DATA_TYPE _M_dx_type;
        Microsoft::WRL::ComPtr<ID3D11Device> _M_pDevice;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> _M_pDeviceContext;
        Microsoft::WRL::ComPtr<ID3DX11FFT> _M_pFFT;
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> _M_pTempUAVs[D3DX11_FFT_MAX_TEMP_BUFFERS];
        Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> _M_pPrecomputeUAVs[D3DX11_FFT_MAX_PRECOMPUTE_BUFFERS];
    };
} // namespace _details

//--------------------------------------------------------------------------------------
// class fft.
//
// This is the class which provides the FFT transformation functionality. At this point
// it exposes 1d, 2d and 3d transformations over floats or std::complex<float>, although
// Direct3D also provides the ability to transform higher dimensions, or be more 
// selective about which dimensions to transform. Such an extension is left for the 
// future or to the interested reader...
//
// After creating an instance, you can use forward_transform and backward_tranform to
// transform your data. The constructor initializes some internal data structures, so 
// it's beneficial to reuse the fft object as long as possible.
//
// Note that the dimensions (extent) of the fft and the extent of all input and output 
// arrays must be identical. If an array is used which has a different extent, an 
// fft_exception is thrown.
//
// The library supports arbitrary extents but best performance can be achieved for 
// powers of 2, followed by numbers whose prime factors are in the set {2,3,5}.
//
// Limitations:
//
//   -- You should only allocate one fft class per accelerator_view. If you need 
//      additional fft classes, create additional accelerator_view's first. This is a 
//      limitation of the Direct3D FFT API's.
//
//   -- Class fft is not thread safe. Or more accurately, the FFT API is not thread 
//      safe. So again, create additional fft objects such that each thread has its own.
//
//--------------------------------------------------------------------------------------
template <typename _Element_type, int _Dim>
class fft : public _details::fft_base
{
private:
    static_assert(_Dim>=1 && _Dim<=3, "class fft is only available for one, two or three dimensions");
    static_assert(_details::dx_fft_type_helper<_Element_type>::is_type_supported, "class fft only supports element types float and std::complex<float>");

public:
    //--------------------------------------------------------------------------------------
    // Constructor. Throws fft_exception on failure.
    //--------------------------------------------------------------------------------------
    fft(
        concurrency::extent<_Dim> _Transform_extent, 
        const concurrency::accelerator_view& _Av = concurrency::accelerator().default_view, 
        float _Forward_scale = 0.0f, 
        float _Inverse_scale = 0.0f)
        :extent(_Transform_extent), 
        fft_base(
            _details::dx_fft_type_helper<_Element_type>::dx_type, 
            _Dim, 
            &_Transform_extent[0], 
            _Av, 
            _Forward_scale, 
            _Inverse_scale)
    {
    }

    //--------------------------------------------------------------------------------------
    // Forward transform. 
    //  -- Throws fft_exception on failure. 
    //  -- Arrays extents must be identical to those of the fft object.
    //  -- It is permissible for the input and output arrays to be references to the same 
    //     array.
    //--------------------------------------------------------------------------------------
    void forward_transform(const concurrency::array<_Element_type, _Dim>& input, concurrency::array<std::complex<typename _details::dx_fft_type_helper<_Element_type>::precision_type>, _Dim>& output) const
    {
        transform(true, input, output);
    }

    //--------------------------------------------------------------------------------------
    // Inverse transform. 
    //  -- Throws fft_exception on failure. 
    //  -- Arrays extents must be identical to those of the fft object.
    //  -- It is permissible for the input and output arrays to be references to the same 
    //     array.
    //--------------------------------------------------------------------------------------
    void inverse_transform(const concurrency::array<std::complex<typename _details::dx_fft_type_helper<_Element_type>::precision_type>, _Dim>& input, concurrency::array<_Element_type, _Dim>& output) const
    {
        transform(false, input, output);
    }

    //--------------------------------------------------------------------------------------
    // The extent of the fft transform.
    //--------------------------------------------------------------------------------------
    const concurrency::extent<_Dim> extent;

private:

    template <typename _Input_element_type, typename _Output_element_type>
    void transform(bool _Forward, const concurrency::array<_Input_element_type, _Dim>& _Input, concurrency::array<_Output_element_type, _Dim>& _Output) const
    {
        if (_Input.extent != extent)
            throw fft_exception("The input extent in transform is invalid", E_INVALIDARG);

        if (_Output.extent != extent)
            throw fft_exception("The output extent in transform is invalid", E_INVALIDARG);

        HRESULT hr = S_OK;

        Microsoft::WRL::ComPtr<ID3D11Buffer> pBufferIn;
        direct3d::get_buffer(_Input)->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(pBufferIn.GetAddressOf()));
        
        Microsoft::WRL::ComPtr<ID3D11Buffer> pBufferOut;
        direct3d::get_buffer(_Output)->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(pBufferOut.GetAddressOf()));
        
        hr = base_transform(_Forward, pBufferIn.Get(), pBufferOut.Get());

        if (FAILED(hr)) throw fft_exception("transform failed", hr);
    }
};
