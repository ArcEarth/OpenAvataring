//--------------------------------------------------------------------------------------
// File: amp_fft.cpp
//
// Implementation of C++ AMP wrapper over the Direct3D FFT API's.
//
// A number of important notes regarding the implementation:
//
// The below code does allow for multi-threaded usage of the fft_base object, however
// it appears the the Direct3D API's themselves are not thread-safe. So the 
// guidelines are to use one fft object per thread, per accelerator_view.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "amp_fft.h"

namespace _details
{

//--------------------------------------------------------------------------------------
// Initialize the fft_base object.
//--------------------------------------------------------------------------------------
fft_base::fft_base(
    D3DX11_FFT_DATA_TYPE _Dx_type, 
    int _Dim, 
    const int* _Transform_extent, 
    const concurrency::accelerator_view& _Av, 
    float _Forward_scale, 
    float _Inverse_scale)
{
    HRESULT hr = S_OK;

    // Get a device context
    IUnknown *u = concurrency::direct3d::get_device(_Av);
    _ASSERTE(_M_pDevice.Get() == nullptr);
    u->QueryInterface(__uuidof(ID3D11Device), reinterpret_cast<void**>(_M_pDevice.GetAddressOf()));
    _ASSERTE(_M_pDeviceContext.Get() == nullptr);
    _M_pDevice->GetImmediateContext(_M_pDeviceContext.GetAddressOf());
    u->Release();

    _M_dx_type = _Dx_type;

    // Create an FFT interface and get buffer into
    UINT totalElementSize = 1;
    D3DX11_FFT_BUFFER_INFO fft_buffer_info;
    {
        D3DX11_FFT_DESC fft_desc;
        ZeroMemory(&fft_desc, sizeof(fft_desc));
        fft_desc.NumDimensions = _Dim;
        for (int i=0; i<_Dim; i++)
        {
            totalElementSize *= _Transform_extent[i];
            fft_desc.ElementLengths[_Dim-1-i] = _Transform_extent[i];
        }
        fft_desc.DimensionMask = (1 << _Dim) - 1;
        fft_desc.Type = _Dx_type;
        ZeroMemory(&fft_buffer_info, sizeof(fft_buffer_info));
        HRESULT hr = D3DX11CreateFFT( _M_pDeviceContext.Get(), &fft_desc, 0, &fft_buffer_info, _M_pFFT.GetAddressOf());
        if (FAILED(hr)) throw fft_exception("Failed in fft constructor", hr);

        if (_Forward_scale != 0.0f)
            set_forward_scale(_Forward_scale);

        if (_Inverse_scale != 0.0f)
            set_inverse_scale(_Inverse_scale);
    }

    // Make sure we have at least two buffers that are big enough for input/output
    if (fft_buffer_info.NumTempBufferSizes < 2)
    {
        for (UINT i=fft_buffer_info.NumTempBufferSizes; i<2; i++)
        {
            fft_buffer_info.NumTempBufferSizes = 0;
        }
        fft_buffer_info.NumTempBufferSizes = 2;
    }
    UINT elementFloatSize = (_Dx_type == D3DX11_FFT_DATA_TYPE_COMPLEX) ? 2 : 1;
    _M_Total_float_size = totalElementSize * elementFloatSize;
    for (UINT i=0; i<2; i++)
    {
        if (fft_buffer_info.TempBufferFloatSizes[i] < _M_Total_float_size)
            fft_buffer_info.TempBufferFloatSizes[i] = _M_Total_float_size;
    }

    ID3D11UnorderedAccessView *tempUAVs[D3DX11_FFT_MAX_TEMP_BUFFERS] = {0};
    ID3D11UnorderedAccessView *precomputeUAVs[D3DX11_FFT_MAX_PRECOMPUTE_BUFFERS] = {0};

    // Allocate temp and pre-computed buffers
    for (UINT i=0; i<fft_buffer_info.NumTempBufferSizes; i++)
    {
        Microsoft::WRL::ComPtr<ID3D11Buffer> pBuffer;
        _ASSERTE(_M_pTempUAVs[i].Get() == nullptr);
        hr = create_raw_buffer_and_uav(fft_buffer_info.TempBufferFloatSizes[i], pBuffer.GetAddressOf(), _M_pTempUAVs[i].GetAddressOf());
        tempUAVs[i] = _M_pTempUAVs[i].Get();

        if (FAILED(hr)) goto cleanup;
    }

    for (UINT i=0; i<fft_buffer_info.NumPrecomputeBufferSizes; i++)
    {
        Microsoft::WRL::ComPtr<ID3D11Buffer> pBuffer;
        _ASSERTE(_M_pPrecomputeUAVs[i].Get() == nullptr);
        hr = create_raw_buffer_and_uav(fft_buffer_info.PrecomputeBufferFloatSizes[i], pBuffer.GetAddressOf(), _M_pPrecomputeUAVs[i].GetAddressOf());
        precomputeUAVs[i] = _M_pPrecomputeUAVs[i].Get();

        if (FAILED(hr)) goto cleanup;
    }

    // Attach buffers and precompute

    hr = _M_pFFT->AttachBuffersAndPrecompute(
        fft_buffer_info.NumTempBufferSizes, 
        &tempUAVs[0], 
        fft_buffer_info.NumPrecomputeBufferSizes, 
        &precomputeUAVs[0]);
    if (FAILED(hr)) goto cleanup;

cleanup:
    if (FAILED(hr)) throw fft_exception("Failed in fft constructor", hr);
}

//--------------------------------------------------------------------------------------
// Sets the scale on the forward transform. A value of 0.0f may NOT be used and results
// in an fft_exception.
//--------------------------------------------------------------------------------------
void fft_base::set_forward_scale(float scale)
{
    if (scale == 0.0f)
        throw fft_exception("Invalid scale value in set_forward_scale", E_INVALIDARG);

    HRESULT hr = _M_pFFT->SetForwardScale(scale);
    if (FAILED(hr)) throw fft_exception("set_forward_scale failed", hr);
}

//--------------------------------------------------------------------------------------
// Get the scale on the forward transform.
//--------------------------------------------------------------------------------------
float fft_base::get_forward_scale() const
{
    return _M_pFFT->GetForwardScale();
}

//--------------------------------------------------------------------------------------
// Sets the scale on the inverse transform. A value of 0.0f may NOT be used and results
// in an fft_exception.
//--------------------------------------------------------------------------------------
void fft_base::set_inverse_scale(float scale)
{
    if (scale == 0.0f)
        throw fft_exception("Invalid scale value in set_inverse_scale", E_INVALIDARG);

    HRESULT hr = _M_pFFT->SetInverseScale(scale);
    if (FAILED(hr)) throw fft_exception("set_inverse_scale failed", hr);
}

//--------------------------------------------------------------------------------------
// Get the scale on the inverse transform.
//--------------------------------------------------------------------------------------
float fft_base::get_inverse_scale() const
{
    return _M_pFFT->GetInverseScale();
}
        
//--------------------------------------------------------------------------------------
// Generic transform handler.
//
// The bulk of the logic here is around buffer usage and moving data around before and
// after the call to the Direct3D API.
//--------------------------------------------------------------------------------------
HRESULT fft_base::base_transform(bool _Forward, ID3D11Buffer *pBufferIn, ID3D11Buffer *pBufferOut) const
{
    HRESULT hr = S_OK;

    UINT inputFloatSize = _M_Total_float_size;
    UINT outputFloatSize = _M_Total_float_size;

    // When processing real numbers, if it is forward transform
    // then the output is twice the size of input as it produces
    // complex data and for inverse transform, the input is complex
    // data and hence twice the size of the output which are real numbers
    if (_M_dx_type != D3DX11_FFT_DATA_TYPE_COMPLEX) 
    {
        if (_Forward) {
            outputFloatSize *= 2;
        }
        else {
            inputFloatSize *= 2;
        }
    }

    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> pUAVIn, pUAVOut;

    hr = create_uav(inputFloatSize, pBufferIn, pUAVIn.GetAddressOf());
    if (FAILED(hr)) goto cleanup;

    hr = create_uav(outputFloatSize, pBufferOut, pUAVOut.GetAddressOf());
    if (FAILED(hr)) goto cleanup;
        
    // Do the transform
    if (_Forward)
    {
        hr = _M_pFFT->ForwardTransform(pUAVIn.Get(), pUAVOut.GetAddressOf());
    }
    else
    {
        hr = _M_pFFT->InverseTransform(pUAVIn.Get(), pUAVOut.GetAddressOf());
    }
    if (FAILED(hr)) goto cleanup;

cleanup:
    return hr;
}

//--------------------------------------------------------------------------------------
// Create an unstructured raw buffer and a UAV on top of it.
//--------------------------------------------------------------------------------------
HRESULT fft_base::create_raw_buffer_and_uav(UINT floatSize, ID3D11Buffer **ppBuffer, ID3D11UnorderedAccessView **ppUAV) const
{
    if (!ppBuffer || !ppUAV) return E_POINTER;
    if (*ppBuffer || *ppUAV) return E_INVALIDARG;

    HRESULT hr = S_OK;

    hr = create_raw_buffer(floatSize, ppBuffer);
    if (FAILED(hr)) goto cleanup;

    hr = create_uav(floatSize, *ppBuffer, ppUAV);
    if (FAILED(hr)) goto cleanup;

cleanup:
    if (FAILED(hr))
    {
        if (*ppBuffer) 
        {
            (*ppBuffer)->Release();
            *ppBuffer = NULL;
        }
        if (*ppUAV)
        {
            (*ppUAV)->Release();
            *ppUAV = NULL;
        }
    }
    return hr;
}

//--------------------------------------------------------------------------------------
// Create an unstructured raw buffer.
//--------------------------------------------------------------------------------------
HRESULT fft_base::create_raw_buffer(UINT floatSize, ID3D11Buffer **ppBuffer) const
{
    D3D11_BUFFER_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.ByteWidth = floatSize * sizeof(float);
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    desc.Usage = D3D11_USAGE_DEFAULT;

    return _M_pDevice->CreateBuffer(&desc, NULL, ppBuffer);
}

//--------------------------------------------------------------------------------------
// Create a UAV on top of an unstructured buffer.
//--------------------------------------------------------------------------------------
HRESULT fft_base::create_uav(UINT floatSize, ID3D11Buffer *pBuffer, ID3D11UnorderedAccessView **ppUAV) const
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
    ZeroMemory(&UAVDesc, sizeof(UAVDesc));
    UAVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    UAVDesc.Buffer.FirstElement = 0;
    UAVDesc.Buffer.NumElements = floatSize;
    UAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

    return _M_pDevice->CreateUnorderedAccessView( pBuffer, &UAVDesc, ppUAV);
}

} // namespace _details
