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
* This file contains the helpers templates on which
* the amp_algorithms_direct3d.h header depends.
*---------------------------------------------------------------------------*/

#pragma once
#include <d3d11.h>
#include <d3dcsx.h>
#include <wrl\client.h>
#pragma comment(lib, "d3dcsx")

#include <amp_algorithms.h>

namespace amp_algorithms
{
    namespace direct3d
    {
        namespace _details
        {
            inline void _check_hresult(HRESULT _hr, std::string _exception_msg = "")
            {
                if (FAILED(_hr))
                {
                    std::stringstream _out;
                    _out << _exception_msg << " 0x" << std::hex << _hr << ".";
                    throw runtime_exception(_out.str().c_str(), _hr);
                }
            }

            inline Microsoft::WRL::ComPtr<ID3D11Device> _get_d3d11_device_ptr(const concurrency::accelerator_view &av)
            {
                IUnknown *u = concurrency::direct3d::get_device(av);
                Microsoft::WRL::ComPtr<ID3D11Device> dev_ptr;
                auto hr = u->QueryInterface(__uuidof(ID3D11Device), reinterpret_cast<void**>(dev_ptr.GetAddressOf()));
                u->Release();
                _check_hresult(hr);
                return dev_ptr;
            }

            template<typename T, unsigned int Rank>
            inline Microsoft::WRL::ComPtr<ID3D11Buffer> _get_d3d11_buffer_ptr(const concurrency::array<T, Rank> &a)
            {
                IUnknown *u = concurrency::direct3d::get_buffer(a);
                Microsoft::WRL::ComPtr<ID3D11Buffer> buf_ptr;
                auto hr = u->QueryInterface(__uuidof(ID3D11Buffer), reinterpret_cast<void**>(buf_ptr.GetAddressOf()));
                u->Release();
                _check_hresult(hr);
                return buf_ptr;
            }

            inline Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> _create_d3d11_uav(Microsoft::WRL::ComPtr<ID3D11Device> &device, Microsoft::WRL::ComPtr<ID3D11Buffer> &pSrcBuff, DXGI_FORMAT view_format)
            {
                D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
                ZeroMemory(&desc, sizeof(desc));
                desc.Format = view_format;
                desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;

                D3D11_BUFFER_DESC descBuff;
                pSrcBuff->GetDesc(&descBuff);
                desc.Buffer.FirstElement = 0;
                desc.Buffer.NumElements = descBuff.ByteWidth / sizeof(int);

                Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> pUAView;
                _check_hresult(device->CreateUnorderedAccessView(reinterpret_cast<ID3D11Resource*>(pSrcBuff.Get()), &desc, pUAView.GetAddressOf()), "Failed to create view");

                return pUAView;
            }

            template <typename _Type>
            struct _dx_scan_type_helper
            {
                static const bool is_type_supported = false;
            };

            template <>
            struct _dx_scan_type_helper<int>
            {
                static const bool is_type_supported = true;
                static const D3DX11_SCAN_DATA_TYPE dx_scan_type = D3DX11_SCAN_DATA_TYPE_INT;
                static const DXGI_FORMAT dx_view_type = DXGI_FORMAT_R32_SINT;
            };

            template <>
            struct _dx_scan_type_helper<unsigned int>
            {
                // Note: Despite what the MSDN says D3DCSX does not support uint, 
                // we can partially support it by treating it as int.
                static const bool is_type_supported = true;
                static const D3DX11_SCAN_DATA_TYPE dx_scan_type = D3DX11_SCAN_DATA_TYPE_INT;
                static const DXGI_FORMAT dx_view_type = DXGI_FORMAT_R32_SINT;
            };

            template <>
            struct _dx_scan_type_helper<float>
            {
                static const bool is_type_supported = true;
                static const D3DX11_SCAN_DATA_TYPE dx_scan_type = D3DX11_SCAN_DATA_TYPE_FLOAT;
                static const DXGI_FORMAT dx_view_type = DXGI_FORMAT_R32_FLOAT;
            };

            struct _dx_state_cleaner
            {
                _dx_state_cleaner(Microsoft::WRL::ComPtr<ID3D11DeviceContext> &context) : m_immediate_context(context)
                {
                    memset(m_uavs, 0, D3D11_PS_CS_UAV_REGISTER_COUNT * sizeof(ID3D11UnorderedAccessView*));
                    m_immediate_context->CSGetUnorderedAccessViews(0, D3D11_PS_CS_UAV_REGISTER_COUNT, m_uavs);
                }

                ~_dx_state_cleaner()
                {
                    m_immediate_context->CSSetUnorderedAccessViews(0, D3D11_PS_CS_UAV_REGISTER_COUNT, m_uavs, nullptr);
                    for (unsigned int i = 0; i < D3D11_PS_CS_UAV_REGISTER_COUNT; ++i)
                    {
                        if (m_uavs[i] != nullptr)
                        {
                            m_uavs[i]->Release();
                        }
                    }
                }

            private:
                ID3D11UnorderedAccessView *m_uavs[D3D11_PS_CS_UAV_REGISTER_COUNT];
                Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_immediate_context;
            };

            // Scan helper that converts binary functions from C++ AMP library to DirectX scan operation codes
            template <typename BinaryFunction>
            struct _dx_scan_op_helper
            {
                static const bool is_op_supported = false;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::plus<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_ADD;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::max<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_MAX;
            };

            // Further specialize amp_algorithms::max for uint and mark as not supported.
            template <>
            struct _dx_scan_op_helper<amp_algorithms::max<unsigned int>>
            {
                // max is not supported for uint, as our implementation is based on int,
                // this will return incorrect results for values that are greater than numeric_limits<int>::max().
                static const bool is_op_supported = false;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::min<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_MIN;
            };

            // Further specialize amp_algorithms::min for uint and mark as not supported.
            template <>
            struct _dx_scan_op_helper<amp_algorithms::min<unsigned int>>
            {
                // min is not supported for uint, as our implementation is based on int,
                // this will return incorrect results for values that are greater than numeric_limits<int>::max().
                static const bool is_op_supported = false;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::multiplies<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_MUL;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::bit_and<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_AND;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::bit_or<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_OR;
            };

            template <typename T>
            struct _dx_scan_op_helper<amp_algorithms::bit_xor<T>>
            {
                static const bool is_op_supported = true;
                static const D3DX11_SCAN_OPCODE dx_op_type = D3DX11_SCAN_OPCODE_XOR;
            };
        } // namespace amp_algorithms::direct3d::_details
    } // namespace amp_algorithms::direct3d
} // namespace amp_algorithms
