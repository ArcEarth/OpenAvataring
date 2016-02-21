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
 * amax_test_bench.h
 *
 * Provides an interface to implement a common testing framework for multiple
 * BLAS routines of various data types.
 *
 *---------------------------------------------------------------------------*/

#pragma once

#include <vector>
#include <string>
#include <functional>

#include "ampxblas.h"
#include "cblas_wrapper.h"

#include "ampblas_complex.h"

#include "ampblas_test_list.h"
#include "ampblas_test_util.h"
#include "ampblas_test_timer.h"

#include "ampcblas_runtime.h"

#ifdef max
#undef max
#endif

template <typename value_type, template <typename> class parameters>
class test_case : public test_list_item
{
public:

    // common typedefs
    typedef typename test_case<value_type,parameters> test_type;
	typedef typename parameters<value_type> typed_parameters;
    typedef typename std::vector<typed_parameters> typed_parameter_container;

	// related types
	typedef typename get_promoted_type<value_type>::value promoted_type;
	typedef typename get_real_type<value_type>::value real_type;
	typedef typename get_complex_type<value_type>::value complex_type;

    // exception types
    class ampblas_test_runtime_exception
    {
    public:
        ampblas_test_runtime_exception(const ampblas_result& err) 
            : err_(err) {}
        ampblas_result get() const { return err_; } 
    private:
        ampblas_result err_;
    };

    void ampblas_test_runtime_error(const ampblas_result& err)
    {
        throw ampblas_test_runtime_exception(err);
    }

    class ampblas_test_accuracy_exception
    {
    public:
        ampblas_test_accuracy_exception( double residual, double threshold ) 
			: residual_(residual), threshold_(threshold) {}
        
        double residual() const 
        { 
            return residual_; 
        }

        double threshold() const 
        { 
            return threshold_; 
        }

    private:
        double residual_;
		double threshold_;
    };

    void ampblas_test_accuracy_error(double residual, double threshold)
    {
        throw ampblas_test_accuracy_exception(residual,threshold);
    }

    // main execution loop
    void run_all_tests() 
    { 
        std::cout << "Running " << parameter_list.size() << " " << blas_prefix<value_type>() << name() << " tests... ";

        // see if test is runable on this accelerator
		if (is_double())
        {
            // see if we at least have limited doubles
            if(!ampcblas::get_current_accelerator_view().accelerator.supports_limited_double_precision)
		    {
			    std::cout << "SKIPPED (no double support)" << std::endl;
			    return;
		    }

            // some routines require full double support
            if (requires_full_double() && !ampcblas::get_current_accelerator_view().accelerator.supports_double_precision)
		    {
			    std::cout << "SKIPPED (needs full double support)" << std::endl;
			    return;
		    }
        }

        bool failed = false;
        for (const typed_parameters& p : parameter_list)
        {
            try
            {
                run_cblas_test(p);
            }
            catch (const ampblas_test_runtime_exception& e)
            {
                failed = true;
				std::cout << "\n  Runtime error with " << p.name() << "where " << e.get();
            }
            catch (const ampblas_test_accuracy_exception& e)
            {
                failed = true;
				std::cout << "\n  Accuracy error with " << p.name() << "where " << e.residual() << " > " << e.threshold();
            }
			catch (...)
			{
				failed = true;
				std::cout << "\n  Unknown error with " << p.name();
			}
        }

		if (!failed)
			std::cout << "PASSED";
		
		std::cout << std::endl;
    }

private:

	high_resolution_timer timer;
    typed_parameter_container parameter_list;
    double ref_time;
    double amp_time;

    virtual std::string name() const = 0;
    virtual void run_cblas_test( const typed_parameters& p ) = 0;

    // level of accuracy; override for routine if required
    virtual real_type fudge_factor() const { return real_type(64); }

	virtual bool is_double() const
	{
		// by default, check the real type; overload for odd cases
		return ( typeid(real_type) == typeid(double) );
	}

    virtual bool requires_full_double() const
    {
        // by default, limited doubles are OK
        // except for routines that use: 
        //  a) int -> double cases
        //  b) division
        //  c) recip
        return false;
    }

protected:

    void add_test(const typed_parameters& p) 
    { 
        parameter_list.push_back(p); 
    }

    template <typename P1, typename P2>
    void paramter_exploder(const P1& p1, const P2& p2)
    {
        container_exploder(parameter_list, p1, p2);
    }

    template <typename P1, typename P2, typename P3>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3)
    {
        container_exploder(parameter_list, p1, p2, p3);
    }

    template <typename P1, typename P2, typename P3, typename P4>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4)
    {
        container_exploder(parameter_list, p1, p2, p3, p4);
    }

    template <typename P1, typename P2, typename P3, typename P4, typename P5>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4, const P5& p5)
    {
        container_exploder(parameter_list, p1, p2, p3, p4, p5);
    }

    template <typename P1, typename P2, typename P3, typename P4, typename P5, typename P6>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4, const P5& p5, const P6& p6)
    {
        container_exploder(parameter_list, p1, p2, p3, p4, p5, p6);
    }

	template <typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4, const P5& p5, const P6& p6, const P7& p7)
    {
        container_exploder(parameter_list, p1, p2, p3, p4, p5, p6, p7);
    } 

	template <typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4, const P5& p5, const P6& p6, const P7& p7, const P8& p8)
    {
        container_exploder(parameter_list, p1, p2, p3, p4, p5, p6, p7, p8);
    } 

	template <typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4, const P5& p5, const P6& p6, const P7& p7, const P8& p8, const P9& p9)
    {
        container_exploder(parameter_list, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    } 

	template <typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10>
    void paramter_exploder(const P1& p1, const P2& p2, const P3& p3, const P4& p4, const P5& p5, const P6& p6, const P7& p7, const P8& p8, const P9& p9, const P10& p10)
    {
        container_exploder(parameter_list, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
    }

    void start_reference_test() 
    {
        timer.restart();
    }

    void stop_reference_test()
    { 
        ref_time = timer.elapsed(); 
    }

    double reference_time() const
    {
        return ref_time;
    }

    void start_ampblas_test() 
    {
        timer.restart();
    }

    void stop_ampblas_test() 
    {
        ampblas_result err = ampblas_get_last_error(); 
        if ( err )
            ampblas_test_runtime_error(err);

        // synchronize
        ampcblas::get_current_accelerator_view().wait();

        amp_time = timer.elapsed();
    }

    double ampblas_time() const
    {
        return amp_time;
    }

    // single value error check
    template <typename test_type>
    void check_error(const test_type& ref, const test_type& amp) 
    {
        real_type norm = abs(ref-amp);
		real_type threshold = fudge_factor() * std::numeric_limits<real_type>::epsilon();
		
		if (norm > threshold)
            ampblas_test_accuracy_error(norm, threshold);
    }

    // single value error check from an n-reduction
    template <typename test_type>
    void check_error(int n, const test_type& ref, const test_type& amp) 
    {
        real_type norm = abs(ref-amp);
		real_type threshold = n * fudge_factor() * std::numeric_limits<real_type>::epsilon();
		
		if (norm > threshold)
            ampblas_test_accuracy_error(norm, threshold);
    }

    // single value error check from an n-reduction
    void check_error_promoted(int n, const promoted_type& ref, const promoted_type& amp) 
    {
        promoted_type norm = abs(ref-amp);
		promoted_type threshold = n * fudge_factor() * std::numeric_limits<promoted_type>::epsilon();
		
		if (norm > threshold)
            ampblas_test_accuracy_error(norm, threshold);
    }

    // vector error check
    void check_error(const test_vector<value_type>& ref, const test_vector<value_type>& amp)
    {				
		// vector one norm
		// ||x||_1 = sum_i(|x_i|)
		
		real_type norm = 0;
		real_type threshold = fudge_factor() * std::numeric_limits<real_type>::epsilon() * ref.n();
		
		for (int i=0; i<ref.n(); i++)
			norm += abs(ref[i] - amp[i]);	
				
		if (norm > threshold)
            ampblas_test_accuracy_error(norm, threshold);
	}

    // matrix error check (inf norm)
    void check_error(const test_matrix<value_type>& ref, const test_matrix<value_type>& amp) 
    {
		// matrix one norm
		// ||A||_1 = max_j(sum_i(|a_ij|))
		
		real_type norm = 0;
		real_type threshold = fudge_factor() * std::numeric_limits<real_type>::epsilon() * ref.m();
		
		for (int j=0; j<ref.n(); j++)
		{
			real_type sum = 0;
			for (int i=0; i<ref.m(); i++)
				sum += abs(ref(i,j) - amp(i,j));
			norm = std::max(norm,sum);
		}
        		
		if (norm > threshold)
            ampblas_test_accuracy_error(norm, threshold); 
    }
};
