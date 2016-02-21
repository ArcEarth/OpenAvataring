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
 * cblas_wrapper.cpp
 *
 * Provides a more portable interface to a BLAS implementation translated by
 * F2C.  Currenlty only provides wrappers to functions implemented in 
 * ampblas.
 *
 *---------------------------------------------------------------------------*/

#include "cblas_wrapper.h"

namespace f2c_cblas {

#include "f2c.h"
#include "cblas.h"

} // namesapce f2c_cblas

namespace cblas {

	// ------------------------------------------------------------------------
	// F2C CBLAS Type Adapters
	// ------------------------------------------------------------------------

    class Integer
    {
    private:
        f2c_cblas::integer i_;
    public:
        Integer( const int& i ) : i_(i) {}
        inline operator f2c_cblas::integer*() { return &i_; }
        inline operator const f2c_cblas::integer*() const { return &i_; }
    };

    inline f2c_cblas::real* Real( const float* s ) { return const_cast<f2c_cblas::real*>(s); } 
    inline f2c_cblas::real* Real( float* s ) { return static_cast<f2c_cblas::real*>(s); }
    inline f2c_cblas::real* Real( float& s ) { return static_cast<f2c_cblas::real*>(&s); }

    inline f2c_cblas::doublereal* DoubleReal( const double* d ) { return const_cast<f2c_cblas::doublereal*>(d); } 
    inline f2c_cblas::doublereal* DoubleReal( double* d ) { return static_cast<f2c_cblas::doublereal*>(d); }
    inline f2c_cblas::doublereal* DoubleReal( double& d ) { return static_cast<f2c_cblas::doublereal*>(&d); }

    inline f2c_cblas::complex* Complex( const cblas::complex_float* c ) { return reinterpret_cast<f2c_cblas::complex*>(const_cast<cblas::complex_float*>(c)); }
    inline f2c_cblas::complex* Complex( cblas::complex_float* c ) { return reinterpret_cast<f2c_cblas::complex*>(c); }
    inline f2c_cblas::complex* Complex( cblas::complex_float& c ) { return reinterpret_cast<f2c_cblas::complex*>(&c); }

    inline f2c_cblas::doublecomplex* DoubleComplex( const cblas::complex_double* z ) { return reinterpret_cast<f2c_cblas::doublecomplex*>(const_cast<cblas::complex_double*>(z)); }
    inline f2c_cblas::doublecomplex* DoubleComplex( cblas::complex_double* z ) { return reinterpret_cast<f2c_cblas::doublecomplex*>(z); }
    inline f2c_cblas::doublecomplex* DoubleComplex( cblas::complex_double& z ) { return reinterpret_cast<f2c_cblas::doublecomplex*>(&z); }

	class Option
	{
	private:
		char option_;

	public:

		Option(const enum class side& side)
		{
			switch (side)
			{
			case side::left:
				option_ = 'L';
				break;
			case side::right:
				option_ = 'R';
				break;
			}
		}

		Option(const enum class uplo& uplo)
		{
			switch (uplo)
			{
			case uplo::lower:
				option_ = 'L';
				break;
			case uplo::upper:
				option_ = 'U';
				break;
			}
		}

		Option(const enum class transpose& trans)
		{
			switch (trans)
			{
			case transpose::no_trans:
				option_ = 'N';
				break;
			case transpose::trans:
				option_ = 'T';
				break;
			case transpose::conj_trans:
				option_ = 'C';
				break;
			}
		}

		Option(const enum class diag& diag)
		{
			switch (diag)
			{
			case diag::non_unit:
				option_ = 'N';
				break;
			case diag::unit:
				option_ = 'U';
				break;
			}
		}

		inline operator char*() 
        { 
            return &option_; 
        }
	};

    // ------------------------------------------------------------------------
    // BLAS 1 
    // ------------------------------------------------------------------------

    // ASUM
    float  SASUM( int N, const float*          X, int INCX ) { return  f2c_cblas::sasum_( Integer(N), Real(X),          Integer(INCX) ); }
    double DASUM( int N, const double*         X, int INCX ) { return  f2c_cblas::dasum_( Integer(N), DoubleReal(X),    Integer(INCX) ); }
    float  SCSUM( int N, const complex_float*  X, int INCX ) { return f2c_cblas::scasum_( Integer(N), Complex(X),       Integer(INCX) ); } 
    double DZSUM( int N, const complex_double* X, int INCX ) { return f2c_cblas::dzasum_( Integer(N), DoubleComplex(X), Integer(INCX) ); } 

    // AMAX
    int ISAMAX( int N, const float*          X, int INCX ) { return f2c_cblas::isamax_( Integer(N), Real(X),          Integer(INCX) ); }
    int IDAMAX( int N, const double*         X, int INCX ) { return f2c_cblas::idamax_( Integer(N), DoubleReal(X),    Integer(INCX) ); }
    int ICAMAX( int N, const complex_float*  X, int INCX ) { return f2c_cblas::icamax_( Integer(N), Complex(X),       Integer(INCX) ); }
    int IZAMAX( int N, const complex_double* X, int INCX ) { return f2c_cblas::izamax_( Integer(N), DoubleComplex(X), Integer(INCX) ); }
    
    // AXPY    
    void SAXPY( int N, float          ALPHA, const float*          X, int INCX, float*          Y, int INCY ) { f2c_cblas::saxpy_( Integer(N), Real(ALPHA),          Real(X),          Integer(INCX), Real(Y),       Integer(INCY) ); }
    void DAXPY( int N, double         ALPHA, const double*         X, int INCX, double*         Y, int INCY ) { f2c_cblas::daxpy_( Integer(N), DoubleReal(ALPHA),    DoubleReal(X),    Integer(INCX), DoubleReal(Y), Integer(INCY) ); }
    void CAXPY( int N, complex_float  ALPHA, const complex_float*  X, int INCX, complex_float*  Y, int INCY ) { f2c_cblas::caxpy_( Integer(N), Complex(ALPHA),       Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY) ); }
    void ZAXPY( int N, complex_double ALPHA, const complex_double* X, int INCX, complex_double* Y, int INCY ) { f2c_cblas::zaxpy_( Integer(N), DoubleComplex(ALPHA), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY) ); }

    // COPY
    void SCOPY( int N, float*          X, int INCX, float*          Y, int INCY ) { f2c_cblas::scopy_( Integer(N), Real(X),          Integer(INCX), Real(Y),          Integer(INCY) ); }
    void DCOPY( int N, double*         X, int INCX, double*         Y, int INCY ) { f2c_cblas::dcopy_( Integer(N), DoubleReal(X),    Integer(INCX), DoubleReal(Y),    Integer(INCY) ); }
    void CCOPY( int N, complex_float*  X, int INCX, complex_float*  Y, int INCY ) { f2c_cblas::ccopy_( Integer(N), Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY) ); }
    void ZCOPY( int N, complex_double* X, int INCX, complex_double* Y, int INCY ) { f2c_cblas::zcopy_( Integer(N), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY) ); }

    // DOT
    double         DDOT( int N, const double*         X, int INCX, const double*         Y, int INCY ) { return   f2c_cblas::ddot_( Integer(N), DoubleReal(X),    Integer(INCX), DoubleReal(Y),    Integer(INCY) ); }
    double        DSDOT( int N, const float*          X, int INCX, const float*          Y, int INCY ) { return  f2c_cblas::dsdot_( Integer(N), Real(X),          Integer(INCX), Real(Y),          Integer(INCY) ); }
    float          SDOT( int N, const float*          X, int INCX, const float*          Y, int INCY ) { return   f2c_cblas::sdot_( Integer(N), Real(X),          Integer(INCX), Real(Y),          Integer(INCY) ); }
    
    // Complex DOT
    complex_float  CDOTU( int N, const complex_float*  X, int INCX, const complex_float*  Y, int INCY ) { complex_float ret;  f2c_cblas::cdotu_( Complex(ret),       Integer(N), Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY) ); return ret; }
    complex_float  CDOTC( int N, const complex_float*  X, int INCX, const complex_float*  Y, int INCY ) { complex_float ret;  f2c_cblas::cdotc_( Complex(ret),       Integer(N), Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY) ); return ret; }
    complex_double ZDOTU( int N, const complex_double* X, int INCX, const complex_double* Y, int INCY ) { complex_double ret; f2c_cblas::zdotu_( DoubleComplex(ret), Integer(N), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY) ); return ret; }
    complex_double ZDOTC( int N, const complex_double* X, int INCX, const complex_double* Y, int INCY ) { complex_double ret; f2c_cblas::zdotc_( DoubleComplex(ret), Integer(N), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY) ); return ret; }

    // NRM2
    float   SNRM2( int N, const float*          X, int INCX ) { return f2c_cblas::snrm2_ ( Integer(N), Real(X),          Integer(INCX) ); }
    double  DNRM2( int N, const double*         X, int INCX ) { return f2c_cblas::dnrm2_ ( Integer(N), DoubleReal(X),    Integer(INCX) ); }
    float  SCNRM2( int N, const complex_float*  X, int INCX ) { return f2c_cblas::scnrm2_( Integer(N), Complex(X),       Integer(INCX) ); }
    double DZNRM2( int N, const complex_double* X, int INCX ) { return f2c_cblas::dznrm2_( Integer(N), DoubleComplex(X), Integer(INCX) ); }

    // ROT
    void SROT( int N, float*  X, int INCX, float*  Y, int INCY, float  C, float  S ) { f2c_cblas::srot_( Integer(N), Real(X),       Integer(INCX), Real(Y),       Integer(INCY), Real(C),       Real(S)       ); }
    void DROT( int N, double* X, int INCX, double* Y, int INCY, double C, double S ) { f2c_cblas::drot_( Integer(N), DoubleReal(X), Integer(INCX), DoubleReal(Y), Integer(INCY), DoubleReal(C), DoubleReal(S) ); }

    // SCAL
    void  SSCAL( int N, float          ALPHA, float*          X, int INCX ) { f2c_cblas::sscal_ ( Integer(N), Real(ALPHA),          Real(X),          Integer(INCX) ); }
    void  DSCAL( int N, double         ALPHA, double*         X, int INCX ) { f2c_cblas::dscal_ ( Integer(N), DoubleReal(ALPHA),    DoubleReal(X),    Integer(INCX) ); }
    void  CSCAL( int N, complex_float  ALPHA, complex_float*  X, int INCX ) { f2c_cblas::cscal_ ( Integer(N), Complex(ALPHA),       Complex(X),       Integer(INCX) ); }
    void  ZSCAL( int N, complex_double ALPHA, complex_double* X, int INCX ) { f2c_cblas::zscal_ ( Integer(N), DoubleComplex(ALPHA), DoubleComplex(X), Integer(INCX) ); }
    void CSSCAL( int N, float          ALPHA, complex_float*  X, int INCX ) { f2c_cblas::csscal_( Integer(N), Real(ALPHA),          Complex(X),       Integer(INCX) ); }
    void ZDSCAL( int N, double         ALPHA, complex_double* X, int INCX ) { f2c_cblas::zdscal_( Integer(N), DoubleReal(ALPHA),    DoubleComplex(X), Integer(INCX) ); }

    // SWAP
    void SSWAP( int N, float*          X, int INCX, float*          Y, int INCY ) { f2c_cblas::sswap_( Integer(N), Real(X),          Integer(INCX), Real(Y),       Integer(INCY) ); }
    void DSWAP( int N, double*         X, int INCX, double*         Y, int INCY ) { f2c_cblas::dswap_( Integer(N), DoubleReal(X),    Integer(INCX), DoubleReal(Y), Integer(INCY) ); }
    void CSWAP( int N, complex_float*  X, int INCX, complex_float*  Y, int INCY ) { f2c_cblas::cswap_( Integer(N), Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY) ); }
    void ZSWAP( int N, complex_double* X, int INCX, complex_double* Y, int INCY ) { f2c_cblas::zswap_( Integer(N), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY) ); }

	// ------------------------------------------------------------------------
	// BLAS 2
	// ------------------------------------------------------------------------

	// GER
    void SGER ( int M, int N, float          ALPHA, const float*          X, int INCX, const float*          Y, int INCY, float*          A, int LDA ) { f2c_cblas::sger_ ( Integer(M), Integer(N), Real(ALPHA),          Real(X),          Integer(INCX), Real(Y),          Integer(INCY), Real(A),          Integer(LDA) ); }
    void DGER ( int M, int N, double         ALPHA, const double*         X, int INCX, const double*         Y, int INCY, double*         A, int LDA ) { f2c_cblas::dger_ ( Integer(M), Integer(N), DoubleReal(ALPHA),    DoubleReal(X),    Integer(INCX), DoubleReal(Y),    Integer(INCY), DoubleReal(A),    Integer(LDA) ); }
    void CGERU( int M, int N, complex_float  ALPHA, const complex_float*  X, int INCX, const complex_float*  Y, int INCY, complex_float*  A, int LDA ) { f2c_cblas::cgeru_( Integer(M), Integer(N), Complex(ALPHA),       Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY), Complex(A),       Integer(LDA) ); }
    void CGERC( int M, int N, complex_float  ALPHA, const complex_float*  X, int INCX, const complex_float*  Y, int INCY, complex_float*  A, int LDA ) { f2c_cblas::cgerc_( Integer(M), Integer(N), Complex(ALPHA),       Complex(X),       Integer(INCX), Complex(Y),       Integer(INCY), Complex(A),       Integer(LDA) ); }
    void ZGERU( int M, int N, complex_double ALPHA, const complex_double* X, int INCX, const complex_double* Y, int INCY, complex_double* A, int LDA ) { f2c_cblas::zgeru_( Integer(M), Integer(N), DoubleComplex(ALPHA), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY), DoubleComplex(A), Integer(LDA) ); }
    void ZGERC( int M, int N, complex_double ALPHA, const complex_double* X, int INCX, const complex_double* Y, int INCY, complex_double* A, int LDA ) { f2c_cblas::zgerc_( Integer(M), Integer(N), DoubleComplex(ALPHA), DoubleComplex(X), Integer(INCX), DoubleComplex(Y), Integer(INCY), DoubleComplex(A), Integer(LDA) ); }

	// GEMV
	void SGEMV( enum class transpose TRANSA, int M, int N, float          ALPHA, const float*          A, int LDA, const float*          X, int INCX, float          BETA, float*          Y, int INCY ) { f2c_cblas::sgemv_( Option(TRANSA), Integer(M), Integer(N), Real(ALPHA),          Real(A),          Integer(LDA), Real(X),          Integer(INCX), Real(BETA),          Real(Y),          Integer(INCY) ); }
	void DGEMV( enum class transpose TRANSA, int M, int N, double         ALPHA, const double*         A, int LDA, const double*         X, int INCX, double         BETA, double*         Y, int INCY ) { f2c_cblas::dgemv_( Option(TRANSA), Integer(M), Integer(N), DoubleReal(ALPHA),    DoubleReal(A),    Integer(LDA), DoubleReal(X),    Integer(INCX), DoubleReal(BETA),    DoubleReal(Y),    Integer(INCY) ); }
    void CGEMV( enum class transpose TRANSA, int M, int N, complex_float  ALPHA, const complex_float*  A, int LDA, const complex_float*  X, int INCX, complex_float  BETA, complex_float*  Y, int INCY ) { f2c_cblas::cgemv_( Option(TRANSA), Integer(M), Integer(N), Complex(ALPHA),       Complex(A),       Integer(LDA), Complex(X),       Integer(INCX), Complex(BETA),       Complex(Y),       Integer(INCY) ); }
    void ZGEMV( enum class transpose TRANSA, int M, int N, complex_double ALPHA, const complex_double* A, int LDA, const complex_double* X, int INCX, complex_double BETA, complex_double* Y, int INCY ) { f2c_cblas::zgemv_( Option(TRANSA), Integer(M), Integer(N), DoubleComplex(ALPHA), DoubleComplex(A), Integer(LDA), DoubleComplex(X), Integer(INCX), DoubleComplex(BETA), DoubleComplex(Y), Integer(INCY) ); }

    // SYR                                                                                                   
    void SSYR( enum class uplo UPLO, int N, float  ALPHA, const float*          X, int INCX, float*          A, int LDA) { f2c_cblas::ssyr_( Option(UPLO), Integer(N), Real(ALPHA),       Real(X),          Integer(INCX), Real(A),          Integer(LDA) ); }
    void DSYR( enum class uplo UPLO, int N, double ALPHA, const double*         X, int INCX, double*         A, int LDA) { f2c_cblas::dsyr_( Option(UPLO), Integer(N), DoubleReal(ALPHA), DoubleReal(X),    Integer(INCX), DoubleReal(A),    Integer(LDA) ); }
    void CHER( enum class uplo UPLO, int N, float  ALPHA, const complex_float*  X, int INCX, complex_float*  A, int LDA) { f2c_cblas::cher_( Option(UPLO), Integer(N), Real(ALPHA),       Complex(X),       Integer(INCX), Complex(A),       Integer(LDA) ); }
    void ZHER( enum class uplo UPLO, int N, double ALPHA, const complex_double* X, int INCX, complex_double* A, int LDA) { f2c_cblas::zher_( Option(UPLO), Integer(N), DoubleReal(ALPHA), DoubleComplex(X), Integer(INCX), DoubleComplex(A), Integer(LDA) ); }

    // SYMV
    void SSYMV( enum class uplo UPLO, int N, float          ALPHA, float*          A, int LDA, float*          X, int INCX, float          BETA, float*          Y, int INCY) { f2c_cblas::ssymv_( Option(UPLO), Integer(N), Real(ALPHA),          Real(A),          Integer(LDA), Real(X),          Integer(INCX), Real(BETA),          Real(Y),          Integer(INCY)); }
    void DSYMV( enum class uplo UPLO, int N, double         ALPHA, double*         A, int LDA, double*         X, int INCX, double         BETA, double*         Y, int INCY) { f2c_cblas::dsymv_( Option(UPLO), Integer(N), DoubleReal(ALPHA),    DoubleReal(A),    Integer(LDA), DoubleReal(X),    Integer(INCX), DoubleReal(BETA),    DoubleReal(Y),    Integer(INCY)); }
    void CHEMV( enum class uplo UPLO, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  X, int INCX, complex_float  BETA, complex_float*  Y, int INCY) { f2c_cblas::chemv_( Option(UPLO), Integer(N), Complex(ALPHA),       Complex(A),       Integer(LDA), Complex(X),       Integer(INCX), Complex(BETA),       Complex(Y),       Integer(INCY)); }
    void ZHEMV( enum class uplo UPLO, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* X, int INCX, complex_double BETA, complex_double* Y, int INCY) { f2c_cblas::zhemv_( Option(UPLO), Integer(N), DoubleComplex(ALPHA), DoubleComplex(A), Integer(LDA), DoubleComplex(X), Integer(INCX), DoubleComplex(BETA), DoubleComplex(Y), Integer(INCY)); }
	
    // TRMV
    void STRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, float*           A, int LDA, float*           X, int INCX) { f2c_cblas::strmv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), Real(A),          Integer(LDA), Real(X),          Integer(INCX) ); }
    void DTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, double*          A, int LDA, double*          X, int INCX) { f2c_cblas::dtrmv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), DoubleReal(A),    Integer(LDA), DoubleReal(X),    Integer(INCX) ); }
    void CTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_float*   A, int LDA, complex_float*   X, int INCX) { f2c_cblas::ctrmv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), Complex(A),       Integer(LDA), Complex(X),       Integer(INCX) ); }
    void ZTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_double*  A, int LDA, complex_double*  X, int INCX) { f2c_cblas::ztrmv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), DoubleComplex(A), Integer(LDA), DoubleComplex(X), Integer(INCX) ); }

    // TRSV
    void STRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, float*           A, int LDA, float*           X, int INCX) { f2c_cblas::strsv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), Real(A),          Integer(LDA), Real(X),          Integer(INCX) ); }
    void DTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, double*          A, int LDA, double*          X, int INCX) { f2c_cblas::dtrsv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), DoubleReal(A),    Integer(LDA), DoubleReal(X),    Integer(INCX) ); }
    void CTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_float*   A, int LDA, complex_float*   X, int INCX) { f2c_cblas::ctrsv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), Complex(A),       Integer(LDA), Complex(X),       Integer(INCX) ); }
    void ZTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_double*  A, int LDA, complex_double*  X, int INCX) { f2c_cblas::ztrsv_( Option(UPLO), Option(TRANS), Option(DIAG), Integer(N), DoubleComplex(A), Integer(LDA), DoubleComplex(X), Integer(INCX) ); }

    // ------------------------------------------------------------------------
	// BLAS 3
	// ------------------------------------------------------------------------

	// GEMM
	void SGEMM( enum class transpose TRANSA, enum class transpose TRANBS, int M, int N, int K, float          ALPHA, float*          A, int LDA, float*          B, int LDB, float          beta, float*          C, int LDC ) { f2c_cblas::sgemm_( Option(TRANSA), Option(TRANBS), Integer(M), Integer(N), Integer(K), Real(ALPHA),          Real(A),          Integer(LDA), Real(B),          Integer(LDB), Real(beta),          Real(C),          Integer(LDC) ); }
	void DGEMM( enum class transpose TRANSA, enum class transpose TRANBS, int M, int N, int K, double         ALPHA, double*         A, int LDA, double*         B, int LDB, double         beta, double*         C, int LDC ) { f2c_cblas::dgemm_( Option(TRANSA), Option(TRANBS), Integer(M), Integer(N), Integer(K), DoubleReal(ALPHA),    DoubleReal(A),    Integer(LDA), DoubleReal(B),    Integer(LDB), DoubleReal(beta),    DoubleReal(C),    Integer(LDC) ); }
    void CGEMM( enum class transpose TRANSA, enum class transpose TRANBS, int M, int N, int K, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB, complex_float  beta, complex_float*  C, int LDC ) { f2c_cblas::cgemm_( Option(TRANSA), Option(TRANBS), Integer(M), Integer(N), Integer(K), Complex(ALPHA),       Complex(A),       Integer(LDA), Complex(B),       Integer(LDB), Complex(beta),       Complex(C),       Integer(LDC) ); }
	void ZGEMM( enum class transpose TRANSA, enum class transpose TRANBS, int M, int N, int K, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB, complex_double beta, complex_double* C, int LDC ) { f2c_cblas::zgemm_( Option(TRANSA), Option(TRANBS), Integer(M), Integer(N), Integer(K), DoubleComplex(ALPHA), DoubleComplex(A), Integer(LDA), DoubleComplex(B), Integer(LDB), DoubleComplex(beta), DoubleComplex(C), Integer(LDC) ); }

    // TRMM
	void STRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, float          ALPHA, float*          A, int LDA, float*          B, int LDB ) { f2c_cblas::strmm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), Real(ALPHA),		     Real(A),          Integer(LDA), Real(B),          Integer(LDB) ); }
	void DTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, double         ALPHA, double*         A, int LDA, double*         B, int LDB ) { f2c_cblas::dtrmm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), DoubleReal(ALPHA),    DoubleReal(A),    Integer(LDA), DoubleReal(B),    Integer(LDB) ); }
    void CTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB ) { f2c_cblas::ctrmm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), Complex(ALPHA),		 Complex(A),       Integer(LDA), Complex(B),       Integer(LDB) ); }
	void ZTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB ) { f2c_cblas::ztrmm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), DoubleComplex(ALPHA), DoubleComplex(A), Integer(LDA), DoubleComplex(B), Integer(LDB) ); }

	// TRSM
	void STRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, float          ALPHA, float*          A, int LDA, float*          B, int LDB ) { f2c_cblas::strsm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), Real(ALPHA),		     Real(A),          Integer(LDA), Real(B),          Integer(LDB) ); }
	void DTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, double         ALPHA, double*         A, int LDA, double*         B, int LDB ) { f2c_cblas::dtrsm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), DoubleReal(ALPHA),    DoubleReal(A),    Integer(LDA), DoubleReal(B),    Integer(LDB) ); }
    void CTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB ) { f2c_cblas::ctrsm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), Complex(ALPHA),		 Complex(A),       Integer(LDA), Complex(B),       Integer(LDB) ); }
	void ZTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB ) { f2c_cblas::ztrsm_( Option(SIDE), Option(UPLO), Option(TRANSA), Option(DIAG), Integer(M), Integer(N), DoubleComplex(ALPHA), DoubleComplex(A), Integer(LDA), DoubleComplex(B), Integer(LDB) ); }
	
    // SYMM
	void SSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, float  ALPHA, float*  A, int LDA, float*  B, int LDB, float  BETA, float*  C, int LDC) { f2c_cblas::ssymm_( Option(SIDE), Option(UPLO), Integer(M), Integer(N), Real(ALPHA),       Real(A),       Integer(LDA), Real(B),       Integer(LDB), Real(BETA),       Real(C),       Integer(LDC)); }
	void DSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, double ALPHA, double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC) { f2c_cblas::dsymm_( Option(SIDE), Option(UPLO), Integer(M), Integer(N), DoubleReal(ALPHA), DoubleReal(A), Integer(LDA), DoubleReal(B), Integer(LDB), DoubleReal(BETA), DoubleReal(C), Integer(LDC)); }

	// SYRK
	void SSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, float*          A, int LDA, float  BETA, float*          C, int LDC) { f2c_cblas::ssyrk_( Option(UPLO), Option(TRANS), Integer(N), Integer(K), Real(ALPHA),       Real(A),          Integer(LDA), Real(BETA),       Real(C),          Integer(LDC)); }
	void DSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, double*         A, int LDA, double BETA, double*         C, int LDC) { f2c_cblas::dsyrk_( Option(UPLO), Option(TRANS), Integer(N), Integer(K), DoubleReal(ALPHA), DoubleReal(A),    Integer(LDA), DoubleReal(BETA), DoubleReal(C),    Integer(LDC)); }
    void CHERK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, complex_float*  A, int LDA, float  BETA, complex_float*  C, int LDC) { f2c_cblas::cherk_( Option(UPLO), Option(TRANS), Integer(N), Integer(K), Real(ALPHA),       Complex(A),       Integer(LDA), Real(BETA),       Complex(C),       Integer(LDC)); }
    void ZHERK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, complex_double* A, int LDA, double BETA, complex_double* C, int LDC) { f2c_cblas::zherk_( Option(UPLO), Option(TRANS), Integer(N), Integer(K), DoubleReal(ALPHA), DoubleComplex(A), Integer(LDA), DoubleReal(BETA), DoubleComplex(C), Integer(LDC)); }

	// SYR2K
	void SSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, float*  A, int LDA, float*  B, int LDB, float  BETA, float*  C, int LDC) { f2c_cblas::ssyr2k_( Option(UPLO), Option(TRANS), Integer(N), Integer(K), Real(ALPHA),       Real(A),       Integer(LDA), Real(B),       Integer(LDB), Real(BETA),       Real(C),       Integer(LDC)); }
	void DSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC) { f2c_cblas::dsyr2k_( Option(UPLO), Option(TRANS), Integer(N), Integer(K), DoubleReal(ALPHA), DoubleReal(A), Integer(LDA), DoubleReal(B), Integer(LDB), DoubleReal(BETA), DoubleReal(C), Integer(LDC)); }

} // namespace cblas
