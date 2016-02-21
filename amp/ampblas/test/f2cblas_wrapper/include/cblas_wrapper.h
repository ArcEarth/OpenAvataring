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
 * cblas_wrapper.h
 *
 * Provides a more portable interface to a BLAS implementation as well as C++
 * overloaded functions. Currenlty only provides wrappers to functions
 * implemented in ampblas.
 *
 *---------------------------------------------------------------------------*/

#pragma once
#ifndef CBLAS_WRAPPER_H
#define CBLAS_WRAPPER_H

namespace cblas {

    // ------------------------------------------------------------------------
    // Complex Types
    // ------------------------------------------------------------------------

    template <typename T>
    struct complex
    {
        T real;
        T imag;
    };

    typedef complex<float> complex_float;
    typedef complex<double> complex_double;

    template <typename T>
    struct real_type { typedef typename T type; };

    template <typename T>
    struct real_type<complex<T>> { typedef typename T type; };

	// ------------------------------------------------------------------------
	// Options
	// ------------------------------------------------------------------------

	enum class order {row_major, col_major};
	enum class transpose {no_trans, trans, conj_trans};
	enum class uplo {upper, lower};
	enum class diag {non_unit, unit};
	enum class side {left, right};

    // ------------------------------------------------------------------------
    // BLAS 1
    // ------------------------------------------------------------------------

    // ASUM
    float  SASUM( int N, const float*          X, int INCX );
    double DASUM( int N, const double*         X, int INCX );
    float  SCSUM( int N, const complex_float*  X, int INCX );
    double DZSUM( int N, const complex_double* X, int INCX );

    // AMAX
    int ISAMAX( int N, const float*          X, int INCX );
    int IDAMAX( int N, const double*         X, int INCX );
    int ICAMAX( int N, const complex_float*  X, int INCX );
    int IZAMAX( int N, const complex_double* X, int INCX );

    // AXPY
    void SAXPY( int N, float          ALPHA, const float*          X, int INCX, float*          Y, int INCY );
    void DAXPY( int N, double         ALPHA, const double*         X, int INCX, double*         Y, int INCY );
    void CAXPY( int N, complex_float  ALPHA, const complex_float*  X, int INCX, complex_float*  Y, int INCY );
    void ZAXPY( int N, complex_double ALPHA, const complex_double* X, int INCX, complex_double* Y, int INCY );

    // COPY
    void SCOPY( int N, float*          X, int INCX, float*          Y, int INCY );
    void DCOPY( int N, double*         X, int INCX, double*         Y, int INCY );
    void CCOPY( int N, complex_float*  X, int INCX, complex_float*  Y, int INCY );
    void ZCOPY( int N, complex_double* X, int INCX, complex_double* Y, int INCY );

    // DOT
    float           SDOT( int N, const float*          X, int INCX, const float*          Y, int INCY );
    double         DSDOT( int N, const float*          X, int INCX, const float*          Y, int INCY );
    double          DDOT( int N, const double*         X, int INCX, const double*         Y, int INCY );
    complex_float  CDOTU( int N, const complex_float*  X, int INCX, const complex_float*  Y, int INCY );
    complex_float  CDOTC( int N, const complex_float*  X, int INCX, const complex_float*  Y, int INCY );
    complex_double ZDOTU( int N, const complex_double* X, int INCX, const complex_double* Y, int INCY );
    complex_double ZDOTC( int N, const complex_double* X, int INCX, const complex_double* Y, int INCY );

    // NRM2
    float   SNRM2( int N, const float*          X, int INCX );
    double  DNRM2( int N, const double*         X, int INCX );
    float  SCNRM2( int N, const complex_float*  X, int INCX );
    double DZNRM2( int N, const complex_double* X, int INCX );

    // ROT
    void SROT( int N, float*  X, int INCX, float*  Y, int INCY, float  C, float  S );
    void DROT( int N, double* X, int INCX, double* Y, int INCY, double C, double S );

    // SCAL
    void SSCAL ( int N, float          A, float*          X, int INCX );
    void DSCAL ( int N, double         A, double*         X, int INCX );
    void CSCAL ( int N, complex_float  A, complex_float*  X, int INCX );
    void ZSCAL ( int N, complex_double A, complex_double* X, int INCX );
    void CSSCAL( int N, float          A, complex_float*  X, int INCX );
    void ZDSCAL( int N, double         A, complex_double* X, int INCX );

    // SWAP
    void SSWAP( int N, float*          X, int INCX, float*          Y, int INCY );
    void DSWAP( int N, double*         X, int INCX, double*         Y, int INCY );
    void CSWAP( int N, complex_float*  X, int INCX, complex_float*  Y, int INCY );
    void ZSWAP( int N, complex_double* X, int INCX, complex_double* Y, int INCY );

	// ------------------------------------------------------------------------
	// BLAS 2
	// ------------------------------------------------------------------------

	// GER
    void SGER( int M, int N, float          ALPHA, const float*          X, int INCX, const float*          Y, int INCY, float*          A, int LDA );
    void DGER( int M, int N, double         ALPHA, const double*         X, int INCX, const double*         Y, int INCY, double*         A, int LDA );
    void CGERU( int M, int N, complex_float  ALPHA, const complex_float*  X, int INCX, const complex_float*  Y, int INCY, complex_float*  A, int LDA );
    void CGERC( int M, int N, complex_float  ALPHA, const complex_float*  X, int INCX, const complex_float*  Y, int INCY, complex_float*  A, int LDA );
    void ZGERU( int M, int N, complex_double ALPHA, const complex_double* X, int INCX, const complex_double* Y, int INCY, complex_double* A, int LDA );
    void ZGERC( int M, int N, complex_double ALPHA, const complex_double* X, int INCX, const complex_double* Y, int INCY, complex_double* A, int LDA );
	
	// GEMV
	void SGEMV( enum class transpose TRANSA, int M, int N, float          ALPHA, const float*          A, int LDA, const float*          X, int INCX, float          BETA, float*          Y, int INCY );
    void DGEMV( enum class transpose TRANSA, int M, int N, double         ALPHA, const double*         A, int LDA, const double*         X, int INCX, double         BETA, double*         Y, int INCY );
    void CGEMV( enum class transpose TRANSA, int M, int N, complex_float  ALPHA, const complex_float*  A, int LDA, const complex_float*  X, int INCX, complex_float  BETA, complex_float*  Y, int INCY );
	void ZGEMV( enum class transpose TRANSA, int M, int N, complex_double ALPHA, const complex_double* A, int LDA, const complex_double* X, int INCX, complex_double BETA, complex_double* Y, int INCY );
	
    // SYR
    void SSYR( enum class uplo UPLO, int N, float  ALPHA, const float*          X, int INCX, float*          A, int LDA );
    void DSYR( enum class uplo UPLO, int N, double ALPHA, const double*         X, int INCX, double*         A, int LDA );
    void CHER( enum class uplo UPLO, int N, float  ALPHA, const complex_float*  X, int INCX, complex_float*  A, int LDA );
    void ZHER( enum class uplo UPLO, int N, double ALPHA, const complex_double* X, int INCX, complex_double* A, int LDA );

    // SYMV
    void SSYMV( enum class uplo UPLO, int N, float          ALPHA, float*          A, int LDA, float*          X, int INCX, float          BETA, float*          Y, int INCY);
    void DSYMV( enum class uplo UPLO, int N, double         ALPHA, double*         A, int LDA, double*         X, int INCX, double         BETA, double*         Y, int INCY);
    void CHEMV( enum class uplo UPLO, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  X, int INCX, complex_float  BETA, complex_float*  Y, int INCY);
    void ZHEMV( enum class uplo UPLO, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* X, int INCX, complex_double BETA, complex_double* Y, int INCY);
	
    // TRMV
    void STRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, float*          A, int LDA, float*          X, int INCX);
    void DTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, double*         A, int LDA, double*         X, int INCX);
    void CTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_float*  A, int LDA, complex_float*  X, int INCX);
    void ZTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_double* A, int LDA, complex_double* X, int INCX);

    // TRSV
    void STRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, float*          A, int LDA, float*          X, int INCX);
    void DTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, double*         A, int LDA, double*         X, int INCX);
    void CTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_float*  A, int LDA, complex_float*  X, int INCX);
    void ZTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_double* A, int LDA, complex_double* X, int INCX);
	
	// ------------------------------------------------------------------------
	// BLAS 3
	// ------------------------------------------------------------------------

	// GEMM
	void SGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, float          ALPHA, float*          A, int LDA, float*          B, int LDB, float          BETA, float*          C, int LDC );
	void DGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, double         ALPHA, double*         A, int LDA, double*         B, int LDB, double         BETA, double*         C, int LDC );
    void CGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB, complex_float  BETA, complex_float*  C, int LDC );
	void ZGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB, complex_double BETA, complex_double* C, int LDC );

    // TRSM
	void STRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, float          ALPHA, float*          A, int LDA, float*          B, int LDB );
	void DTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, double         ALPHA, double*         A, int LDA, double*         B, int LDB );
    void CTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB );
	void ZTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB );

	// TRSM
	void STRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, float          ALPHA, float*          A, int LDA, float*          B, int LDB );
	void DTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, double         ALPHA, double*         A, int LDA, double*         B, int LDB );
    void CTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB );
	void ZTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB );
	
	// SYMM
	void SSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, float  ALPHA, float*  A, int LDA, float*  B, int LDB, float  BETA, float*  C, int LDC );
	void DSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, double ALPHA, double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC );

	// SYRK
	void SSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, float*          A, int LDA, float  BETA, float*          C, int LDC );
	void DSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, double*         A, int LDA, double BETA, double*         C, int LDC );
    void CHERK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, complex_float*  A, int LDA, float  BETA, complex_float*  C, int LDC );
	void ZHERK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, complex_double* A, int LDA, double BETA, complex_double* C, int LDC );

	// SYR2K
	void SSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, float*  A, int LDA, float*  B, int LDB, float  BETA, float*  C, int LDC );
	void DSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC );

    // ------------------------------------------------------------------------
    // BLAS 1 C++ Overloads
    // ------------------------------------------------------------------------

    // xASUM
    template <typename T> typename real_type<T>::type xASUM( int N, const T*              X, int INCX );
    template <>                         inline  float xASUM( int N, const float*          X, int INCX ) { return SASUM( N, X, INCX ); }
    template <>                         inline double xASUM( int N, const double*         X, int INCX ) { return DASUM( N, X, INCX ); }
    template <>                         inline  float xASUM( int N, const complex_float*  X, int INCX ) { return SCSUM( N, X, INCX ); }
    template <>                         inline double xASUM( int N, const complex_double* X, int INCX ) { return DZSUM( N, X, INCX ); }

    // xAMAX
    template <typename T> int IxAMAX( int N, const T*              X, int INCX );
    template <>    inline int IxAMAX( int N, const double*         X, int INCX ) { return IDAMAX( N, X, INCX); }
    template <>    inline int IxAMAX( int N, const float*          X, int INCX ) { return ISAMAX( N, X, INCX); }
    template <>    inline int IxAMAX( int N, const complex_float*  X, int INCX ) { return ICAMAX( N, X, INCX); }
    template <>    inline int IxAMAX( int N, const complex_double* X, int INCX ) { return IZAMAX( N, X, INCX); }

     // xAXPY
    template <typename T> void xAXPY( int N, T              ALPHA, const T*              X, int INCX, T*              Y, int INCY );
    template <>    inline void xAXPY( int N, float          ALPHA, const float*          X, int INCX, float*          Y, int INCY ) { SAXPY( N, ALPHA, X, INCX, Y, INCY ); }
    template <>    inline void xAXPY( int N, double         ALPHA, const double*         X, int INCX, double*         Y, int INCY ) { DAXPY( N, ALPHA, X, INCX, Y, INCY ); } 
    template <>    inline void xAXPY( int N, complex_float  ALPHA, const complex_float*  X, int INCX, complex_float*  Y, int INCY ) { CAXPY( N, ALPHA, X, INCX, Y, INCY ); } 
    template <>    inline void xAXPY( int N, complex_double ALPHA, const complex_double* X, int INCX, complex_double* Y, int INCY ) { ZAXPY( N, ALPHA, X, INCX, Y, INCY ); } 

    // xCOPY
    template <typename T> void xCOPY( int N, T*              X, int INCX, T*              Y, int INCY );
    template <>    inline void xCOPY( int N, float*          X, int INCX, float*          Y, int INCY ) { SCOPY( N, X, INCX, Y, INCY ); }
    template <>    inline void xCOPY( int N, double*         X, int INCX, double*         Y, int INCY ) { DCOPY( N, X, INCX, Y, INCY ); }
    template <>    inline void xCOPY( int N, complex_float*  X, int INCX, complex_float*  Y, int INCY ) { CCOPY( N, X, INCX, Y, INCY ); }
    template <>    inline void xCOPY( int N, complex_double* X, int INCX, complex_double* Y, int INCY ) { ZCOPY( N, X, INCX, Y, INCY ); }

    // xDOT
    template <typename T, typename U>             U xDOT( int N, const T*              X, int INCX, const T*              Y, int INCY );
    template <>               inline          float xDOT( int N, const float*          X, int INCX, const float*          Y, int INCY ) { return SDOT ( N, X, INCX, Y, INCY ); }
    template <>               inline         double xDOT( int N, const float*          X, int INCX, const float*          Y, int INCY ) { return DSDOT( N, X, INCX, Y, INCY ); }
    template <>               inline         double xDOT( int N, const double*         X, int INCX, const double*         Y, int INCY ) { return DDOT ( N, X, INCX, Y, INCY ); }
    template <>               inline  complex_float xDOT( int N, const complex_float*  X, int INCX, const complex_float*  Y, int INCY ) { return CDOTC( N, X, INCX, Y, INCY ); }
    template <>               inline complex_double xDOT( int N, const complex_double* X, int INCX, const complex_double* Y, int INCY ) { return ZDOTC( N, X, INCX, Y, INCY ); }

    // xNRM2
    template <typename T> typename real_type<T>::type xNRM2( int N, const T*              X,  int INCX );
    template <>                         inline  float xNRM2( int N, const float*          X,  int INCX ) { return  SNRM2( N, X, INCX); }
    template <>                         inline double xNRM2( int N, const double*         X,  int INCX ) { return  DNRM2( N, X, INCX); }
    template <>                         inline  float xNRM2( int N, const complex_float*  X,  int INCX ) { return SCNRM2( N, X, INCX); }
    template <>                         inline double xNRM2( int N, const complex_double* X,  int INCX ) { return DZNRM2( N, X, INCX); } 
    
    // xROT
    template <typename T> void xROT( int N, T*      X, int INCX, T*      Y, int INCY, T      C, T      S );
    template <>    inline void xROT( int N, double* X, int INCX, double* Y, int INCY, double C, double S ) { DROT( N, X, INCX, Y, INCY, C, S ); } 
    template <>    inline void xROT( int N, float*  X, int INCX, float*  Y, int INCY, float  C, float  S ) { SROT( N, X, INCX, Y, INCY, C, S ); }

    // xSCAL
    template <typename T, typename U> void xSCAL( int N, U              ALPHA, T*              X, int INCX );  
    template <>                inline void xSCAL( int N, float          ALPHA, float*          X, int INCX ) {  SSCAL( N, ALPHA, X, INCX ); }
    template <>                inline void xSCAL( int N, double         ALPHA, double*         X, int INCX ) {  DSCAL( N, ALPHA, X, INCX ); }
    template <>                inline void xSCAL( int N, float          ALPHA, complex_float*  X, int INCX ) { CSSCAL( N, ALPHA, X, INCX ); }
    template <>                inline void xSCAL( int N, complex_float  ALPHA, complex_float*  X, int INCX ) {  CSCAL( N, ALPHA, X, INCX ); }
    template <>                inline void xSCAL( int N, double         ALPHA, complex_double* X, int INCX ) { ZDSCAL( N, ALPHA, X, INCX ); }
    template <>                inline void xSCAL( int N, complex_double ALPHA, complex_double* X, int INCX ) {  ZSCAL( N, ALPHA, X, INCX ); }

    // SWAP
    template <typename T> void xSWAP( int N, T*              X, int INCX, T*              Y, int INCY );
    template <>    inline void xSWAP( int N, float*          X, int INCX, float*          Y, int INCY ) { SSWAP( N, X, INCX, Y, INCY ); }
    template <>    inline void xSWAP( int N, double*         X, int INCX, double*         Y, int INCY ) { DSWAP( N, X, INCX, Y, INCY ); }
    template <>    inline void xSWAP( int N, complex_float*  X, int INCX, complex_float*  Y, int INCY ) { CSWAP( N, X, INCX, Y, INCY ); }
    template <>    inline void xSWAP( int N, complex_double* X, int INCX, complex_double* Y, int INCY ) { ZSWAP( N, X, INCX, Y, INCY ); }

	// ------------------------------------------------------------------------
    // BLAS 2 C++ Overloads
    // ------------------------------------------------------------------------

	// xGER                                                       
    template <typename T> void xGER( int M, int N, T              ALPHA, const T*              X, int INCX, const T*              Y, int INCY, T*              A, int LDA );
    template <>    inline void xGER( int M, int N, float          ALPHA, const float*          X, int INCX, const float*          Y, int INCY, float*          A, int LDA ) {  SGER( M, N, ALPHA, X, INCX, Y, INCY, A, LDA ); }
    template <>    inline void xGER( int M, int N, double         ALPHA, const double*         X, int INCX, const double*         Y, int INCY, double*         A, int LDA ) {  DGER( M, N, ALPHA, X, INCX, Y, INCY, A, LDA ); }
    template <>    inline void xGER( int M, int N, complex_float  ALPHA, const complex_float*  X, int INCX, const complex_float*  Y, int INCY, complex_float*  A, int LDA ) { CGERC( M, N, ALPHA, X, INCX, Y, INCY, A, LDA ); }
    template <>    inline void xGER( int M, int N, complex_double ALPHA, const complex_double* X, int INCX, const complex_double* Y, int INCY, complex_double* A, int LDA ) { ZGERC( M, N, ALPHA, X, INCX, Y, INCY, A, LDA ); }

	// xGEMV
	template <typename T> void xGEMV( enum class transpose TRANSA, int M, int N, T              ALPHA, const T*              A, int LDA, const T*              X, int INCX, T              BETA, T*              Y, int INCY );
	template <>    inline void xGEMV( enum class transpose TRANSA, int M, int N, double         ALPHA, const double*         A, int LDA, const double*         X, int INCX, double         BETA, double*         Y, int INCY ) { DGEMV(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
	template <>    inline void xGEMV( enum class transpose TRANSA, int M, int N, float          ALPHA, const float*          A, int LDA, const float*          X, int INCX, float          BETA, float*          Y, int INCY ) { SGEMV(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
    template <>    inline void xGEMV( enum class transpose TRANSA, int M, int N, complex_float  ALPHA, const complex_float*  A, int LDA, const complex_float*  X, int INCX, complex_float  BETA, complex_float*  Y, int INCY ) { CGEMV(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
	template <>    inline void xGEMV( enum class transpose TRANSA, int M, int N, complex_double ALPHA, const complex_double* A, int LDA, const complex_double* X, int INCX, complex_double BETA, complex_double* Y, int INCY ) { ZGEMV(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }

    // xSYR
    template <typename T> void xSYR( enum class uplo UPLO, int N, typename real_type<T>::type ALPHA, const T*              X, int INCX, T*              A, int LDA );
    template <>    inline void xSYR( enum class uplo UPLO, int N, float                       ALPHA, const float*          X, int INCX, float*          A, int LDA ) { SSYR( UPLO, N, ALPHA, X, INCX, A, LDA ); }
    template <>    inline void xSYR( enum class uplo UPLO, int N, double                      ALPHA, const double*         X, int INCX, double*         A, int LDA ) { DSYR( UPLO, N, ALPHA, X, INCX, A, LDA ); }
    template <>    inline void xSYR( enum class uplo UPLO, int N, float                       ALPHA, const complex_float*  X, int INCX, complex_float*  A, int LDA ) { CHER( UPLO, N, ALPHA, X, INCX, A, LDA ); }
    template <>    inline void xSYR( enum class uplo UPLO, int N, double                      ALPHA, const complex_double* X, int INCX, complex_double* A, int LDA ) { ZHER( UPLO, N, ALPHA, X, INCX, A, LDA ); }

    // xSYMV
    template <typename T> void xSYMV( enum class uplo UPLO, int N, T              ALPHA, T*              A, int LDA, T*              X, int INCX, T              BETA, T*              Y, int INCY );
    template <>    inline void xSYMV( enum class uplo UPLO, int N, float          ALPHA, float*          A, int LDA, float*          X, int INCX, float          BETA, float*          Y, int INCY ) { SSYMV(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
    template <>    inline void xSYMV( enum class uplo UPLO, int N, double         ALPHA, double*         A, int LDA, double*         X, int INCX, double         BETA, double*         Y, int INCY ) { DSYMV(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
    template <>    inline void xSYMV( enum class uplo UPLO, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  X, int INCX, complex_float  BETA, complex_float*  Y, int INCY ) { CHEMV(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
    template <>    inline void xSYMV( enum class uplo UPLO, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* X, int INCX, complex_double BETA, complex_double* Y, int INCY ) { ZHEMV(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY); }
	
    // xTRMV
    template <typename T> void xTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, T*              A, int LDA, T*              X, int INCX );
    template <>    inline void xTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, float*          A, int LDA, float*          X, int INCX ) { STRMV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }
    template <>    inline void xTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, double*         A, int LDA, double*         X, int INCX ) { DTRMV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }
    template <>    inline void xTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_float*  A, int LDA, complex_float*  X, int INCX ) { CTRMV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }
    template <>    inline void xTRMV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_double* A, int LDA, complex_double* X, int INCX ) { ZTRMV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }

    // xTRSV
    template <typename T> void xTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, T*              A, int LDA, T*              X, int INCX );
    template <>    inline void xTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, float*          A, int LDA, float*          X, int INCX ) { STRSV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }
    template <>    inline void xTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, double*         A, int LDA, double*         X, int INCX ) { DTRSV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }
    template <>    inline void xTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_float*  A, int LDA, complex_float*  X, int INCX ) { CTRSV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }
    template <>    inline void xTRSV( enum class uplo UPLO, enum class transpose TRANS, enum class diag DIAG, int N, complex_double* A, int LDA, complex_double* X, int INCX ) { ZTRSV(UPLO, TRANS, DIAG, N, A, LDA, X, INCX); }

	// ------------------------------------------------------------------------
    // BLAS 3 C++ Overloads
    // ------------------------------------------------------------------------

	// xGEMM
	template <typename T> void xGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, T              ALPHA, T*              A, int LDA, T*              B, int LDB, T              BETA, T*              C, int LDC );                                                
	template <>    inline void xGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, float          ALPHA, float*          A, int LDA, float*          B, int LDB, float          BETA, float*          C, int LDC ) { SGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }
	template <>    inline void xGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, double         ALPHA, double*         A, int LDA, double*         B, int LDB, double         BETA, double*         C, int LDC ) { DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }
    template <>    inline void xGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB, complex_float  BETA, complex_float*  C, int LDC ) { CGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }
	template <>    inline void xGEMM( enum class transpose TRANSA, enum class transpose TRANSB, int M, int N, int K, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB, complex_double BETA, complex_double* C, int LDC ) { ZGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }

	// xTRMM
	template <typename T> void xTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, T              ALPHA, T*              A, int LDA, T*              B, int LDB );
	template <>    inline void xTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, float          ALPHA, float*          A, int LDA, float*          B, int LDB ) { STRMM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); } 
	template <>    inline void xTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, double         ALPHA, double*         A, int LDA, double*         B, int LDB ) { DTRMM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); }
    template <>    inline void xTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB ) { CTRMM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); } 
	template <>    inline void xTRMM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB ) { ZTRMM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); }

    // xTRSM
	template <typename T> void xTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, T              ALPHA, T*              A, int LDA, T*              B, int LDB );
	template <>    inline void xTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, float          ALPHA, float*          A, int LDA, float*          B, int LDB ) { STRSM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); } 
	template <>    inline void xTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, double         ALPHA, double*         A, int LDA, double*         B, int LDB ) { DTRSM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); }
    template <>    inline void xTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_float  ALPHA, complex_float*  A, int LDA, complex_float*  B, int LDB ) { CTRSM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); } 
	template <>    inline void xTRSM( enum class side SIDE, enum class uplo UPLO, enum class transpose TRANSA, enum class diag DIAG, int M, int N, complex_double ALPHA, complex_double* A, int LDA, complex_double* B, int LDB ) { ZTRSM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB); }

	// xSYMM
	template <typename T> void xSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, T      ALPHA, T*      A, int LDA, T*      B, int LDB, T      BETA, T*      C, int LDC);
	template <>    inline void xSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, float  ALPHA, float*  A, int LDA, float*  B, int LDB, float  BETA, float*  C, int LDC) { SSYMM( SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }
	template <>    inline void xSYMM( enum class side SIDE, enum class uplo UPLO, int M, int N, double ALPHA, double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC) { DSYMM( SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }

	// xSYRK
	template <typename T> void xSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, typename real_type<T>::type ALPHA, T*              A, int LDA, typename real_type<T>::type      BETA, T*              C, int LDC);
	template <>    inline void xSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float                       ALPHA, float*          A, int LDA, float                            BETA, float*          C, int LDC) { SSYRK( UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC); }
	template <>    inline void xSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double                      ALPHA, double*         A, int LDA, double                           BETA, double*         C, int LDC) { DSYRK( UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC); }
    template <>    inline void xSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float                       ALPHA, complex_float*  A, int LDA, float                            BETA, complex_float*  C, int LDC) { CHERK( UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC); }
	template <>    inline void xSYRK( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double                      ALPHA, complex_double* A, int LDA, double                           BETA, complex_double* C, int LDC) { ZHERK( UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC); }

	// xSYR2K
	template <typename T> void xSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, T      ALPHA, T*      A, int LDA, T*      B, int LDB, T      BETA, T*      C, int LDC);
	template <>    inline void xSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, float  ALPHA, float*  A, int LDA, float*  B, int LDB, float  BETA, float*  C, int LDC) { SSYR2K( UPLO, TRANS, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }
	template <>    inline void xSYR2K( enum class uplo UPLO, enum class transpose TRANS, int N, int K, double ALPHA, double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC) { DSYR2K( UPLO, TRANS, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); }

} // namespace cblas

#endif // CBLAS_WRAPPER_H
