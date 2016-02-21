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
 * ampblas.h 
 *
 * BLAS levels 1,2,3 library header for C++ AMP.
 *
 * This file contains C++ template BLAS APIs for generic data types.
 *
 *---------------------------------------------------------------------------*/

#ifndef AMPBLAS_H
#define AMPBLAS_H

// BLAS 1
#include "detail/amax.h"
#include "detail/asum.h"
#include "detail/axpy.h"
#include "detail/copy.h"
#include "detail/dot.h"
#include "detail/nrm2.h"
#include "detail/rot.h"
#include "detail/scal.h"
#include "detail/swap.h"

// BLAS 2
#include "detail/gemv.h"
#include "detail/ger.h"
#include "detail/symv.h"
#include "detail/syr.h"
#include "detail/trmv.h"
#include "detail/trsv.h"

// BLAS 3
#include "detail/gemm.h"
#include "detail/symm.h"
#include "detail/syr2k.h"
#include "detail/syrk.h"
#include "detail/trmm.h"
#include "detail/trsm.h"

#endif //AMPBLAS_H
