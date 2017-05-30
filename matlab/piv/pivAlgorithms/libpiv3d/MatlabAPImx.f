!*************************************************************************************
! 
!  MATLAB (R) is a trademark of The Mathworks (R) Corporation
! 
!  Filename:    MatlabAPImx.f
!  Programmer:  James Tursa
!  Version:     1.01
!  Date:        December 11, 2009
!  Copyright:   (c) 2009 by James Tursa, All Rights Reserved
! 
!   This code uses the BSD License:
! 
!   Redistribution and use in source and binary forms, with or without 
!   modification, are permitted provided that the following conditions are 
!   met:
! 
!      * Redistributions of source code must retain the above copyright 
!        notice, this list of conditions and the following disclaimer.
!      * Redistributions in binary form must reproduce the above copyright 
!        notice, this list of conditions and the following disclaimer in 
!        the documentation and/or other materials provided with the distribution
!       
!   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
!   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
!   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
!   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
!   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
!   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
!   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
!   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
!   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
!   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
!   POSSIBILITY OF SUCH DAMAGE.
!
!   The mx interfaces included are (see MATLAB doc for use):
!
!      mxAddField
!      mxCalcSingleSubscript
!      mxCalloc
!      mxClassIDFromClassName
!      mxCopyCharacterToPtr  ! The Ptr is a pointer to a C-style string
!      mxCopyComplex16ToPtr
!      mxCopyComplex8ToPtr
!      mxCopyInteger1ToPtr
!      mxCopyInteger2ToPtr
!      mxCopyInteger4ToPtr
!      mxCopyPtrToCharacter  ! The Ptr is a pointer to a C-style string
!      mxCopyPtrToComplex16
!      mxCopyPtrToComplex8
!      mxCopyPtrToInteger1
!      mxCopyPtrToInteger2
!      mxCopyPtrToInteger4
!      mxCopyPtrToPtrArray
!      mxCopyPtrToReal4
!      mxCopyPtrToReal8
!      mxCopyReal4ToPtr
!      mxCopyReal8ToPtr
!      mxCreateCellArray
!      mxCreateCellMatrix
!      mxCreateCharArray
!      mxCreateCharMatrixFromStrings
!      mxCreateDoubleMatrix
!      mxCreateDoubleScalar
!      mxCreateNumericArray
!      mxCreateNumericMatrix
!      mxCreateSparse
!      mxCreateString
!      mxCreateStructArray
!      mxCreateStructMatrix
!      mxDestroyArray
!      mxDuplicateArray
!      mxFree
!      mxGetCell
!      mxGetClassID
!      mxGetClassName
!      mxGetData
!      mxGetDimensions
!      mxGetElementSize
!      mxGetEps
!      mxGetField
!      mxGetFieldByNumber
!      mxGetFieldNameByNumber
!      mxGetFieldNumber
!      mxGetImagData
!      mxGetInf
!      mxGetIr
!      mxGetJc
!      mxGetM
!      mxGetN
!      mxGetNaN
!      mxGetNumberOfDimensions
!      mxGetNumberOfElements
!      mxGetNumberOfFields
!      mxGetNzmax
!      mxGetPi
!      mxGetPr
!      mxGetProperty
!      mxGetScalar
!      mxGetString
!      mxIsCell
!      mxIsChar
!      mxIsClass
!      mxIsComplex
!      mxIsDouble
!      mxIsEmpty
!      mxIsFinite
!      mxIsFromGlobalWS
!      mxIsInf
!      mxIsInt16
!      mxIsInt32
!      mxIsInt64
!      mxIsInt8
!      mxIsLogical
!      mxIsNaN
!      mxIsNumeric
!      mxIsSingle
!      mxIsSparse
!      mxIsStruct
!      mxIsUint16
!      mxIsUint32
!      mxIsUint64
!      mxIsUint8
!      mxMalloc
!      mxRealloc
!      mxRemoveField
!      mxSetCell
!      mxSetData
!      mxSetDimensions
!      mxSetField
!      mxSetFieldByNumber
!      mxSetImagData
!      mxSetIr
!      mxSetJc
!      mxSetM
!      mxSetN
!      mxSetNzmax
!      mxSetPi
!      mxSetPr
!
!   The extra mx routines included are (see MATLAB doc for use):
!
!      mxCopyCharsToCharacter  ! works with result of mxGetChars
!      mxCopyCharacterToChars  ! works with result of mxGetChars
!      mxCopyComplex32ToPtr
!      mxCopyInteger8ToPtr
!      mxCopyPtrToComplex32
!      mxCopyPtrToInteger8
!      mxCopyPtrToReal16
!      mxCopyReal16ToPtr
!      mxGetChars
!
!   Addition logical routines included are (see doc below for use):
!
!     mxCopyPtrToLogical
!     mxCopyPtrToLogical1
!     mxCopyPtrToLogical2
!     mxCopyPtrToLogical4
!     mxCopyLogicalToPtr
!     mxCopyLogical1ToPtr
!     mxCopyLogical2ToPtr
!     mxCopyLogical3ToPtr
!     mxCreateLogicalArray
!     mxCreateLogicalMatrix
!     mxCreateLogicalScalar
!     mxCreateLogical1Scalar
!     mxCreateLogical2Scalar
!     mxCreateLogical4Scalar
!     mxGetLogicalScalar
!     mxGetLogical1Scalar
!     mxGetLogical2Scalar
!     mxGetLogical4Scalar
!     mxGetLogicals
!     mxIsLogicalScalar
!     mxIsLogicalScalarTrue
!
!   Note: If your compiler does not support real(16) or complex(16)
!         variable types, you will need to compile with the -DSMALLMODEL
!         option on the mex command line.
!
!   The Fortran Pointer fp routines are:
!
!     fpGetPr1Double aka fpGetPr1
!     fpGetPr2Double aka fpGetPr2 aka fpGetPr
!     fpGetPr3Double aka fpGetPr3
!     fpGetPr4Double aka fpGetPr4
!     fpGetPr5Double aka fpGetPr5
!     fpGetPr6Double aka fpGetPr6
!     fpGetPr7Double aka fpGetPr7
!
!     fpGetPi1Double aka fpGetPi1
!     fpGetPi2Double aka fpGetPi2 aka fpGetPi
!     fpGetPi3Double aka fpGetPi3
!     fpGetPi4Double aka fpGetPi4
!     fpGetPi5Double aka fpGetPi5
!     fpGetPi6Double aka fpGetPi6
!     fpGetPi7Double aka fpGetPi7
!
!     fpGetPrCopy1Double aka fpGetPrCopy1
!     fpGetPrCopy2Double aka fpGetPrCopy2 aka fpGetPrCopy
!     fpGetPrCopy3Double aka fpGetPrCopy3
!     fpGetPrCopy4Double aka fpGetPrCopy4
!     fpGetPrCopy5Double aka fpGetPrCopy5
!     fpGetPrCopy6Double aka fpGetPrCopy6
!     fpGetPrCopy7Double aka fpGetPrCopy7
!
!     fpGetPiCopy1Double aka fpGetPiCopy1
!     fpGetPiCopy2Double aka fpGetPiCopy2 aka fpGetPiCopy
!     fpGetPiCopy3Double aka fpGetPiCopy3
!     fpGetPiCopy4Double aka fpGetPiCopy4
!     fpGetPiCopy5Double aka fpGetPiCopy5
!     fpGetPiCopy6Double aka fpGetPiCopy6
!     fpGetPiCopy7Double aka fpGetPiCopy7
!
!     fpGetPzCopy1Double aka fpGetPrCopy1
!     fpGetPzCopy2Double aka fpGetPrCopy2 aka fpGetPzCopy
!     fpGetPzCopy3Double aka fpGetPrCopy3
!     fpGetPzCopy4Double aka fpGetPrCopy4
!     fpGetPzCopy5Double aka fpGetPrCopy5
!     fpGetPzCopy6Double aka fpGetPrCopy6
!     fpGetPzCopy7Double aka fpGetPrCopy7
!
!     fpReal, a generic name for all of:
!     fpReal1Double aka fpReal1
!     fpReal2Double aka fpReal2
!     fpReal3Double aka fpReal3
!     fpReal4Double aka fpReal4
!     fpReal5Double aka fpReal5
!     fpReal6Double aka fpReal6
!     fpReal7Double aka fpReal7
!    
!     fpImag, a generic name for all of:
!     fpImag1Double aka fpImag1
!     fpImag2Double aka fpImag2
!     fpImag3Double aka fpImag3
!     fpImag4Double aka fpImag4
!     fpImag5Double aka fpImag5
!     fpImag6Double aka fpImag6
!     fpImag7Double aka fpImag7
!
!     fpAllocate, a generic name for all of:
!     fpAllocate1Double aka fpAllocate1
!     fpAllocate2Double aka fpAllocate2
!     fpAllocate3Double aka fpAllocate3
!     fpAllocate4Double aka fpAllocate4
!     fpAllocate5Double aka fpAllocate5
!     fpAllocate6Double aka fpAllocate6
!     fpAllocate7Double aka fpAllocate7
!
!     fpDeallocate, a generic name for all of:
!     fpDeallocate1Double aka fpDeallocate1
!     fpDeallocate2Double aka fpDeallocate2
!     fpDeallocate3Double aka fpDeallocate3
!     fpDeallocate4Double aka fpDeallocate4
!     fpDeallocate5Double aka fpDeallocate5
!     fpDeallocate6Double aka fpDeallocate6
!     fpDeallocate7Double aka fpDeallocate7
!
!     fpAllocateZ, a generic name for all of:
!     fpAllocateZ1Double aka fpAllocateZ1
!     fpAllocateZ2Double aka fpAllocateZ2
!     fpAllocateZ3Double aka fpAllocateZ3
!     fpAllocateZ4Double aka fpAllocateZ4
!     fpAllocateZ5Double aka fpAllocateZ5
!     fpAllocateZ6Double aka fpAllocateZ6
!     fpAllocateZ7Double aka fpAllocateZ7
!
!     fzDeallocate, a generic name for all of:
!     fzDeallocate1Double aka fzDeallocate1
!     fzDeallocate2Double aka fzDeallocate2
!     fzDeallocate3Double aka fzDeallocate3
!     fzDeallocate4Double aka fzDeallocate4
!     fzDeallocate5Double aka fzDeallocate5
!     fzDeallocate6Double aka fzDeallocate6
!     fzDeallocate7Double aka fzDeallocate7
!
!     fpReshape, a generic name for all of:
!     fpReshape11Double aka fpReshape11
!     fpReshape12Double aka fpReshape12
!     fpReshape13Double aka fpReshape13
!     fpReshape14Double aka fpReshape14
!     fpReshape15Double aka fpReshape15
!     fpReshape16Double aka fpReshape16
!     fpReshape17Double aka fpReshape17
!     fpReshape21Double aka fpReshape21
!     fpReshape22Double aka fpReshape22
!     fpReshape23Double aka fpReshape23
!     fpReshape24Double aka fpReshape24
!     fpReshape25Double aka fpReshape25
!     fpReshape26Double aka fpReshape26
!     fpReshape27Double aka fpReshape27
!     fpReshape31Double aka fpReshape31
!     fpReshape32Double aka fpReshape32
!     fpReshape33Double aka fpReshape33
!     fpReshape34Double aka fpReshape34
!     fpReshape35Double aka fpReshape35
!     fpReshape36Double aka fpReshape36
!     fpReshape37Double aka fpReshape37
!     fpReshape41Double aka fpReshape41
!     fpReshape42Double aka fpReshape42
!     fpReshape43Double aka fpReshape43
!     fpReshape44Double aka fpReshape44
!     fpReshape45Double aka fpReshape45
!     fpReshape46Double aka fpReshape46
!     fpReshape47Double aka fpReshape47
!     fpReshape51Double aka fpReshape51
!     fpReshape52Double aka fpReshape52
!     fpReshape53Double aka fpReshape53
!     fpReshape54Double aka fpReshape54
!     fpReshape55Double aka fpReshape55
!     fpReshape56Double aka fpReshape56
!     fpReshape57Double aka fpReshape57
!     fpReshape61Double aka fpReshape61
!     fpReshape62Double aka fpReshape62
!     fpReshape63Double aka fpReshape63
!     fpReshape64Double aka fpReshape64
!     fpReshape65Double aka fpReshape65
!     fpReshape66Double aka fpReshape66
!     fpReshape67Double aka fpReshape67
!     fpReshape71Double aka fpReshape71
!     fpReshape72Double aka fpReshape72
!     fpReshape73Double aka fpReshape73
!     fpReshape74Double aka fpReshape74
!     fpReshape75Double aka fpReshape75
!     fpReshape76Double aka fpReshape76
!     fpReshape77Double aka fpReshape77
!
!     fzReshape, a generic name for all of:
!     fzReshape11Double aka fzReshape11
!     fzReshape12Double aka fzReshape12
!     fzReshape13Double aka fzReshape13
!     fzReshape14Double aka fzReshape14
!     fzReshape15Double aka fzReshape15
!     fzReshape16Double aka fzReshape16
!     fzReshape17Double aka fzReshape17
!     fzReshape21Double aka fzReshape21
!     fzReshape22Double aka fzReshape22
!     fzReshape23Double aka fzReshape23
!     fzReshape24Double aka fzReshape24
!     fzReshape25Double aka fzReshape25
!     fzReshape26Double aka fzReshape26
!     fzReshape27Double aka fzReshape27
!     fzReshape31Double aka fzReshape31
!     fzReshape32Double aka fzReshape32
!     fzReshape33Double aka fzReshape33
!     fzReshape34Double aka fzReshape34
!     fzReshape35Double aka fzReshape35
!     fzReshape36Double aka fzReshape36
!     fzReshape37Double aka fzReshape37
!     fzReshape41Double aka fzReshape41
!     fzReshape42Double aka fzReshape42
!     fzReshape43Double aka fzReshape43
!     fzReshape44Double aka fzReshape44
!     fzReshape45Double aka fzReshape45
!     fzReshape46Double aka fzReshape46
!     fzReshape47Double aka fzReshape47
!     fzReshape51Double aka fzReshape51
!     fzReshape52Double aka fzReshape52
!     fzReshape53Double aka fzReshape53
!     fzReshape54Double aka fzReshape54
!     fzReshape55Double aka fzReshape55
!     fzReshape56Double aka fzReshape56
!     fzReshape57Double aka fzReshape57
!     fzReshape61Double aka fzReshape61
!     fzReshape62Double aka fzReshape62
!     fzReshape63Double aka fzReshape63
!     fzReshape64Double aka fzReshape64
!     fzReshape65Double aka fzReshape65
!     fzReshape66Double aka fzReshape66
!     fzReshape67Double aka fzReshape67
!     fzReshape71Double aka fzReshape71
!     fzReshape72Double aka fzReshape72
!     fzReshape73Double aka fzReshape73
!     fzReshape74Double aka fzReshape74
!     fzReshape75Double aka fzReshape75
!     fzReshape76Double aka fzReshape76
!     fzReshape77Double aka fzReshape77
!
!     fpGetCells1
!     fpGetCells2
!     fpGetCells3
!     fpGetCells4
!     fpGetCells5
!     fpGetCells6
!     fpGetCells7
!
!     fpGetDimensions
!
!     random_number
!
!  Definitions of the routines:
!
!-----
!     fpGetPr1
!
!     function fpGetPr1Double( mx ) result(fp)
!     real(8), pointer :: fp(:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     1D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a non-empty double class variable. The size of the returned
!     Fortran pointer fp is the total number of actual elements of the real
!     data area of the mxArray. In the case of a sparse mxArray input, the
!     size of fp is the total number of non-zero elements, not nzmax. This
!     is the only fpGetPr_Double routine that can be called with sparse
!     matrices, since all of the other fpGetPr_Double routines require full
!     arrays as inputs. If the input mxArray is not of the required class,
!     a null pointer is returned. If the input is multi-dimensional, the
!     routine still returns the rank 1 pointer as if all of the elements
!     were in one column. i.e., there is never a mismatch between rank when
!     using this routine.
!
!     Function fpGetPr1 is a generic name for fpGetPr1Double.
!-----
!     fpGetPr2
!
!     function fpGetPr2Double( mx ) result(fp)
!     real(8), pointer :: fp(:,:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     2D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a 1D-2D non-empty full double class variable. The size of the
!     returned Fortran pointer fp is the same as the size of the the real
!     data area of the mxArray. If the input mxArray is not of the required
!     class or rank, a null pointer is returned.
!
!     Functions fpGetPr and fpGetPr2 are generic names for fpGetPr2Double.
!-----
!     fpGetPr3
!
!     function fpGetPr3Double( mx ) result(fp)
!     real(8), pointer :: fp(:,:,:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     3D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a 1D-3D non-empty full double class variable. The size of the
!     returned Fortran pointer fp is the same as the size of the the real
!     data area of the mxArray, padded with 1's if necessary. If the input
!     mxArray is not of the required class or rank, a null pointer is returned.
!
!     Function fpGetPr3 is a generic name for fpGetPr3Double.
!-----
!     fpGetPr4
!
!     function fpGetPr4Double( mx ) result(fp)
!     real(8), pointer :: fp(:,:,:,:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     4D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a 1D-4D non-empty full double class variable. The size of the
!     returned Fortran pointer fp is the same as the size of the the real
!     data area of the mxArray, padded with 1's if necessary. If the input
!     mxArray is not of the required class or rank, a null pointer is returned.
!
!     Function fpGetPr4 is a generic name for fpGetPr4Double.
!-----
!     fpGetPr5
!
!     function fpGetPr5Double( mx ) result(fp)
!     real(8), pointer :: fp(:,:,:,:,:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     5D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a 1D-5D non-empty full double class variable. The size of the
!     returned Fortran pointer fp is the same as the size of the the real
!     data area of the mxArray, padded with 1's if necessary. If the input
!     mxArray is not of the required class or rank, a null pointer is returned.
!
!     Function fpGetPr5 is a generic name for fpGetPr5Double.
!-----
!     fpGetPr6
!
!     function fpGetPr6Double( mx ) result(fp)
!     real(8), pointer :: fp(:,:,:,:,:,:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     6D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a 1D-6D non-empty full double class variable. The size of the
!     returned Fortran pointer fp is the same as the size of the the real
!     data area of the mxArray, padded with 1's if necessary. If the input
!     mxArray is not of the required class or rank, a null pointer is returned.
!
!     Function fpGetPr6 is a generic name for fpGetPr6Double.
!-----
!     fpGetPr7
!
!     function fpGetPr7Double( mx ) result(fp)
!     real(8), pointer :: fp(:,:,:,:,:,:,:)
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     7D Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a 1D-7D non-empty full double class variable. The size of the
!     returned Fortran pointer fp is the same as the size of the the real
!     data area of the mxArray, padded with 1's if necessary. If the input
!     mxArray is not of the required class or rank, a null pointer is returned.
!
!     Function fpGetPr7 is a generic name for fpGetPr7Double.
!-----
!     fpGetPr0
!
!     function fpGetPr0Double( mx ) result(fp)
!     real(8), pointer :: fp
!     mwPointer, intent(in) :: mx
!
!     This function takes a pointer to an mxArray as input and returns a
!     scalar Fortran pointer to the real data area of the mxArray. The mxArray
!     must be a scalar. If the input mxArray is not of the required class or
!     rank, a null pointer is returned.
!
!     Function fpGetPr0 is a generic name for fpGetPr0Double.
!-----
!     fpGetPi__
!
!     For each fpGetPr__ function, there is a corresponding function fpGetPi__
!     function that points to the imaginary data. If the input mxArray variable
!     is pure real, then the fpGetPi__ functions return a null pointer.
!-----
!     fpGetPrCopy__
!     fpGetPiCopy__
!
!     For each fpGetPr__ and fpGetPi__ function, there are corresponding functions
!     fpGetPrCopy__ and fpGetPiCopy__ functions. The difference is that for the
!     __Copy__ functions, memory is dynamically allocated and the mxArray data
!     is copied into that dynamically allocated memory. A pointer to that
!     dynamically allocated memory is returned. This memory should be deallocated
!     with the fpDeallocate function when you are done using it. However, since it
!     is allocated with the MATLAB API mxMalloc function, it will be automatically
!     garbage collected when the mex function returns to MATLAB if you do not do
!     so manually.
!-----
!     fpGetPzCopy__
!
!     For each fpGetPr__ and fpGetPi__ function pair, there is a corresponding
!     function fpGetPzCopy__ that returns a complex pointer. If the mxArray
!     argument is pure real, then zeros will be used for the imaginary part.
!     For these functions, memory is dynamically allocated and the mxArray data
!     is copied into that dynamically allocated memory. A pointer to that
!     dynamically allocated memory is returned. This memory should be deallocated
!     with the fpDeallocate function when you are done using it. However, since it
!     is allocated with the MATLAB API mxMalloc function, it will be automatically
!     garbage collected when the mex function returns to MATLAB if you do not do
!     so manually.
!-----
!     fpReal
!
!     The fpReal function takes a complex array input and returns a pointer to
!     the real portion of the array. Note that the pointer returned is a real
!     pointer, not a complex pointer. i.e., the pointer is actually pointing at
!     interleaved memory (the real part) of the input variable directly. No copy
!     is made.
!-----
!     fpImag
!
!     Same as fpReal except points at the imaginary part of the complex array
!     input. Note that the pointer returned is a real pointer, not a complex
!     pointer. i.e., the pointer is actually pointing at interleaved memory
!     (the imaginary part) of the input variable directly. No copy is made.
!-----
!     fpAllocate
!
!     The generic function fpAllocate takes 0 - 7 input arguments specifying the
!     array size to allocate. It returns a Fortran pointer to the allocated
!     memory. The rank of the returned pointer is determined by the number of
!     inputs. One input will return a rank 1 pointer, two inputs will return a
!     rank 2 pointer, etc. The memory must be deallocated with the fpDeallocate
!     routine when you are done with it. All of the fpAllocate specific functions
!     use mxMalloc in the background, so memory that is not part of a plhs(*)
!     variable will be garbage collected when the mex routine returns to MATLAB.
!     fpAllocate allocates space for doubles (i.e., real(8) or real*8). 0 arguments
!     means allocate a scalar.
!-----
!     fpAllocateZ
!
!     Same as fpAllocate except fpAllocateZ retunrs a complex pointer.
!-----
!     fpDeallocate
!
!     The generic function fpDeallocate takes a Fortran pointer that was allocated
!     with the fpAllocate or fpAllocateZ functions and deallocates the memory using
!     mxFree in the background. The passed pointer is also nullified.
!-----
!     fpReshape
!
!     The generic function fpReshape takes a Fortran pointer or Fortran array as
!     input, along with the desired new shape values, and returns a Fortran pointer
!     to the same memory as the input but with the new shape. The restristions are
!     that the total number of elements must be the same, and the memory must be
!     contiguous. i.e., you cannot pass a noncontiguous array slice to this routine.
!     If there is a problem with a size mismatch or non-contiguous memory, the
!     routine returns a null pointer. fpReshape works with double (i.e., real(8) or
!     real*8) Fortran pointers or regular Fortran double arrays.
!-----
!     fpGetCells1
!
!     function fpGetCells1( mx ) result(fp)
!     mwPointer, pointer :: fp(:)
!     mwPointer, intent(in) :: mx
!
!     The function fpGetCells1 returns a 1D Fortran pointer to the data area of a
!     cell array. The data area actually contains pointers to mxArray variables, so
!     the returned pointer should be used as an array of mxArray variables pointers.
!     This routine works regardless of the number of dimensions of the input mx.
!-----
!     fpGetCells2 ... fpGetCells7
!
!     These routines work the same way as fpGetCells1 except that they return pointers
!     to 2D thru 7D dimensioned arrays.
!-----
!     fpGetDimensions
!
!     function fpGetDimensions( mx ) result(fp)
!     mwSize, pointer :: fp(:)
!     mwPointer, intent(in) :: mx
!
!     The function fpGetDimensions returns a 1D Fortran pointer to the dimensions
!     area of the input mxArray variable mx. This should be regarded as read-only
!     unless you actually want to change the dimensions of the original mxArray.
!-----
!     random_number
!
!     subroutine random_number1( z )
!     complex(8), intent(out) :: z(:)
!
!     The subroutine random_number is a generic name for eight individual routines
!     that fill the complex(8) array with uniform random numbers. The Fortran
!     intrinsic routine random_number for real(8) variables is used for this in
!     the background. The argument z can be any rank from 0 to 7 (0 means scalar).
!-----
!
!  Change Log:
!  2009/Oct/27 --> Initial Release
!  2009/Dec/11 --> Changed default address function to LOC instead of %LOC
! 
!*************************************************************************************

#include "fintrf.h"

#ifndef mwPointer
#define mwPointer integer(4)
#endif

#ifndef mwSize
#define mwSize integer(4)
#endif

#ifndef mwIndex
#define mwIndex integer(4)
#endif

#ifdef SMALLMODEL
#define NOINTEGER8
#define NOCOMPLEX16
#define NOREAL16
#endif

#ifdef PERCENTLOC
#define loc %LOC
#endif

!\
! If your compiler doesn't support integer(8) data type, then uncomment this line
!/
!#define NOINTEGER8

!\
! If your compiler doesn't support complex(16) data type, then uncomment this line
!/
!#define NOCOMPLEX16

!\
! If your compiler doesn't support real(16) data type, then uncomment this line
!/
!#define NOREAL16

      module MatlabAPImx
      
      integer*4, parameter :: mxREAL = 0
      integer*4, parameter :: mxCOMPLEX = 1

      integer*4, parameter :: mxUNKNOWN_CLASS  =  0
      integer*4, parameter :: mxCELL_CLASS     =  1
      integer*4, parameter :: mxSTRUCT_CLASS   =  2
      integer*4, parameter :: mxLOGICAL_CLASS  =  3
      integer*4, parameter :: mxCHAR_CLASS     =  4
      integer*4, parameter :: mxDOUBLE_CLASS   =  6
      integer*4, parameter :: mxSINGLE_CLASS   =  7
      integer*4, parameter :: mxINT8_CLASS     =  8
      integer*4, parameter :: mxUINT8_CLASS    =  9
      integer*4, parameter :: mxINT16_CLASS    = 10
      integer*4, parameter :: mxUINT16_CLASS   = 11
      integer*4, parameter :: mxINT32_CLASS    = 12
      integer*4, parameter :: mxUINT32_CLASS   = 13
      integer*4, parameter :: mxINT64_CLASS    = 14
      integer*4, parameter :: mxUINT64_CLASS   = 15
      integer*4, parameter :: mxFUNCTION_CLASS = 16
      
      integer, parameter :: NameLengthMax_ = 63
      
!-----------------------------------------------------      
! Interface definitions for MATLAB API mx functions
!-----------------------------------------------------
      
      interface
!-----
      integer(4) function mxAddField(pm, fieldname)
      mwPointer, intent(in) :: pm
      character(len=*), intent(in) :: fieldname
      end function mxAddField
!-----

!mxArrayToString (C)

!-----

!mxAssert (C)

!-----

!mxAssertS (C)

!-----
      mwIndex function mxCalcSingleSubscript(pm, nsubs, subs)
      mwPointer, intent(in) :: pm
      mwSize, intent(in) :: nsubs
      mwIndex, intent(in) :: subs
      end function mxCalcSingleSubscript
!-----
      mwPointer function mxCalloc(n, elementsize)
      mwSize, intent(in) :: n, elementsize
      end function mxCalloc
!-----
      integer(4) function mxClassIDFromClassName(classname)
      character(len=*), intent(in) :: classname
      end function mxClassIDFromClassName
!-----
      subroutine mxCopyCharacterToPtr(y, px, n)  ! Ptr is to a C-style string
      character(len=*), intent(in) :: y
      mwPointer, intent(out) :: px
      mwSize, intent(in) :: n
      end subroutine mxCopyCharacterToPtr
!-----
!      subroutine mxCopyComplex32ToPtr(y, pr, pi, n)  ! New code below
!      complex(16), intent(in) :: y(n)
!      mwPointer, intent(out) :: pr, pi
!      mwSize, intent(in) :: n
!      end subroutine mxCopyComplex32ToPtr
!-----
      subroutine mxCopyComplex16ToPtr(y, pr, pi, n)
      complex(8), intent(in) :: y(n)
      mwPointer, intent(out) :: pr, pi
      mwSize, intent(in) :: n
      end subroutine mxCopyComplex16ToPtr
!-----
      subroutine mxCopyComplex8ToPtr(y, pr, pi, n)
      complex(4), intent(in) :: y(n)
      mwPointer, intent(out) :: pr, pi
      mwSize, intent(in) :: n
      end subroutine mxCopyComplex8ToPtr
!-----
      subroutine mxCopyInteger1ToPtr(y, px, n)
      integer(1), intent(in) :: y(n)
      mwPointer, intent(out) :: px
      mwSize, intent(in) :: n
      end subroutine mxCopyInteger1ToPtr
!-----
      subroutine mxCopyInteger2ToPtr(y, px, n)
      integer(2), intent(in) :: y(n)
      mwPointer, intent(out) :: px
      mwSize, intent(in) :: n
      end subroutine mxCopyInteger2ToPtr
!-----
      subroutine mxCopyInteger4ToPtr(y, px, n)
      integer(4), intent(in) :: y(n)
      mwPointer, intent(out) :: px
      mwSize, intent(in) :: n
      end subroutine mxCopyInteger4ToPtr
!-----
!      subroutine mxCopyInteger8ToPtr(y, px, n) ! New code below
!      integer(8), intent(in) :: y(n)
!      mwPointer, intent(out) :: px
!      mwSize, intent(in) :: n
!      end subroutine mxCopyInteger8ToPtr
!-----
      subroutine mxCopyPtrToCharacter(px, y, n) ! Ptr is to a C-style string
      mwPointer, intent(in) :: px
      character(len=*), intent(out) :: y
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToCharacter
!-----
!      subroutine mxCopyPtrToComplex32(pr, pi, y, n) ! New code below
!      mwPointer, intent(in) :: pr, pi
!      complex(16), intent(out) :: y(n)
!      mwSize, intent(in) :: n
!      end subroutine mxCopyPtrToComplex32
!-----
      subroutine mxCopyPtrToComplex16(pr, pi, y, n)
      mwPointer, intent(in) :: pr, pi
      complex(8), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToComplex16
!-----
      subroutine mxCopyPtrToComplex8(pr, pi, y, n)
      mwPointer, intent(in) :: pr, pi
      complex(4), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToComplex8
!-----
      subroutine mxCopyPtrToInteger1(px, y, n)
      mwPointer, intent(in) :: px
      integer(1), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToInteger1
!-----
      subroutine mxCopyPtrToInteger2(px, y, n)
      mwPointer, intent(in) :: px
      integer(2), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToInteger2
!-----
      subroutine mxCopyPtrToInteger4(px, y, n)
      mwPointer, intent(in) :: px
      integer(4), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToInteger4
!-----
!      subroutine mxCopyPtrToInteger8(px, y, n) ! New code below
!      mwPointer, intent(in) :: px
!      integer(8), intent(out) :: y(n)
!      mwSize, intent(in) :: n
!      end subroutine mxCopyPtrToInteger8
!-----
      subroutine mxCopyPtrToPtrArray(px, y, n)
      mwPointer, intent(in) :: px
      mwPointer, intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToPtrArray
!-----
      subroutine mxCopyPtrToReal4(px, y, n)
      mwPointer, intent(in) :: px
      real(4), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToReal4
!-----
      subroutine mxCopyPtrToReal8(px, y, n)
      mwPointer, intent(in) :: px
      real(8), intent(out) :: y(n)
      mwSize, intent(in) :: n
      end subroutine mxCopyPtrToReal8
!-----
!      subroutine mxCopyPtrToReal16(px, y, n) ! New code below
!      mwPointer, intent(in) :: px
!      real(16), intent(out) :: y(n)
!      mwSize, intent(in) :: n
!      end subroutine mxCopyPtrToReal16
!-----
      subroutine mxCopyReal4ToPtr(y, px, n)
      real(4), intent(in) :: y(n)
      mwPointer, intent(out) :: px
      mwSize, intent(in) :: n
      end subroutine mxCopyReal4ToPtr
!-----
      subroutine mxCopyReal8ToPtr(y, px, n)
      real(8), intent(in) :: y(n)
      mwPointer, intent(out) :: px
      mwSize, intent(in) :: n
      end subroutine mxCopyReal8ToPtr
!-----
!      subroutine mxCopyReal16ToPtr(y, px, n) ! New code below
!      real(16), intent(in) :: y(n)
!      mwPointer, intent(out) :: px
!      mwSize, intent(in) :: n
!      end subroutine mxCopyReal16ToPtr
!-----
      mwPointer function mxCreateCellArray(ndim, dims)
      mwSize, intent(in) :: ndim
      mwSize, intent(in) :: dims(ndim)
      end function mxCreateCellArray
!-----
      mwPointer function mxCreateCellMatrix(m, n)
      mwSize, intent(in) :: m, n
      end function mxCreateCellMatrix
!-----
      mwPointer function mxCreateCharArray(ndim, dims)
      mwSize, intent(in) :: ndim
      mwSize, intent(in) :: dims(ndim)
      end function mxCreateCharArray
!-----
      mwPointer function mxCreateCharMatrixFromStrings(m, str)
      mwSize, intent(in) :: m
      character(len=*), intent(in) :: str(m)
      end function mxCreateCharMatrixFromStrings
!-----
      mwPointer function mxCreateDoubleMatrix(m, n, ComplexFlag)
      mwSize, intent(in) :: m, n
      integer(4), intent(in) :: ComplexFlag
      end function mxCreateDoubleMatrix
!-----
      mwPointer function mxCreateDoubleScalar(number)
      real(8), intent(in) :: number
      end function mxCreateDoubleScalar
!-----
!      mwPointer function mxCreateLogicalArray( ndim, dims ) ! New code below
!      mwSize, intent(in) :: ndim
!      mwSize, intent(in) :: dims(ndim)
!      end function mxCreateLogicalArray
!-----
!      mwPointer function mxCreateLogicalMatrix( m, n ) ! New code below
!      mwSize, intent(in) :: m, n
!      end function mxCreateLogicalMatrix
!-----
!      mwPointer function mxCreateLogicalScalar( number ) ! New code below
!      logical, intent(in) :: number
!      end function mxCreateLogicalScalar
!-----
      mwPointer function mxCreateNumericArray(ndim, dims,               &
     &                                        classid, ComplexFlag)
      mwSize, intent(in) :: ndim
      mwSize, intent(in) :: dims(ndim)
      integer(4), intent(in) :: classid, ComplexFlag
      end function mxCreateNumericArray
!-----
      mwPointer function mxCreateNumericMatrix(m, n,                    &
     &                                         classid, ComplexFlag)
      mwSize, intent(in) :: m, n
      integer(4), intent(in) :: classid, ComplexFlag
      end function mxCreateNumericMatrix
!-----
      mwPointer function mxCreateSparse(m, n, nzmax, ComplexFlag)
      mwSize, intent(in) :: m, n, nzmax
      integer(4), intent(in) :: ComplexFlag
      end function mxCreateSparse
!-----
!      mwPointer function mxCreateSparseLogicalMatrix(m, n, nzmax) ! See MatlabAPImex, MatlabAPIeng
!      mwSize, intent(in) :: m, n, nzmax                           ! mexCreateSparseLogicalMatrix
!      end function mxCreateSparseLogicalMatrix                    ! engCreateSparseLogicalMatrix
!-----
      mwPointer function mxCreateString(str)
      character(len=*), intent(in) :: str
      end function mxCreateString
!-----
      mwPointer function mxCreateStructArray(ndim, dims,                &
     &                                       nfields, fieldnames)
      mwSize, intent(in) :: ndim
      mwSize, intent(in) :: dims(ndim)
      integer(4), intent(in) :: nfields
      character(len=*), intent(in) :: fieldnames(nfields)
      end function mxCreateStructArray
!-----
      mwPointer function mxCreateStructMatrix(m, n,                     &
     &                                        nfields, fieldnames)
      mwSize, intent(in) :: m, n
      integer(4), intent(in) :: nfields
      character(len=*), intent(in) :: fieldnames(nfields)
      end function mxCreateStructMatrix
!-----
      subroutine mxDestroyArray(pm)
      mwPointer, intent(in) :: pm
      end subroutine mxDestroyArray
!-----
      mwPointer function mxDuplicateArray(pm)
      mwPointer, intent(in) :: pm
      end function mxDuplicateArray
!-----
      subroutine mxFree(ptr)
      mwPointer, intent(in) :: ptr
      end subroutine mxFree
!-----
      mwPointer function mxGetCell(pm, cellindex)
      mwPointer, intent(in) :: pm
      mwIndex, intent(in) :: cellindex
      end function mxGetCell
!-----
!      mwPointer function mxGetChars(pm) ! New code below
!      mwPointer, intent(in) :: pm
!      end function mxGetChars
!-----
      integer(4) function mxGetClassID(pm)
      mwPointer, intent(in) :: pm
      end function mxGetClassID
!-----
      function mxGetClassName(pm)
      integer, parameter :: NameLengthMax_ = 63
      character(len=NameLengthMax_) :: mxGetClassName ! Doc is wrong
      mwPointer, intent(in) :: pm
      end function mxGetClassName
!-----
      mwPointer function mxGetData(pm)
      mwPointer, intent(in) :: pm
      end function mxGetData
!-----
      mwPointer function mxGetDimensions(pm)
      mwPointer, intent(in) :: pm
      end function mxGetDimensions
!-----
      mwSize function mxGetElementSize(pm) ! Doc is wrong
      mwPointer, intent(in) :: pm
      end function mxGetElementSize
!-----
      real(8) function mxGetEps
      end function mxGetEps
!-----
      mwPointer function mxGetField(pm, fieldindex, fieldname)
      mwPointer, intent(in) :: pm
      mwIndex, intent(in) :: fieldindex
      character(len=*), intent(in) :: fieldname
      end function mxGetField
!-----
      mwPointer function mxGetFieldByNumber(pm, fieldindex,             &
     &                                      fieldnumber)
      mwPointer, intent(in) :: pm
      mwIndex, intent(in) :: fieldindex
      integer(4), intent(in) :: fieldnumber
      end function mxGetFieldByNumber
!-----
      function mxGetFieldNameByNumber(pm, fieldnumber) ! Doc is wrong
      integer, parameter :: NameLengthMax_ = 63
      character(len=NameLengthMax_) :: mxGetFieldNameByNumber
      mwPointer, intent(in) :: pm
      integer(4), intent(in) :: fieldnumber
      end function mxGetFieldNameByNumber
!-----
      integer(4) function mxGetFieldNumber(pm, fieldname)
      mwPointer, intent(in) :: pm
      character(len=*), intent(in) :: fieldname
      end function mxGetFieldNumber
!-----
      mwPointer function mxGetImagData(pm)
      mwPointer, intent(in) :: pm
      end function mxGetImagData
!-----
      real(8) function mxGetInf
      end function mxGetInf
!-----
      mwPointer function mxGetIr(pm)
      mwPointer, intent(in) :: pm
      end function mxGetIr
!-----
      mwPointer function mxGetJc(pm)
      mwPointer, intent(in) :: pm
      end function mxGetJc
!-----
!      mwPointer function mxGetLogicals(pm) ! New code below
!      mwPointer, intent(in) :: pm
!      end function mxGetLogicals
!-----
      mwSize function mxGetM( mx )
      mwPointer, intent(in) :: mx
      end function mxGetM
!-----
      mwSize function mxGetN( mx )
      mwPointer, intent(in) :: mx
      end function mxGetN
!-----
      real(8) function mxGetNaN
      end function mxGetNaN
!-----
      mwSize function mxGetNumberOfDimensions( mx )
      mwPointer, intent(in) :: mx
      end function mxGetNumberOfDimensions
!-----
      mwSize function mxGetNumberOfElements( mx )
      mwPointer, intent(in) :: mx
      end function mxGetNumberOfElements
!-----
      integer(4) function mxGetNumberOfFields(pm)
      mwPointer, intent(in) :: pm
      end function mxGetNumberOfFields
!-----
      mwSize function mxGetNzmax(pm)
      mwPointer, intent(in) :: pm
      end function mxGetNzmax
!-----
      mwPointer function mxGetPi( mx )
      mwPointer, intent(in) :: mx
      end function mxGetPi
!-----
      mwPointer function mxGetPr( mx )
      mwPointer, intent(in) :: mx
      end function mxGetPr
!-----
      mwPointer function mxGetProperty(pa, objectindex, propname)
      mwPointer, intent(in) :: pa
      mwIndex, intent(in) :: objectindex
      character(len=*), intent(in) :: propname
      end function mxGetProperty
!-----
      real(8) function mxGetScalar(pm)
      mwPointer, intent(in) :: pm
      end function mxGetScalar
!-----
      integer(4) function mxGetString(pm, str, strlen)
      mwPointer, intent(in) :: pm
      character(len=*), intent(in) :: str
      mwSize, intent(in) :: strlen
      end function mxGetString
!-----
      integer(4) function mxIsCell(pm)
      mwPointer, intent(in) :: pm
      end function mxIsCell
!-----
      integer(4) function mxIsChar(pm)
      mwPointer, intent(in) :: pm
      end function mxIsChar
!-----
      integer(4) function mxIsClass(pm, classname)
      mwPointer, intent(in) :: pm
      character(len=*), intent(in) :: classname
      end function mxIsClass
!-----
      integer(4) function mxIsComplex( mx )
      mwPointer, intent(in) :: mx
      end function mxIsComplex
!-----
      integer(4) function mxIsDouble( mx )
      mwPointer, intent(in) :: mx
      end function mxIsDouble
!-----
      integer(4) function mxIsEmpty(pm)
      mwPointer, intent(in) :: pm
      end function mxIsEmpty
!-----
      integer(4) function mxIsFinite(number)
      real(8), intent(in) :: number
      end function mxIsFinite
!-----
      integer(4) function mxIsFromGlobalWS(pm)
      mwPointer, intent(in) :: pm
      end function mxIsFromGlobalWS
!-----
      integer(4) function mxIsInf(number)
      real(8), intent(in) :: number
      end function mxIsInf
!-----
      integer(4) function mxIsInt16(pm)
      mwPointer, intent(in) :: pm
      end function mxIsInt16
!-----
      integer(4) function mxIsInt32(pm)
      mwPointer, intent(in) :: pm
      end function mxIsInt32
!-----
      integer(4) function mxIsInt64(pm)
      mwPointer, intent(in) :: pm
      end function mxIsInt64
!-----
      integer(4) function mxIsInt8(pm)
      mwPointer, intent(in) :: pm
      end function mxIsInt8
!-----
      integer(4) function mxIsLogical(pm)
      mwPointer, intent(in) :: pm
      end function mxIsLogical
!-----
!      integer(4) function mxIsLogicalScalar(pm) ! New code below
!      mwPointer, intent(in) :: pm
!      end function mxIsLogicalScalar
!-----
!      integer(4) function mxIsLogicalScalarTrue(pm) ! New code below
!      mwPointer, intent(in) :: pm
!      end function mxIsLogicalScalarTrue
!-----
      integer(4) function mxIsNaN(number)
      real(8), intent(in) :: number
      end function mxIsNaN
!-----
      integer(4) function mxIsNumeric(pm)
      mwPointer, intent(in) :: pm
      end function mxIsNumeric
!-----
      integer(4) function mxIsSingle(pm)
      mwPointer, intent(in) :: pm
      end function mxIsSingle
!-----
      integer(4) function mxIsSparse(pm)
      mwPointer, intent(in) :: pm
      end function mxIsSparse
!-----
      integer(4) function mxIsStruct(pm)
      mwPointer, intent(in) :: pm
      end function mxIsStruct
!-----
      integer(4) function mxIsUint16(pm)
      mwPointer, intent(in) :: pm
      end function mxIsUint16
!-----
      integer(4) function mxIsUint32(pm)
      mwPointer, intent(in) :: pm
      end function mxIsUint32
!-----
      integer(4) function mxIsUint64(pm)
      mwPointer, intent(in) :: pm
      end function mxIsUint64
!-----
      integer(4) function mxIsUint8(pm)
      mwPointer, intent(in) :: pm
      end function mxIsUint8
!-----
      mwPointer function mxMalloc(n)
      mwSize, intent(in) :: n
      end function mxMalloc
!-----
      mwPointer function mxRealloc(ptr, elementsize)
      mwPointer, intent(in) :: ptr
      mwSize, intent(in) :: elementsize
      end function mxRealloc
!-----
      subroutine mxRemoveField(pm, fieldnumber)
      mwPointer, intent(in) :: pm
      integer(4), intent(in) :: fieldnumber
      end subroutine mxRemoveField
!-----
      subroutine mxSetCell(pm, cellindex, cellvalue)
      mwPointer, intent(in) :: pm, cellvalue
      mwIndex, intent(in) :: cellindex
      end subroutine mxSetCell
!-----

!mxSetClassName (C)

!-----
      subroutine mxSetData(pm, pr)
      mwPointer, intent(in) :: pm, pr
      end subroutine mxSetData
!-----
      integer(4) function mxSetDimensions(pm, dims, ndim)
      mwPointer, intent(in) :: pm
      mwSize, intent(in) :: ndim
      mwSize, intent(in) :: dims(ndim)
      end function mxSetDimensions
!-----
      subroutine mxSetField(pm, fieldindex, fieldname, pvalue)
      mwPointer, intent(in) :: pm, pvalue
      mwIndex, intent(in) :: fieldindex
      character(len=*), intent(in) :: fieldname
      end subroutine mxSetField
!-----
      subroutine mxSetFieldByNumber(pm, fieldindex,                     &
     &                              fieldnumber, pvalue)
      mwPointer, intent(in) :: pm, pvalue
      mwIndex, intent(in) :: fieldindex
      integer(4), intent(in) :: fieldnumber
      end subroutine mxSetFieldByNumber
!-----
      subroutine mxSetImagData(pm, pi)
      mwPointer, intent(in) :: pm, pi
      end subroutine mxSetImagData
!-----
      subroutine mxSetIr(pm, ir)
      mwPointer, intent(in) :: pm, ir
      end subroutine mxSetIr
!-----
      subroutine mxSetJc(pm, jc)
      mwPointer, intent(in) :: pm, jc
      end subroutine mxSetJc
!-----
      subroutine mxSetM(pm, m)
      mwPointer, intent(in) :: pm
      mwSize, intent(in) :: m
      end subroutine mxSetM
!-----
      subroutine mxSetN(pm, n)
      mwPointer, intent(in) :: pm
      mwSize, intent(in) :: n
      end subroutine mxSetN
!-----
      subroutine mxSetNzmax(pm, nzmax)
      mwPointer, intent(in) :: pm
      mwSize, intent(in) :: nzmax
      end subroutine mxSetNzmax
!-----
      subroutine mxSetPi(pm, pi)
      mwPointer, intent(in) :: pm, pi
      end subroutine mxSetPi
!-----
      subroutine mxSetPr(pm, pr)
      mwPointer, intent(in) :: pm, pr
      end subroutine mxSetPr
!-----
      end interface
      
!-----------------------------------------------------      
! Interface definitions for Fortran Pointer functions
!-----------------------------------------------------
      
      interface fpGetPr
          module procedure fpGetPr2Double
      end interface
      
      interface fpGetPr0
          module procedure fpGetPr0Double
      end interface
      
      interface fpGetPr1
          module procedure fpGetPr1Double
      end interface
      
      interface fpGetPr2
          module procedure fpGetPr2Double
      end interface
      
      interface fpGetPr3
          module procedure fpGetPr3Double
      end interface
      
      interface fpGetPr4
          module procedure fpGetPr4Double
      end interface
      
      interface fpGetPr5
          module procedure fpGetPr5Double
      end interface
      
      interface fpGetPr6
          module procedure fpGetPr6Double
      end interface
      
      interface fpGetPr7
          module procedure fpGetPr7Double
      end interface
      
      interface fpGetPi
          module procedure fpGetPi2Double
      end interface
      
      interface fpGetPi0
          module procedure fpGetPi0Double
      end interface
      
      interface fpGetPi1
          module procedure fpGetPi1Double
      end interface
      
      interface fpGetPi2
          module procedure fpGetPi2Double
      end interface
      
      interface fpGetPi3
          module procedure fpGetPi3Double
      end interface
      
      interface fpGetPi4
          module procedure fpGetPi4Double
      end interface
      
      interface fpGetPi5
          module procedure fpGetPi5Double
      end interface
      
      interface fpGetPi6
          module procedure fpGetPi6Double
      end interface
      
      interface fpGetPi7
          module procedure fpGetPi7Double
      end interface
      
      interface fpGetPrCopy
          module procedure fpGetPrCopy2Double
      end interface
      
      interface fpGetPrCopy0
          module procedure fpGetPrCopy0Double
      end interface
      
      interface fpGetPrCopy1
          module procedure fpGetPrCopy1Double
      end interface
      
      interface fpGetPrCopy2
          module procedure fpGetPrCopy2Double
      end interface
      
      interface fpGetPrCopy3
          module procedure fpGetPrCopy3Double
      end interface
      
      interface fpGetPrCopy4
          module procedure fpGetPrCopy4Double
      end interface
      
      interface fpGetPrCopy5
          module procedure fpGetPrCopy5Double
      end interface
      
      interface fpGetPrCopy6
          module procedure fpGetPrCopy6Double
      end interface
      
      interface fpGetPrCopy7
          module procedure fpGetPrCopy7Double
      end interface
      
      interface fpGetPiCopy
          module procedure fpGetPiCopy2Double
      end interface
      
      interface fpGetPiCopy0
          module procedure fpGetPiCopy0Double
      end interface
      
      interface fpGetPiCopy1
          module procedure fpGetPiCopy1Double
      end interface
      
      interface fpGetPiCopy2
          module procedure fpGetPiCopy2Double
      end interface
      
      interface fpGetPiCopy3
          module procedure fpGetPiCopy3Double
      end interface
      
      interface fpGetPiCopy4
          module procedure fpGetPiCopy4Double
      end interface
      
      interface fpGetPiCopy5
          module procedure fpGetPiCopy5Double
      end interface
      
      interface fpGetPiCopy6
          module procedure fpGetPiCopy6Double
      end interface
      
      interface fpGetPiCopy7
          module procedure fpGetPiCopy7Double
      end interface

      interface fpGetPzCopy
          module procedure fpGetPzCopy2Double
      end interface
      
      interface fpGetPzCopy0
          module procedure fpGetPzCopy0Double
      end interface
      
      interface fpGetPzCopy1
          module procedure fpGetPzCopy1Double
      end interface
      
      interface fpGetPzCopy2
          module procedure fpGetPzCopy2Double
      end interface
      
      interface fpGetPzCopy3
          module procedure fpGetPzCopy3Double
      end interface
      
      interface fpGetPzCopy4
          module procedure fpGetPzCopy4Double
      end interface
      
      interface fpGetPzCopy5
          module procedure fpGetPzCopy5Double
      end interface
      
      interface fpGetPzCopy6
          module procedure fpGetPzCopy6Double
      end interface
      
      interface fpGetPzCopy7
          module procedure fpGetPzCopy7Double
      end interface
      
      interface fpReal
          module procedure fpReal1Double, fpReal2Double, fpReal3Double, &
     &                     fpReal4Double, fpReal5Double, fpReal6Double, &
     &                     fpReal7Double
      end interface
      
      interface fpReal1
          module procedure fpReal1Double
      end interface
      
      interface fpReal2
          module procedure fpReal2Double
      end interface
      
      interface fpReal3
          module procedure fpReal3Double
      end interface
      
      interface fpReal4
          module procedure fpReal4Double
      end interface
      
      interface fpReal5
          module procedure fpReal5Double
      end interface
      
      interface fpReal6
          module procedure fpReal6Double
      end interface
      
      interface fpReal7
          module procedure fpReal7Double
      end interface
      
      interface fpImag
          module procedure fpImag1Double, fpImag2Double, fpImag3Double, &
     &                     fpImag4Double, fpImag5Double, fpImag6Double, &
     &                     fpImag7Double
      end interface
      
      interface fpImag1
          module procedure fpImag1Double
      end interface
      
      interface fpImag2
          module procedure fpImag2Double
      end interface
      
      interface fpImag3
          module procedure fpImag3Double
      end interface
      
      interface fpImag4
          module procedure fpImag4Double
      end interface
      
      interface fpImag5
          module procedure fpImag5Double
      end interface
      
      interface fpImag6
          module procedure fpImag6Double
      end interface
      
      interface fpImag7
          module procedure fpImag7Double
      end interface
      
      interface fpAllocate
          module procedure fpAllocate1Double,  fpAllocate2Double,       &
     &                     fpAllocate3Double,  fpAllocate4Double,       &
     &                     fpAllocate5Double,  fpAllocate6Double,       &
     &                     fpAllocate7Double,  fpAllocate0Double
     
      end interface
      
      interface fpAllocate0
          module procedure fpAllocate0Double
      end interface
      
      interface fpAllocate1
          module procedure fpAllocate1Double
      end interface
      
      interface fpAllocate2
          module procedure fpAllocate2Double
      end interface
      
      interface fpAllocate3
          module procedure fpAllocate3Double
      end interface
      
      interface fpAllocate4
          module procedure fpAllocate4Double
      end interface
      
      interface fpAllocate5
          module procedure fpAllocate5Double
      end interface
      
      interface fpAllocate6
          module procedure fpAllocate6Double
      end interface
      
      interface fpAllocate7
          module procedure fpAllocate7Double
      end interface
      
      interface fpAllocateZ
          module procedure fpAllocateZ1Double, fpAllocateZ2Double,      &
     &                     fpAllocateZ3Double, fpAllocateZ4Double,      &
     &                     fpAllocateZ5Double, fpAllocateZ6Double,      &
     &                     fpAllocateZ7Double, fpAllocateZ0Double
      end interface
      
      interface fpAllocateZ0
          module procedure fpAllocateZ0Double
      end interface
      
      interface fpAllocateZ1
          module procedure fpAllocateZ1Double
      end interface
      
      interface fpAllocateZ2
          module procedure fpAllocateZ2Double
      end interface
      
      interface fpAllocateZ3
          module procedure fpAllocateZ3Double
      end interface
      
      interface fpAllocateZ4
          module procedure fpAllocateZ4Double
      end interface
      
      interface fpAllocateZ5
          module procedure fpAllocateZ5Double
      end interface
      
      interface fpAllocateZ6
          module procedure fpAllocateZ6Double
      end interface
      
      interface fpAllocateZ7
          module procedure fpAllocateZ7Double
      end interface
      
      interface fpDeallocate
          module procedure fpDeallocate1Double, fpDeallocate2Double,    &
     &                     fpDeallocate3Double, fpDeallocate4Double,    &
     &                     fpDeallocate5Double, fpDeallocate6Double,    &
     &                     fpDeallocate7Double, fzDeallocate1Double,    &
     &                     fzDeallocate2Double, fzDeallocate3Double,    &
     &                     fzDeallocate4Double, fzDeallocate5Double,    &
     &                     fzDeallocate6Double, fzDeallocate7Double,    &
     &                     fpDeallocate0Double, fzDeallocate0Double,    &
     &                     fpDeallocate1Character
      end interface
      
      interface fpDeallocate0
          module procedure fpDeallocate0Double
      end interface
      
      interface fpDeallocate1
          module procedure fpDeallocate1Double
      end interface
      
      interface fpDeallocate2
          module procedure fpDeallocate2Double
      end interface
      
      interface fpDeallocate3
          module procedure fpDeallocate3Double
      end interface
      
      interface fpDeallocate4
          module procedure fpDeallocate4Double
      end interface
      
      interface fpDeallocate5
          module procedure fpDeallocate5Double
      end interface
      
      interface fpDeallocate6
          module procedure fpDeallocate6Double
      end interface
      
      interface fpDeallocate7
          module procedure fpDeallocate7Double
      end interface
      
      interface fzDeallocate
          module procedure fzDeallocate2Double
      end interface
      
      interface fzDeallocate0
          module procedure fzDeallocate0Double
      end interface
      
      interface fzDeallocate1
          module procedure fzDeallocate1Double
      end interface
      
      interface fzDeallocate2
          module procedure fzDeallocate2Double
      end interface
      
      interface fzDeallocate3
          module procedure fzDeallocate3Double
      end interface
      
      interface fzDeallocate4
          module procedure fzDeallocate4Double
      end interface
      
      interface fzDeallocate5
          module procedure fzDeallocate5Double
      end interface
      
      interface fzDeallocate6
          module procedure fzDeallocate6Double
      end interface
      
      interface fzDeallocate7
          module procedure fzDeallocate7Double
      end interface

      interface fpStride
          module procedure fpStride1Double, fpStride2Double,            &
     &                     fpStride3Double, fpStride4Double,            &
     &                     fpStride5Double, fpStride6Double,            &
     &                     fpStride7Double, fzStride1Double,            &
     &                     fzStride2Double, fzStride3Double,            &
     &                     fzStride4Double, fzStride5Double,            &
     &                     fzStride6Double, fzStride7Double
      end interface
      
      interface fpReshape
          module procedure fpReshape11Double, fpReshape12Double,        &
     &                     fpReshape13Double, fpReshape14Double,        &
     &                     fpReshape15Double, fpReshape16Double,        &
     &                     fpReshape17Double,                           &
     &                     fpReshape21Double, fpReshape22Double,        &
     &                     fpReshape23Double, fpReshape24Double,        &
     &                     fpReshape25Double, fpReshape26Double,        &
     &                     fpReshape27Double,                           &
     &                     fpReshape31Double, fpReshape32Double,        &
     &                     fpReshape33Double, fpReshape34Double,        &
     &                     fpReshape35Double, fpReshape36Double,        &
     &                     fpReshape37Double,                           &
     &                     fpReshape41Double, fpReshape42Double,        &
     &                     fpReshape43Double, fpReshape44Double,        &
     &                     fpReshape45Double, fpReshape46Double,        &
     &                     fpReshape47Double,                           &
     &                     fpReshape51Double, fpReshape52Double,        &
     &                     fpReshape53Double, fpReshape54Double,        &
     &                     fpReshape55Double, fpReshape56Double,        &
     &                     fpReshape57Double,                           &
     &                     fpReshape61Double, fpReshape62Double,        &
     &                     fpReshape63Double, fpReshape64Double,        &
     &                     fpReshape65Double, fpReshape66Double,        &
     &                     fpReshape67Double,                           &
     &                     fpReshape71Double, fpReshape72Double,        &
     &                     fpReshape73Double, fpReshape74Double,        &
     &                     fpReshape75Double, fpReshape76Double,        &
     &                     fpReshape77Double,                           &
     &                     fzReshape11Double, fzReshape12Double,        &
     &                     fzReshape13Double, fzReshape14Double,        &
     &                     fzReshape15Double, fzReshape16Double,        &
     &                     fzReshape17Double,                           &
     &                     fzReshape21Double, fzReshape22Double,        &
     &                     fzReshape23Double, fzReshape24Double,        &
     &                     fzReshape25Double, fzReshape26Double,        &
     &                     fzReshape27Double,                           &
     &                     fzReshape31Double, fzReshape32Double,        &
     &                     fzReshape33Double, fzReshape34Double,        &
     &                     fzReshape35Double, fzReshape36Double,        &
     &                     fzReshape37Double,                           &
     &                     fzReshape41Double, fzReshape42Double,        &
     &                     fzReshape43Double, fzReshape44Double,        &
     &                     fzReshape45Double, fzReshape46Double,        &
     &                     fzReshape47Double,                           &
     &                     fzReshape51Double, fzReshape52Double,        &
     &                     fzReshape53Double, fzReshape54Double,        &
     &                     fzReshape55Double, fzReshape56Double,        &
     &                     fzReshape57Double,                           &
     &                     fzReshape61Double, fzReshape62Double,        &
     &                     fzReshape63Double, fzReshape64Double,        &
     &                     fzReshape65Double, fzReshape66Double,        &
     &                     fzReshape67Double,                           &
     &                     fzReshape71Double, fzReshape72Double,        &
     &                     fzReshape73Double, fzReshape74Double,        &
     &                     fzReshape75Double, fzReshape76Double,        &
     &                     fzReshape77Double
      end interface
      
      interface fpReshape11
          module procedure fpReshape11Double
      end interface
      
      interface fpReshape12
          module procedure fpReshape12Double
      end interface
      
      interface fpReshape13
          module procedure fpReshape13Double
      end interface
      
      interface fpReshape14
          module procedure fpReshape14Double
      end interface
      
      interface fpReshape15
          module procedure fpReshape15Double
      end interface
      
      interface fpReshape16
          module procedure fpReshape16Double
      end interface
      
      interface fpReshape17
          module procedure fpReshape17Double
      end interface
      
      interface fpReshape21
          module procedure fpReshape21Double
      end interface
      
      interface fpReshape22
          module procedure fpReshape22Double
      end interface
      
      interface fpReshape23
          module procedure fpReshape23Double
      end interface
      
      interface fpReshape24
          module procedure fpReshape24Double
      end interface
      
      interface fpReshape25
          module procedure fpReshape25Double
      end interface
      
      interface fpReshape26
          module procedure fpReshape26Double
      end interface
      
      interface fpReshape27
          module procedure fpReshape27Double
      end interface

      interface fpReshape31
          module procedure fpReshape31Double
      end interface
      
      interface fpReshape32
          module procedure fpReshape32Double
      end interface
      
      interface fpReshape33
          module procedure fpReshape33Double
      end interface
      
      interface fpReshape34
          module procedure fpReshape34Double
      end interface
      
      interface fpReshape35
          module procedure fpReshape35Double
      end interface
      
      interface fpReshape36
          module procedure fpReshape36Double
      end interface
      
      interface fpReshape37
          module procedure fpReshape37Double
      end interface
      
      interface fpReshape41
          module procedure fpReshape41Double
      end interface
      
      interface fpReshape42
          module procedure fpReshape42Double
      end interface
      
      interface fpReshape43
          module procedure fpReshape43Double
      end interface
      
      interface fpReshape44
          module procedure fpReshape44Double
      end interface
      
      interface fpReshape45
          module procedure fpReshape45Double
      end interface
      
      interface fpReshape46
          module procedure fpReshape46Double
      end interface
      
      interface fpReshape47
          module procedure fpReshape47Double
      end interface
      
      interface fpReshape51
          module procedure fpReshape51Double
      end interface
      
      interface fpReshape52
          module procedure fpReshape52Double
      end interface
      
      interface fpReshape53
          module procedure fpReshape53Double
      end interface
      
      interface fpReshape54
          module procedure fpReshape54Double
      end interface
      
      interface fpReshape55
          module procedure fpReshape55Double
      end interface
      
      interface fpReshape56
          module procedure fpReshape56Double
      end interface
      
      interface fpReshape57
          module procedure fpReshape57Double
      end interface
      
      interface fpReshape61
          module procedure fpReshape61Double
      end interface
      
      interface fpReshape62
          module procedure fpReshape62Double
      end interface
      
      interface fpReshape63
          module procedure fpReshape63Double
      end interface
      
      interface fpReshape64
          module procedure fpReshape64Double
      end interface
      
      interface fpReshape65
          module procedure fpReshape65Double
      end interface
      
      interface fpReshape66
          module procedure fpReshape66Double
      end interface
      
      interface fpReshape67
          module procedure fpReshape67Double
      end interface
      
      interface fpReshape71
          module procedure fpReshape71Double
      end interface
      
      interface fpReshape72
          module procedure fpReshape72Double
      end interface
      
      interface fpReshape73
          module procedure fpReshape73Double
      end interface
      
      interface fpReshape74
          module procedure fpReshape74Double
      end interface
      
      interface fpReshape75
          module procedure fpReshape75Double
      end interface
      
      interface fpReshape76
          module procedure fpReshape76Double
      end interface
      
      interface fpReshape77
          module procedure fpReshape77Double
      end interface
      
      interface fzReshape11
          module procedure fzReshape11Double
      end interface
      
      interface fzReshape12
          module procedure fzReshape12Double
      end interface
      
      interface fzReshape13
          module procedure fzReshape13Double
      end interface
      
      interface fzReshape14
          module procedure fzReshape14Double
      end interface
      
      interface fzReshape15
          module procedure fzReshape15Double
      end interface
      
      interface fzReshape16
          module procedure fzReshape16Double
      end interface
      
      interface fzReshape17
          module procedure fzReshape17Double
      end interface
      
      interface fzReshape21
          module procedure fzReshape21Double
      end interface
      
      interface fzReshape22
          module procedure fzReshape22Double
      end interface
      
      interface fzReshape23
          module procedure fzReshape23Double
      end interface
      
      interface fzReshape24
          module procedure fzReshape24Double
      end interface
      
      interface fzReshape25
          module procedure fzReshape25Double
      end interface
      
      interface fzReshape26
          module procedure fzReshape26Double
      end interface
      
      interface fzReshape27
          module procedure fzReshape27Double
      end interface
      
      interface fzReshape31
          module procedure fzReshape31Double
      end interface
      
      interface fzReshape32
          module procedure fzReshape32Double
      end interface
      
      interface fzReshape33
          module procedure fzReshape33Double
      end interface
      
      interface fzReshape34
          module procedure fzReshape34Double
      end interface
      
      interface fzReshape35
          module procedure fzReshape35Double
      end interface
      
      interface fzReshape36
          module procedure fzReshape36Double
      end interface
      
      interface fzReshape37
          module procedure fzReshape37Double
      end interface
      
      interface fzReshape41
          module procedure fzReshape41Double
      end interface
      
      interface fzReshape42
          module procedure fzReshape42Double
      end interface
      
      interface fzReshape43
          module procedure fzReshape43Double
      end interface
      
      interface fzReshape44
          module procedure fzReshape44Double
      end interface
      
      interface fzReshape45
          module procedure fzReshape45Double
      end interface
      
      interface fzReshape46
          module procedure fzReshape46Double
      end interface
      
      interface fzReshape47
          module procedure fzReshape47Double
      end interface
      
      interface fzReshape51
          module procedure fzReshape51Double
      end interface
      
      interface fzReshape52
          module procedure fzReshape52Double
      end interface
      
      interface fzReshape53
          module procedure fzReshape53Double
      end interface
      
      interface fzReshape54
          module procedure fzReshape54Double
      end interface
      
      interface fzReshape55
          module procedure fzReshape55Double
      end interface
      
      interface fzReshape56
          module procedure fzReshape56Double
      end interface
      
      interface fzReshape57
          module procedure fzReshape57Double
      end interface
      
      interface fzReshape61
          module procedure fzReshape61Double
      end interface
      
      interface fzReshape62
          module procedure fzReshape62Double
      end interface
      
      interface fzReshape63
          module procedure fzReshape63Double
      end interface
      
      interface fzReshape64
          module procedure fzReshape64Double
      end interface
      
      interface fzReshape65
          module procedure fzReshape65Double
      end interface
      
      interface fzReshape66
          module procedure fzReshape66Double
      end interface
      
      interface fzReshape67
          module procedure fzReshape67Double
      end interface
      
      interface fzReshape71
          module procedure fzReshape71Double
      end interface
      
      interface fzReshape72
          module procedure fzReshape72Double
      end interface
      
      interface fzReshape73
          module procedure fzReshape73Double
      end interface
      
      interface fzReshape74
          module procedure fzReshape74Double
      end interface
      
      interface fzReshape75
          module procedure fzReshape75Double
      end interface
      
      interface fzReshape76
          module procedure fzReshape76Double
      end interface
      
      interface fzReshape77
          module procedure fzReshape77Double
      end interface
      
      interface mxArray
          module procedure mxArray1double, mxArray2double,              &
     &     mxArray3double, mxArray4double, mxArray5double,              &
     &     mxArray6double, mxArray7double, mzArray1double,              &
     &     mzArray2double, mzArray3double, mzArray4double,              &
     &     mzArray5double, mzArray6double, mzArray7double,              &
     &     mxArray0double, mzArray0double
      end interface
      
      interface mxArrayHeader
          module procedure mxArrayHeader1double, mxArrayHeader2double,  &
     &                     mxArrayHeader3double, mxArrayHeader4double,  &
     &                     mxArrayHeader5double, mxArrayHeader6double,  &
     &                     mxArrayHeader7double, mxArrayHeader0double
      end interface
      
      interface random_number
          module procedure random_number1double, random_number2double,  &
     &                     random_number3double, random_number4double,  &
     &                     random_number5double, random_number6double,  &
     &                     random_number7double, random_number0double
      end interface

!-----------------------------------------------------      

      contains
      
!-----------------------------------------------------      
! Specific Fortan Replacement functions
!-----------------------------------------------------

      subroutine mxCopyCharacterToChars( y, px, n )
      implicit none
!-ARG
      character(len=*), intent(in) :: y  ! Input, character string
      mwPointer, intent(in) :: px        ! Output, result of a mxGetData call
      mwSize, intent(in) :: n            ! Input, number of characters to copy
!-LOC
      mwSize ly           ! Local, length of y
!-----
      ly = len(y)         ! Length of y. Used later to ensure we don't overflow
!\
! Put the value of px on the stack using the %VAL construct. This is the address
! of the character data from the mxArray variable. We want to put this address on
! the stack so the mxCopyC1toI2 routine can use it as an integer*2 array. Also,
! ensure we don't overrun the length of the array by only copying the minimum of
! len(y) and the desired number of copied characters n. MATLAB stores characters
! as 2-byte elements, whereas Fortran stores characters as 1-byte elements.
!/
      call mxCopyC1toI2( y, %VAL(px), min(n,ly) )
      return
      end subroutine

!-------------------------------------------------------------------------------

      subroutine mxCopyCharsToCharacter( px, y, n )
      implicit none
!-ARG
      mwPointer, intent(in) :: px         ! Input, result of a mxGetData call
      character(len=*), intent(out) :: y  ! Output, character string
      mwSize, intent(in) :: n             ! Input, number of characters to copy
!-LOC
      mwSize ly         ! Local, length of y
!-----
      ly = len(y)       ! Length of y. Used later to ensure we don't overflow
!\
! Put the value of px on the stack using the %VAL construct. This is the address
! of the character data from the mxArray variable. We want to put this address on
! the stack so the mxCopyI2toC1 routine can use it as an integer*2 array. Also,
! ensure we don't overrun the length of the array by only copying the minimum of
! len(y) and the desired number of copied characters n. MATLAB stores characters
! as 2-byte elements, whereas Fortran stores characters as 1-byte elements.
!/
      call mxCopyI2toC1( %VAL(px), y, min(n,ly) )
!\
! If number of copied characters is less than len(y), fill trailing with blanks
!/
      if( n < ly ) y(n+1:) = ' '
      return
      end subroutine

!------------------------------------------------------------------------------

      mwPointer function mxGetChars(pm)
      implicit none
!-ARG
      mwPointer, intent(in) :: pm
!-----
      mxGetChars = mxGetData(pm)
      end function mxGetChars

!------------------------------------------------------------------------------
!
! The default logical routines supplied are:
!
!     subroutine mxCopyPtrToLogical( Ptr, fortran, n )
!       - Copies MATLAB mxArray logical data to fortran logical variable
!
!     subroutine mxCopyLogicalToPtr( fortran, Ptr, n )
!       - Copies fortran logical variable to MATLAB mxArray logical data
!
!     mwPointer function mxCreateLogicalArray( ndim, dims )
!       - Creates MATLAB mxArray logical array
!
!     mwPointer function mxCreateLogicalMatrix( m, n )
!       - Creates MATLAB mxArray logical matrix
!
!     mwPointer function mxCreateLogicalScalar( fortran )
!       - Creates MATLAB mxArray logical scalar from Fortran logical scalar
!
!     logical mxGetLogicalScalar( mxarray )
!       - Returns Fortran logical scalar from the MATLAB mxArray logical scalar
!
!     mwPointer function mxGetLogicals( mxarray )
!       - Returns the address of the first logical in the mxArray.
!         Returns 0 if the specified array is not a logical array.
!
!     integer*4 mxIsLogicalScalar( mxarray )
!       - Returns 1 if mxarray is logical class with size 1 x 1
!         Returns 0 otherwise
!
!     integer*4 mxIsLogicalScalarTrue( mxarray )
!       - Returns 1 if mxarray is logical class with size 1 x 1 and is non-zero
!         Returns 0 otherwise
!        (Note: This is very similar to mxGetLogicalScalar)
!
! The logical*1 specific routines supplied are:
!
!     subroutine mxCopyPtrToLogical1( Ptr, fortran, n )
!       - Copies MATLAB mxArray logical data to fortran logical*1 variable
!
!     subroutine mxCopyLogical1ToPtr( fortran, Ptr, n )
!       - Copies fortran logical*1 variable to MATLAB mxArray logical data
!
!     mwPointer function mxCreateLogical1Scalar( fortran )
!       - Creates MATLAB mxArray logical scalar from Fortran logical*1 scalar
!
!     logical mxGetLogical1Scalar( mxarray )
!       - Returns Fortran logical*1 scalar from the MATLAB mxArray logical scalar
!
! The logical*2 specific routines supplied are:
!
!     subroutine mxCopyPtrToLogical2( Ptr, fortran, n )
!       - Copies MATLAB mxArray logical data to fortran logical*2 variable
!
!     subroutine mxCopyLogical2ToPtr( fortran, Ptr, n )
!       - Copies fortran logical*2 variable to MATLAB mxArray logical data
!
!     mwPointer function mxCreateLogical2Scalar( fortran )
!       - Creates MATLAB mxArray logical scalar from Fortran logical*2 scalar
!
!     logical mxGetLogical2Scalar( mxarray )
!       - Returns Fortran logical*2 scalar from the MATLAB mxArray logical scalar
!
! The logical*4 specific routines supplied are:
!
!     subroutine mxCopyPtrToLogical4( Ptr, fortran, n )
!       - Copies MATLAB mxArray logical data to fortran logical*4 variable
!
!     subroutine mxCopyLogical4ToPtr( fortran, Ptr, n )
!       - Copies fortran logical*4 variable to MATLAB mxArray logical data
!
!     mwPointer function mxCreateLogical4Scalar( fortran )
!       - Creates MATLAB mxArray logical scalar from Fortran logical*4 scalar
!
!     logical mxGetLogical4Scalar( mxarray )
!       - Returns Fortran logical*4 scalar from the MATLAB mxArray logical scalar
!
! Special Note:
!
!   Several of the routines use MATLAB supplied functions such as mxGetData
!   and mxCreateNumericMatrix and mxCreateNumericArray with logical class
!   variables. Although the MATLAB doc does not specifically mention that
!   these functions can be used with logical class variables, it does seem
!   to work ok. The only alternative to using these functions would be to
!   call mexCallMATLAB or engCallMATLAB to convert the logical class variables
!   to/from an int8 class. This would slow up the code and force me to make
!   two versions of the code, one for mex files and another for engine
!   applications, so I opted not to do it this way.
!
!------------------------------------------------------------------------------

!==============================================================================
!==============================================================================
!
!  Default logical routines
!
!==============================================================================
!==============================================================================

!------------------------------------------------------------------------------
!
! routine:     mxCopyPtrToLogical
!
! This routine copies MATLAB logical data to a Fortran logical variable.
!
! Syntax:
!          call mxCopyPtrToLogical( Ptr, fortran, n )
! Where:
!          mwPointer, intent(in) :: Ptr
!          logical, intent(out) :: fortran(n)
!          mwSize, intent(in) :: n
!
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          fortran = Fortran logical variable.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the fortran logical values
!         based on the underlying MATLAB logical data being zero/non-zero.
!
!------------------------------------------------------------------------------

      subroutine mxCopyPtrToLogical( Ptr, fortran, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      mwPointer, intent(in) :: Ptr
      logical, intent(out) :: fortran(n)
!-----
      call mxCopyInteger1ToLogical( %VAL(Ptr), fortran, n )
      return
      end subroutine mxCopyPtrToLogical

!------------------------------------------------------------------------------
!
! routine:     mxCopyLogicalToPtr
!
! This routine copies a Fortran logical variable to MATLAB logical data.
!
! Syntax:
!          call mxCopyLogicalToPtr( fortran, Ptr, n )
! Where:
!          logical, intent(in) :: fortran(n)
!          mwPointer, intent(in) :: Ptr
!          mwSize, intent(in) :: n
!
!          fortran = Fortran logical variable.
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the MATLAB logical data as
!         0 or 1 based on the fortran logical values.
!
!------------------------------------------------------------------------------

      subroutine mxCopyLogicalToPtr( fortran, Ptr, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical, intent(in) :: fortran(n)
      mwPointer, intent(in) :: Ptr
!-----
      call mxCopyLogicalToInteger1( fortran, %VAL(Ptr), n )
      return
      end subroutine mxCopyLogicalToPtr

!------------------------------------------------------------------------------
!
! routine:     mxCreateLogicalArray
!
! This routine creates a MATLAB mxArray logical array.
!
! Syntax:
!          mx = mxCreateLogicalArray( ndim, ndims )
! Where:
!          mwPointer mx
!          mwSize, intent(in) :: ndim
!          mwSize, intent(in) :: dims(ndim)
!
!          mx   = A pointer to the output MATLAB mxArray logical variable
!          ndim = Number of dimensions
!          dims = The dimensions array
!
! Notes:  Initializes the logical data to 0 (false).
!         Uses the mxCreateNumericalArray routine to do this. Even though
!         the MATLAB documentation does not specifically allow for the
!         logical class with the mxCreateNumericalArray routine, it seems to
!         work ok.
!
!         Returns a pointer to the created mxArray, if successful.
!         If unsuccessful in a stand-alone (non-MEX-file) application,
!         mxCreateLogicalArray returns 0. If unsuccessful in a MEX-file,
!         the MEX-file terminates and control returns to the MATLAB prompt.
!         mxCreateLogicalArray is unsuccessful when there is not enough free
!         heap space to create the mxArray.
!
!------------------------------------------------------------------------------

      mwPointer function mxCreateLogicalArray( ndim, dims )
      implicit none
!-ARG
      mwSize, intent(in) :: ndim
      mwSize, intent(in) :: dims(ndim)
!-LOC
      integer*4, parameter :: ComplexFlag = 0
      integer*4 classid
!-----
      classid = mxClassIDFromClassName("logical")
      if( classid == 0 ) then
          mxCreateLogicalArray = 0
      else
          mxCreateLogicalArray = mxCreateNumericArray                   &
     &                          (ndim, dims, classid, ComplexFlag)
      endif
      return
      end function mxCreateLogicalArray

!------------------------------------------------------------------------------
!
! routine:     mxCreateLogicalMatrix
!
! This routine creates a MATLAB mxArray logical matrix.
!
! Syntax:
!          mx = mxCreateLogicalMatrix( m, n )
! Where:
!          mwPointer mx
!          mwSize, intent(in) :: m
!          mwSize, intent(in) :: n
!
!          mx = A pointer to the output MATLAB mxArray logical variable
!          m  = The number of rows
!          n  = The number of columns
!
! Notes:  Initializes the logical data to 0 (false).
!         Uses the mxCreateNumericalMatrix routine to do this. Even though
!         the MATLAB documentation does not specifically allow for the
!         logical class with the mxCreateNumericalMatrix routine, it seems to
!         work ok.
!
!         Returns a pointer to the created mxArray, if successful.
!         If unsuccessful in a stand-alone (non-MEX-file) application,
!         mxCreateLogicalMatrix returns 0. If unsuccessful in a MEX-file,
!         the MEX-file terminates and control returns to the MATLAB prompt.
!         mxCreateLogicalMatrix is unsuccessful when there is not enough free
!         heap space to create the mxArray.
!
!------------------------------------------------------------------------------

      mwPointer function mxCreateLogicalMatrix( m, n )
      implicit none
!-ARG
      mwSize, intent(in) :: m
      mwSize, intent(in) :: n
!-LOC
      integer*4, parameter :: ComplexFlag = 0
      integer*4 classid
!-----
      classid = mxClassIDFromClassName("logical")
      if( classid == 0 ) then
          mxCreateLogicalMatrix = 0
      else
          mxCreateLogicalMatrix = mxCreateNumericMatrix                 &
     &                           (m, n, classid, ComplexFlag)
      endif
      return
      end function mxCreateLogicalMatrix

!------------------------------------------------------------------------------
!
! routine:     mxCreateLogicalScalar
!
! This routine creates a MATLAB mxArray logical scalar from Fortran logical.
!
! Syntax:
!          mx = mxCreateLogicalScalar( fortran )
! Where:
!          mwPointer mx
!          logical, intent(in) :: fortran
!
!          mx      = A pointer to the output MATLAB mxArray logical variable
!          fortran = The Fortran logical scalar variable
!
! Notes:  Uses the mxCreateNumericalMatrix routine to do this. Even though
!         the MATLAB documentation does not specifically allow for the
!         logical class with the mxCreateNumericalMatrix routine, it seems to
!         work ok. Also uses mxGetData with logical class variables.
!
!         Returns a pointer to the created mxArray, if successful.
!         If unsuccessful in a stand-alone (non-MEX-file) application,
!         mxCreateLogicalScalar returns 0. If unsuccessful in a MEX-file,
!         the MEX-file terminates and control returns to the MATLAB prompt.
!         mxCreateLogicalScalar is unsuccessful when there is not enough free
!         heap space to create the mxArray.
!
!------------------------------------------------------------------------------

      mwPointer function mxCreateLogicalScalar( fortran )
      implicit none
!-ARG
      logical, intent(in) :: fortran
!-LOC
      mwSize, parameter :: m = 1
!-----
      mxCreateLogicalScalar = mxCreateLogicalMatrix( m, m )
      if( mxCreateLogicalScalar /= 0 ) then
        call mxCopyLogicalToPtr( (/fortran/),                           &
     &       mxGetData( mxCreateLogicalScalar ), m)
      endif
      return
      end function mxCreateLogicalScalar

!------------------------------------------------------------------------------
!
! routine:     mxGetLogicalScalar
!
! This routine converts the 1st element of a MATLAB mxArray logical variable
! to a Fortran logical scalar.
!
! Syntax:
!          fortran = mxGetLogicalScalar( mx )
! Where:
!          logical fortran
!          mwPointer, intent(in) :: mx
!
!          fortran = The Fortran logical scalar variable
!          mx      = A pointer to the input MATLAB mxArray logical variable
!
! Notes:  Uses the mxGetData routine to do this. Even though the MATLAB doc
!         does not specifically allow for the logical class to work with the
!         mxGetData routine, it seems to work ok. Uses F77 style %VAL to trick
!         the code into using the underlying MATLAB logical data as integer*1
!         data using a support function.
!
!------------------------------------------------------------------------------

      logical function mxGetLogicalScalar( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-FUN
      logical, external :: mxDataToLogical
!-LOC
      mwPointer logicaldata
!-----
      mxGetLogicalScalar = .false.
      if( mxIsLogical(mx) /= 0 ) then
        if( mxGetNumberOfElements(mx) == 1 ) then
          logicaldata = mxGetData( mx )
          mxGetLogicalScalar = mxDataToLogical( %VAL(logicaldata) )
        endif
      endif
      return
      end function mxGetLogicalScalar

!------------------------------------------------------------------------------
!
! routine:     mxGetLogicals
!
! This routine returns the address of the first logical in the mxArray.
! Returns NULL if the specified array is not a logical array.
!
! Syntax:
!          Ptr = mxGetLogicals( mx )
! Where:
!          mwPointer :: Ptr
!          mwPointer, intent(in) :: mx
!
!          Ptr     = The address of the first logical in the mxArray mx.
!          mx      = A pointer to a MATLAB mxArray logical variable
!
! Notes:  Uses the mxGetData routine to do this. Even though the MATLAB doc
!         does not specifically allow for the logical class to work with the
!         mxGetData routine, it seems to work ok.
!
!------------------------------------------------------------------------------

      mwPointer function mxGetLogicals( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      mwPointer logicaldata
!-----
      if( mxIsLogical(mx) /= 0 ) then
        mxGetLogicals = mxGetData(mx)
      else
        mxGetLogicals = 0
      endif
      return
      end function mxGetLogicals

!------------------------------------------------------------------------------
!
! routine:     mxIsLogicalScalar
!
! This routine returns 1 if the mxArray is of class mxLogical and has
! 1-by-1 dimensions, and returns 0 otherwise
!
! Syntax:
!          k = mxIsLogicalScalar( mx )
! Where:
!          integer*4 :: k
!          mwPointer, intent(in) :: mx
!
!          k     = The return value
!          mx    = A pointer to a MATLAB mxArray logical variable
!
!------------------------------------------------------------------------------

      integer*4 function mxIsLogicalScalar( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-----
      mxIsLogicalScalar = 0
      if( mxIsLogical(mx) /= 0 ) then
        if( mxGetNumberOfElements(mx) == 1 ) then
          mxIsLogicalScalar = 1
        endif
      endif
      return
      end function mxIsLogicalScalar

!------------------------------------------------------------------------------
!
! routine:     mxIsLogicalScalarTrue
!
! This routine returns 1 if the value of the mxArray's logical, scalar
! element is true, and 0 otherwise.
!
! Syntax:
!          k = mxIsLogicalScalarTrue( mx )
! Where:
!          ineger*4 :: k
!          mwPointer, intent(in) :: mx
!
!          k       = The result
!          mx      = A pointer to the input MATLAB mxArray logical variable
!
! Notes:  Uses the mxGetData routine to do this. Even though the MATLAB doc
!         does not specifically allow for the logical class to work with the
!         mxGetData routine, it seems to work ok. Uses F77 style %VAL to trick
!         the code into using the underlying MATLAB logical data as integer*1
!         data using a support function.
!
!------------------------------------------------------------------------------

      integer*4 function mxIsLogicalScalarTrue( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-FUN
      logical, external :: mxLSDataToLogical
!-LOC
      mwPointer logicaldata
!-----
      mxIsLogicalScalarTrue = 0
      if( mxIsLogical(mx) /= 0 ) then
        if( mxGetNumberOfElements(mx) == 1 ) then
          logicaldata = mxGetData( mx )
          if( mxLSDataToLogical( %VAL(logicaldata) ) ) then
            mxIsLogicalScalarTrue = 1
          endif
        endif
      endif
      return
      end function mxIsLogicalScalarTrue

!==============================================================================
!==============================================================================
!
!  logical*1 specific routines
!
!==============================================================================
!==============================================================================

!------------------------------------------------------------------------------
!
! routine:     mxCopyPtrToLogical1
!
! This routine copies MATLAB logical data to a Fortran logical*1 variable.
!
! Syntax:
!          call mxCopyPtrToLogical1( Ptr, fortran, n )
! Where:
!          mwPointer, intent(in) :: Ptr
!          logical*1, intent(out) :: fortran(n)
!          mwSize, intent(in) :: n
!
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          fortran = Fortran logical*1 variable.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the fortran logical values
!         based on the underlying MATLAB logical data being zero/non-zero.
!
!------------------------------------------------------------------------------

      subroutine mxCopyPtrToLogical1( Ptr, fortran, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      mwPointer, intent(in) :: Ptr
      logical*1, intent(out) :: fortran(n)
!-----
      call mxCopyInteger1ToLogical1( %VAL(Ptr), fortran, n )
      return
      end subroutine mxCopyPtrToLogical1

!------------------------------------------------------------------------------
!
! routine:     mxCopyLogical1ToPtr
!
! This routine copies a Fortran logical*1 variable to MATLAB logical data.
!
! Syntax:
!          call mxCopyLogical1ToPtr( fortran, Ptr, n )
! Where:
!          logical*1, intent(in) :: fortran(n)
!          mwPointer, intent(in) :: Ptr
!          mwSize, intent(in) :: n
!
!          fortran = Fortran logical*1 variable.
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the MATLAB logical data as
!         0 or 1 based on the fortran logical values.
!
!------------------------------------------------------------------------------

      subroutine mxCopyLogical1ToPtr( fortran, Ptr, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical*1, intent(in) :: fortran(n)
      mwPointer, intent(in) :: Ptr
!-----
      call mxCopyLogical1ToInteger1( fortran, %VAL(Ptr), n )
      return
      end subroutine mxCopyLogical1ToPtr

!------------------------------------------------------------------------------
!
! routine:     mxCreateLogical1Scalar
!
! This routine creates a MATLAB mxArray logical scalar from Fortran logical*1.
!
! Syntax:
!          mx = mxCreateLogical1Scalar( fortran )
! Where:
!          mwPointer mx
!          logical*1, intent(in) :: fortran
!
!          mx      = A pointer to the output MATLAB mxArray logical variable
!          fortran = The Fortran logical*1 scalar variable
!
! Notes:  Uses the mxCreateNumericalMatrix routine to do this. Even though
!         the MATLAB documentation does not specifically allow for the
!         logical class with the mxCreateNumericalMatrix routine, it seems to
!         work ok. Also uses mxGetData with logical class variables.
!
!         Returns a pointer to the created mxArray, if successful.
!         If unsuccessful in a stand-alone (non-MEX-file) application,
!         mxCreateLogical1Scalar returns 0. If unsuccessful in a MEX-file,
!         the MEX-file terminates and control returns to the MATLAB prompt.
!         mxCreateLogical1Scalar is unsuccessful when there is not enough free
!         heap space to create the mxArray.
!
!------------------------------------------------------------------------------

      mwPointer function mxCreateLogical1Scalar( fortran )
      implicit none
!-ARG
      logical*1, intent(in) :: fortran
!-LOC
      mwSize, parameter :: m = 1
!-----
      mxCreateLogical1Scalar = mxCreateLogicalMatrix( m, m )
      if( mxCreateLogical1Scalar /= 0 ) then
        call mxCopyLogical1ToPtr( (/fortran/),                          &
     &       mxGetData( mxCreateLogical1Scalar ), m)
      endif
      return
      end function mxCreateLogical1Scalar

!------------------------------------------------------------------------------
!
! routine:     mxGetLogical1Scalar
!
! This routine converts the 1st element of a MATLAB mxArray logical variable
! to a Fortran logical*1 scalar.
!
! Syntax:
!          fortran = mxGetLogical1Scalar( mx )
! Where:
!          logical*1 fortran
!          mwPointer, intent(in) :: mx
!
!          fortran = The Fortran logical*1 scalar variable
!          mx      = A pointer to the input MATLAB mxArray logical variable
!
! Notes:  Uses the mxGetData routine to do this. Even though the MATLAB doc
!         does not specifically allow for the logical class to work with the
!         mxGetData routine, it seems to work ok. Uses F77 style %VAL to trick
!         the code into using the underlying MATLAB logical data as integer*1
!         data using a support function.
!
!------------------------------------------------------------------------------

      logical*1 function mxGetLogical1Scalar( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-FUN
      logical*1, external :: mxDataToLogical1
!-LOC
      mwPointer logicaldata
!-----
      mxGetLogical1Scalar = .false.
      if( mxIsLogical(mx) /= 0 ) then
        if( mxGetNumberOfElements(mx) == 1 ) then
          logicaldata = mxGetData( mx )
          mxGetLogical1Scalar = mxDataToLogical1( %VAL(logicaldata) )
        endif
      endif
      return
      end function mxGetLogical1Scalar

!==============================================================================
!==============================================================================
!
!  logical*2 specific routines
!
!==============================================================================
!==============================================================================

!------------------------------------------------------------------------------
!
! routine:     mxCopyPtrToLogical2
!
! This routine copies MATLAB logical data to a Fortran logical*2 variable.
!
! Syntax:
!          call mxCopyPtrToLogical2( Ptr, fortran, n )
! Where:
!          mwPointer, intent(in) :: Ptr
!          logical*2, intent(out) :: fortran(n)
!          mwSize, intent(in) :: n
!
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          fortran = Fortran logical*2 variable.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the fortran logical values
!         based on the underlying MATLAB logical data being zero/non-zero.
!
!------------------------------------------------------------------------------

      subroutine mxCopyPtrToLogical2( Ptr, fortran, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      mwPointer, intent(in) :: Ptr
      logical*2, intent(out) :: fortran(n)
!-----
      call mxCopyInteger1ToLogical2( %VAL(Ptr), fortran, n )
      return
      end subroutine mxCopyPtrToLogical2

!------------------------------------------------------------------------------
!
! routine:     mxCopyLogical2ToPtr
!
! This routine copies a Fortran logical variable to MATLAB logical*2 data.
!
! Syntax:
!          call mxCopyLogical2ToPtr( fortran, Ptr, n )
! Where:
!          logical*2, intent(in) :: fortran(n)
!          mwPointer, intent(in) :: Ptr
!          mwSize, intent(in) :: n
!
!          fortran = Fortran logical*2 variable.
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the MATLAB logical data as
!         0 or 1 based on the fortran logical values.
!
!------------------------------------------------------------------------------

      subroutine mxCopyLogical2ToPtr( fortran, Ptr, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical*2, intent(in) :: fortran(n)
      mwPointer, intent(in) :: Ptr
!-----
      call mxCopyLogical2ToInteger1( fortran, %VAL(Ptr), n )
      return
      end subroutine mxCopyLogical2ToPtr

!------------------------------------------------------------------------------
!
! routine:     mxCreateLogical2Scalar
!
! This routine creates a MATLAB mxArray logical scalar from Fortran logical*2.
!
! Syntax:
!          mx = mxCreateLogicalScalar( fortran )
! Where:
!          mwPointer mx
!          logical*2, intent(in) :: fortran
!
!          mx      = A pointer to the output MATLAB mxArray logical variable
!          fortran = The Fortran logical*2 scalar variable
!
! Notes:  Uses the mxCreateNumericalMatrix routine to do this. Even though
!         the MATLAB documentation does not specifically allow for the
!         logical class with the mxCreateNumericalMatrix routine, it seems to
!         work ok. Also uses mxGetData with logical class variables.
!
!         Returns a pointer to the created mxArray, if successful.
!         If unsuccessful in a stand-alone (non-MEX-file) application,
!         mxCreateLogical2Scalar returns 0. If unsuccessful in a MEX-file,
!         the MEX-file terminates and control returns to the MATLAB prompt.
!         mxCreateLogical2Scalar is unsuccessful when there is not enough free
!         heap space to create the mxArray.
!
!------------------------------------------------------------------------------

      mwPointer function mxCreateLogical2Scalar( fortran )
      implicit none
!-ARG
      logical*2, intent(in) :: fortran
!-LOC
      mwSize, parameter :: m = 1
!-----
      mxCreateLogical2Scalar = mxCreateLogicalMatrix( m, m )
      if( mxCreateLogical2Scalar /= 0 ) then
        call mxCopyLogical2ToPtr( (/fortran/),                          &
     &       mxGetData( mxCreateLogical2Scalar ), m)
      endif
      return
      end function mxCreateLogical2Scalar

!------------------------------------------------------------------------------
!
! routine:     mxGetLogical2Scalar
!
! This routine converts the 1st element of a MATLAB mxArray logical variable
! to a Fortran logical*2 scalar.
!
! Syntax:
!          fortran = mxGetLogical2Scalar( mx )
! Where:
!          logical*2 fortran
!          mwPointer, intent(in) :: mx
!
!          fortran = The Fortran logical*2 scalar variable
!          mx      = A pointer to the input MATLAB mxArray logical variable
!
! Notes:  Uses the mxGetData routine to do this. Even though the MATLAB doc
!         does not specifically allow for the logical class to work with the
!         mxGetData routine, it seems to work ok. Uses F77 style %VAL to trick
!         the code into using the underlying MATLAB logical data as integer*1
!         data using a support function.
!
!------------------------------------------------------------------------------

      logical*2 function mxGetLogical2Scalar( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-FUN
      logical*2, external :: mxDataToLogical2
!-LOC
      mwPointer logicaldata
!-----
      mxGetLogical2Scalar = .false.
      if( mxIsLogical(mx) /= 0 ) then
        if( mxGetNumberOfElements(mx) == 1 ) then
          logicaldata = mxGetData( mx )
          mxGetLogical2Scalar = mxDataToLogical2( %VAL(logicaldata) )
        endif
      endif
      return
      end function mxGetLogical2Scalar

!==============================================================================
!==============================================================================
!
!  logical*4 specific routines
!
!==============================================================================
!==============================================================================

!------------------------------------------------------------------------------
!
! routine:     mxCopyPtrToLogical4
!
! This routine copies MATLAB logical data to a Fortran logical*4 variable.
!
! Syntax:
!          call mxCopyPtrToLogical4( Ptr, fortran, n )
! Where:
!          mwPointer, intent(in) :: Ptr
!          logical*4, intent(out) :: fortran(n)
!          mwSize, intent(in) :: n
!
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          fortran = Fortran logical*4 variable.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the fortran logical values
!         based on the underlying MATLAB logical data being zero/non-zero.
!
!------------------------------------------------------------------------------

      subroutine mxCopyPtrToLogical4( Ptr, fortran, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      mwPointer, intent(in) :: Ptr
      logical*4, intent(out) :: fortran(n)
!-----
      call mxCopyInteger1ToLogical4( %VAL(Ptr), fortran, n )
      return
      end subroutine mxCopyPtrToLogical4

!------------------------------------------------------------------------------
!
! routine:     mxCopyLogical4ToPtr
!
! This routine copies a Fortran logical variable to MATLAB logical*4 data.
!
! Syntax:
!          call mxCopyLogical4ToPtr( fortran, Ptr, n )
! Where:
!          logical*4, intent(in) :: fortran(n)
!          mwPointer, intent(in) :: Ptr
!          mwSize, intent(in) :: n
!
!          fortran = Fortran logical*4 variable.
!          Ptr     = Pointer to the logical data, which is the result of
!                    a mxGetData(mxarray) call.
!          n       = Number of elements to copy.
!
! Notes:  Both the underlying mxArray logical variable and the fortran
!         variable can be multi-dimensional in the calling routine. Uses
!         F77 style %VAL to trick the code into using the underlying MATLAB
!         logical data as integer*1 data using a support function. Doesn't
!         actually copy any data, but creates the MATLAB logical data as
!         0 or 1 based on the fortran logical values.
!
!------------------------------------------------------------------------------

      subroutine mxCopyLogical4ToPtr( fortran, Ptr, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical*4, intent(in) :: fortran(n)
      mwPointer, intent(in) :: Ptr
!-----
      call mxCopyLogical4ToInteger1( fortran, %VAL(Ptr), n )
      return
      end subroutine mxCopyLogical4ToPtr

!------------------------------------------------------------------------------
!
! routine:     mxCreateLogical4Scalar
!
! This routine creates a MATLAB mxArray logical scalar from Fortran logical*4.
!
! Syntax:
!          mx = mxCreateLogical4Scalar( fortran )
! Where:
!          mwPointer mx
!          logical*4, intent(in) :: fortran
!
!          mx      = A pointer to the output MATLAB mxArray logical variable
!          fortran = The Fortran logical*4 scalar variable
!
! Notes:  Uses the mxCreateNumericalMatrix routine to do this. Even though
!         the MATLAB documentation does not specifically allow for the
!         logical class with the mxCreateNumericalMatrix routine, it seems to
!         work ok. Also uses mxGetData with logical class variables.
!
!         Returns a pointer to the created mxArray, if successful.
!         If unsuccessful in a stand-alone (non-MEX-file) application,
!         mxCreateLogical4Scalar returns 0. If unsuccessful in a MEX-file,
!         the MEX-file terminates and control returns to the MATLAB prompt.
!         mxCreateLogical4Scalar is unsuccessful when there is not enough free
!         heap space to create the mxArray.
!
!------------------------------------------------------------------------------

      mwPointer function mxCreateLogical4Scalar( fortran )
      implicit none
!-ARG
      logical*4, intent(in) :: fortran
!-LOC
      mwSize, parameter :: m = 1
!-----
      mxCreateLogical4Scalar = mxCreateLogicalMatrix( m, m )
      if( mxCreateLogical4Scalar /= 0 ) then
        call mxCopyLogical4ToPtr( (/fortran/),                          &
     &       mxGetData( mxCreateLogical4Scalar ), m)
      endif
      return
      end function mxCreateLogical4Scalar

!------------------------------------------------------------------------------
!
! routine:     mxGetLogical4Scalar
!
! This routine converts the 1st element of a MATLAB mxArray logical variable
! to a Fortran logical*4 scalar.
!
! Syntax:
!          fortran = mxGetLogical4Scalar( mx )
! Where:
!          logical*4 fortran
!          mwPointer, intent(in) :: mx
!
!          fortran = The Fortran logical*4 scalar variable
!          mx      = A pointer to the input MATLAB mxArray logical variable
!
! Notes:  Uses the mxGetData routine to do this. Even though the MATLAB doc
!         does not specifically allow for the logical class to work with the
!         mxGetData routine, it seems to work ok. Uses F77 style %VAL to trick
!         the code into using the underlying MATLAB logical data as integer*1
!         data using a support function.
!
!------------------------------------------------------------------------------

      logical*4 function mxGetLogical4Scalar( mx )
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-FUN
      logical*4, external :: mxDataToLogical4
!-LOC
      mwPointer logicaldata
!-----
      mxGetLogical4Scalar = .false.
      if( mxIsLogical(mx) /= 0 ) then
        if( mxGetNumberOfElements(mx) == 1 ) then
          logicaldata = mxGetData( mx )
          mxGetLogical4Scalar = mxDataToLogical4( %VAL(logicaldata) )
        endif
      endif
      return
      end function mxGetLogical4Scalar

!-------------------------------------------------------------------------------      
! End of Logical functions
!-------------------------------------------------------------------------------
#ifndef NOCOMPLEX16

      subroutine mxCopyComplex32ToPtr(y, pr, pi, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      complex(16), intent(in) :: y(n)
      mwPointer, intent(in) :: pr, pi
!-----
      call mxCopyComplex32ToReal1616(y, %VAL(pr), %VAL(pi), n)
      end subroutine mxCopyComplex32ToPtr

#endif
!-------------------------------------------------------------------------------
#ifndef NOCOMPLEX16

      subroutine mxCopyPtrToComplex32(pr, pi, y, n)
      implicit none
!-ARG
      mwPointer, intent(in) :: pr, pi
      mwSize, intent(in) :: n
      complex(16), intent(in) :: y(n)
!-----
      call mxCopyReal1616ToComplex32(%VAL(pr), %VAL(pi), y, n)
      end subroutine mxCopyPtrToComplex32

#endif
!-------------------------------------------------------------------------------
#ifndef NOREAL16

      subroutine mxCopyPtrToReal16(px, y, n)
      implicit none
!-ARG
      mwPointer, intent(in) :: px
      mwSize, intent(in) :: n
      real(16), intent(in) :: y(n)
!-----
      call mxCopyReal16ToReal16(%VAL(px), y, n)
      end subroutine mxCopyPtrToReal16

#endif
!-------------------------------------------------------------------------------
#ifndef NOREAL16

      subroutine mxCopyReal16ToPtr(y, px, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(16), intent(in) :: y(n)
      mwPointer, intent(in) :: px
!-----
      call mxCopyReal16ToReal16(y, %VAL(px), n)
      end subroutine mxCopyReal16ToPtr

#endif
!-------------------------------------------------------------------------------
#ifndef NOINTEGER8

      subroutine mxCopyInteger8ToPtr(y, px, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      integer(8), intent(in) :: y(n)
      mwPointer, intent(in) :: px
!-----
      call mxCopyInteger8ToInteger8(y, %VAL(px), n)
      end subroutine mxCopyInteger8ToPtr

#endif
!-------------------------------------------------------------------------------
#ifndef NOINTEGER8

      subroutine mxCopyPtrToInteger8(px, y, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      mwPointer, intent(in) :: px
      integer(8), intent(out) :: y(n)
!-----
      call mxCopyInteger8ToInteger8(%VAL(px), y, n)
      end subroutine mxCopyPtrToInteger8

#endif
      
!-----------------------------------------------------      
! Specific Fortan Pointer functions
!-----------------------------------------------------
      
      function fpGetPr0Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx0
      common /MatlabAPI_COMA0/ Apx0
!-LOC
      mwPointer :: pr
!-----
      nullify( fp )
      if( mxIsDouble(mx) == 1 ) then
          if( mxGetNumberOfElements( mx ) == 1 ) then
              pr = mxGetPr( mx )
              call MatlabAPI_COM_Apx0( %VAL(pr) )
              fp => Apx0
          endif
      endif
      return
      end function fpGetPr0Double
!----------------------------------------------------------------------
      function fpGetPr1Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: pr
      mwSize :: N
      mwSize, pointer :: jc(:)
      mwSize, parameter :: stride = 1
!-----
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 ) then
          pr = mxGetPr( mx )
          N = mxGetNumberOfElements( mx )
          call MatlabAPI_COM_Apx1( %VAL(pr), stride, N )
          fp => Apx1
      elseif( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 1 ) then
          pr = mxGetPr( mx )
          jc = fpGetJc( mx )
          N = mxGetN( mx )
          call MatlabAPI_COM_Apx1( %VAL(pr), stride, jc(N+1) )
          fp => Apx1
      else
          nullify( fp )
      endif
      return
      end function fpGetPr1Double







      function fpGetPr1Single( mx ) result(fp)
      implicit none
      real(4), pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(4), pointer :: Apx1(:)
      common /MatlabAPI_COMSingleA1/ Apx1
!-LOC
      mwPointer :: pr
      mwSize :: N
      mwSize, pointer :: jc(:)
      mwSize, parameter :: stride = 1
!-----
!----- NB NEED TO ADD A CHECK THAT IT IS SINGLE
      if( mxIsSparse(mx) == 0 ) then
          pr = mxGetPr( mx )
          N = mxGetNumberOfElements( mx )
          call MatlabAPI_COM_SingleApx1( %VAL(pr), stride, N )
          fp => Apx1
      else
          nullify( fp )
      endif
      return
      end function fpGetPr1Single





!----------------------------------------------------------------------
      function fpGetPr2Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize, pointer :: dims(:)
!-----
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxGetNumberOfDimensions(mx) == 2 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          call MatlabAPI_COM_Apx2( %VAL(pr), stride, dims )
          fp => Apx2
      else
          nullify( fp )
      endif
      return
      end function fpGetPr2Double

!-----------------------------------------------------------------------
      function fpGetPr2Single( mx ) result(fp)
      implicit none
      real(4), pointer :: fp(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(4), pointer :: Apx2(:,:)
      common /MatlabAPI_COMSingleA2/ Apx2
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize, pointer :: dims(:)
!----- NB NEED TO ADD A CHECK THAT IT IS SINGLE
      if(mxIsSparse(mx) == 0 .and.           &
     &    mxGetNumberOfDimensions(mx) == 2 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          call MatlabAPI_COM_SingleApx2( %VAL(pr), stride, dims )
          fp => Apx2
      else
          nullify( fp )
      endif
      return
      end function fpGetPr2Single



!----------------------------------------------------------------------
      function fpGetPr3Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(3)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.      &
     &    ndim <= 3 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx3( %VAL(pr), stride, dimz )
          fp => Apx3
      else
          nullify( fp )
      endif
      return
      end function fpGetPr3Double


      function fpGetPr3Single( mx ) result(fp)
      implicit none
      real(4), pointer :: fp(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(4), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMSingleA3/ Apx3
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(3)
!----- NB NEED TO ADD A CHECK THAT IT IS SINGLE
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsSparse(mx) == 0 .and.      &
     &    ndim <= 3 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_SingleApx3( %VAL(pr), stride, dimz )
          fp => Apx3
      else
          nullify( fp )
      endif
      return
      end function fpGetPr3Single













!----------------------------------------------------------------------
      function fpGetPr4Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(4)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    ndim <= 4 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx4( %VAL(pr), stride, dimz )
          fp => Apx4
      else
          nullify( fp )
      endif
      return
      end function fpGetPr4Double
!----------------------------------------------------------------------
      function fpGetPr5Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(5)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    ndim <= 5 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx5( %VAL(pr), stride, dimz )
          fp => Apx5
      else
          nullify( fp )
      endif
      return
      end function fpGetPr5Double
!----------------------------------------------------------------------
      function fpGetPr6Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(6)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    ndim <= 6 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx6( %VAL(pr), stride, dimz )
          fp => Apx6
      else
          nullify( fp )
      endif
      return
      end function fpGetPr6Double
!----------------------------------------------------------------------
      function fpGetPr7Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pr
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(7)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    ndim <= 7 ) then
          pr = mxGetPr( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx7( %VAL(pr), stride, dimz )
          fp => Apx7
      else
          nullify( fp )
      endif
      return
      end function fpGetPr7Double
!----------------------------------------------------------------------
      function fpGetPi0Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx0
      common /MatlabAPI_COMA0/ Apx0
!-LOC
      mwPointer :: pi
!-----
      nullify( fp )
      if( mxIsDouble(mx) == 1 .and. mxIsComplex(mx) == 1 ) then
          if( mxGetNumberOfElements( mx ) == 1 ) then
              pi = mxGetPi( mx )
              call MatlabAPI_COM_Apx0( %VAL(pi) )
              fp => Apx0
          endif
      endif
      return
      end function fpGetPi0Double
!----------------------------------------------------------------------
      function fpGetPi1Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize :: N
      mwSize, pointer :: jc(:)
!-----
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 ) then
          pi = mxGetPi( mx )
          N = mxGetNumberOfElements( mx )
          call MatlabAPI_COM_Apx1( %VAL(pi), stride, N )
          fp => Apx1
      elseif( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 1 .and.       &
     &        mxIsComplex(mx) ) then
          pi = mxGetPi( mx )
          N = mxGetN( mx )
          jc = fpGetJc( mx )
          call MatlabAPI_COM_Apx1( %VAL(pi), stride, jc(N+1) )
          fp => Apx1
      else
          nullify( fp )
      endif
      return
      end function fpGetPi1Double
!----------------------------------------------------------------------
      function fpGetPi2Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize, pointer :: dims(:)
!-----
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 .and.                                    &
     &    mxGetNumberOfDimensions(mx) == 2 ) then
          pi = mxGetPi( mx )
          dims => fpGetDimensions( mx )
          call MatlabAPI_COM_Apx2( %VAL(pi), stride, dims )
          fp => Apx2
      else
          nullify( fp )
      endif
      return
      end function fpGetPi2Double
!----------------------------------------------------------------------
      function fpGetPi3Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(3)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 .and. ndim <= 3 ) then
          pi = mxGetPi( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx3( %VAL(pi), stride, dimz )
          fp => Apx3
      else
          nullify( fp )
      endif
      return
      end function fpGetPi3Double
!----------------------------------------------------------------------
      function fpGetPi4Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(4)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 .and. ndim <= 4 ) then
          pi = mxGetPi( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx4( %VAL(pi), stride, dimz )
          fp => Apx4
      else
          nullify( fp )
      endif
      return
      end function fpGetPi4Double
!----------------------------------------------------------------------
      function fpGetPi5Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(5)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 .and. ndim <= 5 ) then
          pi = mxGetPi( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx5( %VAL(pi), stride, dimz )
          fp => Apx5
      else
          nullify( fp )
      endif
      return
      end function fpGetPi5Double
!----------------------------------------------------------------------
      function fpGetPi6Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(6)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 .and. ndim <= 6 ) then
          pi = mxGetPi( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx6( %VAL(pi), stride, dimz )
          fp => Apx6
      else
          nullify( fp )
      endif
      return
      end function fpGetPi6Double
!----------------------------------------------------------------------
      function fpGetPi7Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwSize, parameter :: stride = 1
      mwPointer :: pi
      mwSize :: ndim
      mwSize, pointer :: dims(:)
      mwSize :: dimz(7)
!-----
      ndim = mxGetNumberOfDimensions(mx)
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 0 .and.           &
     &    mxIsComplex(mx) == 1 .and. ndim <= 7 ) then
          pi = mxGetPi( mx )
          dims => fpGetDimensions( mx )
          dimz = 1
          dimz(1:ndim) = dims
          call MatlabAPI_COM_Apx7( %VAL(pi), stride, dimz )
          fp => Apx7
      else
          nullify( fp )
      endif
      return
      end function fpGetPi7Double
!----------------------------------------------------------------------
      function fpGetPrCopy0Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp
!-----
      mp => fpGetPr0(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      fp => fpAllocate0( )
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy0Double      
!----------------------------------------------------------------------
      function fpGetPrCopy1Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:)
      mwSize :: sz
!-----
      mp => fpGetPr1(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz = size(mp)
      fp => fpAllocate1Double(sz)
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy1Double      
!----------------------------------------------------------------------
      function fpGetPrCopy2Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:)
      mwSize :: sz(2)
!-----
      mp => fpGetPr2(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      fp => fpAllocate2Double(sz(1),sz(2))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy2Double      
!----------------------------------------------------------------------
      function fpGetPrCopy3Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:)
      mwSize :: sz(3)
!-----
      mp => fpGetPr3(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      fp => fpAllocate3Double(sz(1),sz(2),sz(3))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy3Double
!----------------------------------------------------------------------
      function fpGetPrCopy4Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:)
      mwSize :: sz(4)
!-----
      mp => fpGetPr4(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      fp => fpAllocate4Double(sz(1),sz(2),sz(3),sz(4))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy4Double
!----------------------------------------------------------------------
      function fpGetPrCopy5Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:)
      mwSize :: sz(5)
!-----
      mp => fpGetPr5(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      fp => fpAllocate5Double(sz(1),sz(2),sz(3),sz(4),sz(5))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy5Double
!----------------------------------------------------------------------
      function fpGetPrCopy6Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:,:)
      mwSize :: sz(6)
!-----
      mp => fpGetPr6(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      sz(6) = size(mp,6)
      fp => fpAllocate6Double(sz(1),sz(2),sz(3),sz(4), sz(5),sz(6))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy6Double
!----------------------------------------------------------------------
      function fpGetPrCopy7Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:,:,:)
      mwSize :: sz(7)
!-----
      mp => fpGetPr7(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      sz(6) = size(mp,6)
      sz(7) = size(mp,7)
      fp => fpAllocate7Double(sz(1),sz(2),sz(3),sz(4), sz(5),sz(6)        &
     &                           , sz(7))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPrCopy7Double
!----------------------------------------------------------------------
      function fpGetPiCopy0Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp
!-----
      mp => fpGetPi0(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      fp => fpAllocate0Double( )
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy0Double      
!----------------------------------------------------------------------
      function fpGetPiCopy1Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:)
      mwSize :: sz
!-----
      mp => fpGetPi1(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz = size(mp,1)
      fp => fpAllocate1Double(sz)
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy1Double      
!----------------------------------------------------------------------
      function fpGetPiCopy2Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:)
      mwSize :: sz(2)
!-----
      mp => fpGetPi2(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      fp => fpAllocate2Double(sz(1),sZ(2))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy2Double      
!----------------------------------------------------------------------
      function fpGetPiCopy3Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:)
      mwSize :: sz(3)
!-----
      mp => fpGetPi3(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      fp => fpAllocate3Double(sz(1),sz(2),sz(3))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy3Double
!----------------------------------------------------------------------
      function fpGetPiCopy4Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:)
      mwSize :: sz(4)
!-----
      mp => fpGetPi4(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      fp => fpAllocate4Double(sz(1),sz(2),sz(3),sz(4))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy4Double
!----------------------------------------------------------------------
      function fpGetPiCopy5Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:)
      mwSize :: sz(5)
!-----
      mp => fpGetPi5(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      fp => fpAllocate5Double(sz(1),sz(2),sz(3),sz(4), sz(5))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy5Double
!----------------------------------------------------------------------
      function fpGetPiCopy6Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:,:)
      mwSize :: sz(6)
!-----
      mp => fpGetPi6(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      sz(6) = size(mp,6)
      fp => fpAllocate6Double(sz(1),sz(2),sz(3),sz(4),sz(5),sz(6))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy6Double
!----------------------------------------------------------------------
      function fpGetPiCopy7Double( mx ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:,:,:)
      mwSize :: sz(7)
!-----
      mp => fpGetPi7(mx)
      if( .not.associated(mp) ) then
          nullify(fp)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      sz(6) = size(mp,6)
      sz(7) = size(mp,7)
      fp => fpAllocate7Double(sz(1),sz(2),sz(3),sz(4),sz(5),sz(6),sz(7))
      if( .not.associated(fp) ) then
          return
      endif
      fp = mp
      return
      end function fpGetPiCopy7Double
!----------------------------------------------------------------------
      function fpGetPzCopy0Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp, mi
!-----
      mp => fpGetPr0(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      fz => fpAllocateZ0Double( )
      mi => fpGetPi0(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, 1)
      else
          call mxCopyReal80ToComplex16(mp, fz, 1)
      endif
      return
      end function fpGetPzCopy0Double      
!----------------------------------------------------------------------
      function fpGetPzCopy1Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:), mi(:)
      mwSize :: sz
!-----
      mp => fpGetPr1(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz = size(mp,1)
      fz => fpAllocateZ1Double(sz)
      mi => fpGetPi1(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy1Double      
!----------------------------------------------------------------------
      function fpGetPzCopy2Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:), mi(:,:)
      mwSize :: sz(2)
!-----
      mp => fpGetPr2(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      fz => fpAllocateZ2Double(sz(1),sz(2))
      mi => fpGetPi2(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy2Double
!----------------------------------------------------------------------
      function fpGetPzCopy3Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:), mi(:,:,:)
      mwSize :: sz(3)
!-----
      mp => fpGetPr3(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      fz => fpAllocateZ3Double(sz(1),sz(2),sz(3))
      mi => fpGetPi3(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy3Double
!----------------------------------------------------------------------
      function fpGetPzCopy4Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:), mi(:,:,:,:)
      mwSize :: sz(4)
!-----
      mp => fpGetPr4(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      fz => fpAllocateZ4Double(sz(1),sz(2),sz(3),sz(4))
      mi => fpGetPi4(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy4Double
!----------------------------------------------------------------------
      function fpGetPzCopy5Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:), mi(:,:,:,:,:)
      mwSize :: sz(5)
!-----
      mp => fpGetPr5(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      fz => fpAllocateZ5Double(sz(1),sz(2),sz(3),sz(4),                   &
     &                 sz(5))
      mi => fpGetPi5(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy5Double
!----------------------------------------------------------------------
      function fpGetPzCopy6Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:,:), mi(:,:,:,:,:,:)
      mwSize :: sz(6)
!-----
      mp => fpGetPr6(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      sz(6) = size(mp,6)
      fz => fpAllocateZ6Double(sz(1),sz(2),sz(3),sz(4),sz(5),sz(6))
      mi => fpGetPi6(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy6Double
!----------------------------------------------------------------------
      function fpGetPzCopy7Double( mx ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      real(8), pointer :: mp(:,:,:,:,:,:,:), mi(:,:,:,:,:,:,:)
      mwSize :: sz(7)
!-----
      mp => fpGetPr7(mx)
      if( .not.associated(mp) ) then
          nullify(fz)
          return
      endif
      sz(1) = size(mp,1)
      sz(2) = size(mp,2)
      sz(3) = size(mp,3)
      sz(4) = size(mp,4)
      sz(5) = size(mp,5)
      sz(6) = size(mp,6)
      sz(7) = size(mp,7)
      fz => fpAllocateZ7Double(sz(1),sz(2),sz(3),sz(4),                   &
     &                 sz(5),sz(6),sz(7))
      mi => fpGetPi7(mx)
      if( associated(mi) ) then
          call mxCopyReal88ToComplex16(mp, mi, fz, size(fz))
      else
          call mxCopyReal80ToComplex16(mp, fz, size(fz))
      endif
      return
      end function fpGetPzCopy7Double
!----------------------------------------------------------------------
      function fpReal1Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      complex(8), intent(in) :: z(:)
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
      mwSize :: N
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              N = size(z,1)
              call MatlabAPI_COM_Apx1( %VAL(zaddress), 2*stride, N )
              fp => Apx1
          endif
      endif
      return
      end function fpReal1Double
!----------------------------------------------------------------------
      function fpImag1Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      complex(8), intent(in) :: z(:)
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
      mwSize :: N
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              N = size(z,1)
              call MatlabAPI_COM_Apx1( %VAL(zaddress+8), 2*stride, N )
              fp => Apx1
          endif
      endif
      return
      end function fpImag1Double
!----------------------------------------------------------------------
      function fpReal2Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      complex(8), intent(in) :: z(:,:)
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx2( %VAL(zaddress), 2*stride,        &
     &                                 (/ size(z,1), size(z,2) /) )
              fp => Apx2
          endif
      endif
      return
      end function fpReal2Double
!----------------------------------------------------------------------
      function fpImag2Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      complex(8), intent(in) :: z(:,:)
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx2( %VAL(zaddress+8), 2*stride,      &
     &                                 (/ size(z,1), size(z,2) /) )
              fp => Apx2
          endif
      endif
      return
      end function fpImag2Double
!----------------------------------------------------------------------
      function fpReal3Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:)
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx3( %VAL(zaddress), 2*stride,        &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3) /) )
              fp => Apx3
          endif
      endif
      return
      end function fpReal3Double
!----------------------------------------------------------------------
      function fpImag3Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:)
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx3( %VAL(zaddress+8), 2*stride,      &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3) /) )
              fp => Apx3
          endif
      endif
      return
      end function fpImag3Double
!----------------------------------------------------------------------
      function fpReal4Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:)
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx4( %VAL(zaddress), 2*stride,        &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4) /) )
              fp => Apx4
          endif
      endif
      return
      end function fpReal4Double
!----------------------------------------------------------------------
      function fpImag4Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:)
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx4( %VAL(zaddress+8), 2*stride,      &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4) /) )
              fp => Apx4
          endif
      endif
      return
      end function fpImag4Double
!----------------------------------------------------------------------
      function fpReal5Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:,:)
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx5( %VAL(zaddress), 2*stride,        &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4),         &
     &                                    size(z,5) /) )
              fp => Apx5
          endif
      endif
      return
      end function fpReal5Double
!----------------------------------------------------------------------
      function fpImag5Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:,:)
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx5( %VAL(zaddress+8), 2*stride,      &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4),         &
     &                                    size(z,5) /) )
              fp => Apx5
          endif
      endif
      return
      end function fpImag5Double
!----------------------------------------------------------------------
      function fpReal6Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:,:,:)
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx6( %VAL(zaddress), 2*stride,        &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4),         &
     &                                    size(z,5), size(z,6) /) )
              fp => Apx6
          endif
      endif
      return
      end function fpReal6Double
!----------------------------------------------------------------------
      function fpImag6Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:,:,:)
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx6( %VAL(zaddress+8), 2*stride,      &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4),         &
     &                                    size(z,5), size(z,6) /) )
              fp => Apx6
          endif
      endif
      return
      end function fpImag6Double
!----------------------------------------------------------------------
      function fpReal7Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:,:,:,:)
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx7( %VAL(zaddress), 2*stride,        &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4),         &
     &                                    size(z,5), size(z,6),         &
     &                                    size(z,7) /) )
              fp => Apx7
          endif
      endif
      return
      end function fpReal7Double
!----------------------------------------------------------------------
      function fpImag7Double( z ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: z(:,:,:,:,:,:,:)
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: zaddress
      mwSize :: stride
!-----
      nullify(fp)
      zaddress = loc(z)
      if( zaddress /= 0 ) then
          stride = fpStride(z)
          if( stride /= 0 ) then
              call MatlabAPI_COM_Apx7( %VAL(zaddress+8), 2*stride,      &
     &                                 (/ size(z,1), size(z,2),         &
     &                                    size(z,3), size(z,4),         &
     &                                    size(z,5), size(z,6),         &
     &                                    size(z,7) /) )
              fp => Apx7
          endif
      endif
      return
      end function fpImag7Double
!----------------------------------------------------------------------
      function fpGetDimensions( mx ) result(fp)
      implicit none
      mwSize, pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwSize, pointer :: Dpx(:)
      common /MatlabAPI_COMD/ Dpx
!-LOC
      mwSize :: ndim
      mwPointer :: dims
!-----
      ndim = mxGetNumberOfDimensions( mx )
      dims = mxGetDimensions( mx )
      call MatlabAPI_COM_Dpx(%VAL(dims), ndim)
      fp => Dpx
      end function fpGetDimensions
!----------------------------------------------------------------------
      function fpGetIr( mx ) result(fp)
      implicit none
      mwIndex, pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwIndex, pointer :: Ipx(:)
      common /MatlabAPI_COMI/ Ipx
!-LOC
      mwPointer :: ir
      mwSize :: NZMAX
!-----
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 1 ) then
          ir = mxGetIr( mx )
          NZMAX = mxGetNzmax( mx )
          call MatlabAPI_COM_Ipx( %VAL(ir), NZMAX )
          fp => Ipx
      else
          nullify( fp )
      endif
      return
      end function fpGetIr
!----------------------------------------------------------------------
      function fpGetJc( mx ) result(fp)
      implicit none
      mwIndex, pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwIndex, pointer :: Ipx(:)
      common /MatlabAPI_COMI/ Ipx
!-LOC
      mwPointer :: jc
      mwSize :: N
!-----
      if( mxIsDouble(mx) == 1 .and. mxIsSparse(mx) == 1 ) then
          jc = mxGetJc( mx )
          N = mxGetN( mx )
          call MatlabAPI_COM_Ipx( %VAL(jc), N+1 )
          fp => Ipx
      else
          nullify( fp )
      endif
      return
      end function fpGetJc
!----------------------------------------------------------------------
      function fpGetCells1( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx1(:)
      common /MatlabAPI_COMP1/ Ppx1
!-LOC
      mwSize :: N
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          N = mxGetNumberOfElements( mx )
          if( N > 0 ) then
              cells = mxGetData( mx )
              call MatlabAPI_COM_Ppx1(%VAL(cells), N)
              fp => Ppx1
          endif
      endif
      end function fpGetCells1
!----------------------------------------------------------------------
      function fpGetCells2( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx2(:,:)
      common /MatlabAPI_COMP2/ Ppx2
!-LOC
      mwSize, pointer :: dims(:)
      mwSize :: ndim
      mwSize :: dimz(2)
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          if( mxGetNumberOfElements(mx) > 0 ) then
              dims => fpGetDimensions(mx)
              ndim = size(dims)
              if( ndim <= 2 ) then
                  dimz = 1
                  dimz(1:ndim) = dims
                  cells = mxGetData( mx )
                  call MatlabAPI_COM_Ppx2(%VAL(cells), dimz)
                  fp => Ppx2
              endif
          endif
      endif
      end function fpGetCells2
!----------------------------------------------------------------------
      function fpGetCells3( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx3(:,:,:)
      common /MatlabAPI_COMP3/ Ppx3
!-LOC
      mwSize, pointer :: dims(:)
      mwSize :: ndim
      mwSize :: dimz(3)
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          if( mxGetNumberOfElements(mx) > 0 ) then
              dims => fpGetDimensions(mx)
              ndim = size(dims)
              if( ndim <= 3 ) then
                  dimz = 1
                  dimz(1:ndim) = dims
                  cells = mxGetData( mx )
                  call MatlabAPI_COM_Ppx3(%VAL(cells), dimz)
                  fp => Ppx3
              endif
          endif
      endif
      end function fpGetCells3
!----------------------------------------------------------------------
      function fpGetCells4( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx4(:,:,:,:)
      common /MatlabAPI_COMP4/ Ppx4
!-LOC
      mwSize, pointer :: dims(:)
      mwSize :: ndim
      mwSize :: dimz(4)
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          if( mxGetNumberOfElements(mx) > 0 ) then
              dims => fpGetDimensions(mx)
              ndim = size(dims)
              if( ndim <= 4 ) then
                  dimz = 1
                  dimz(1:ndim) = dims
                  cells = mxGetData( mx )
                  call MatlabAPI_COM_Ppx4(%VAL(cells), dimz)
                  fp => Ppx4
              endif
          endif
      endif
      end function fpGetCells4
!----------------------------------------------------------------------
      function fpGetCells5( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx5(:,:,:,:,:)
      common /MatlabAPI_COMP5/ Ppx5
!-LOC
      mwSize, pointer :: dims(:)
      mwSize :: ndim
      mwSize :: dimz(5)
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          if( mxGetNumberOfElements(mx) > 0 ) then
              dims => fpGetDimensions(mx)
              ndim = size(dims)
              if( ndim <= 5 ) then
                  dimz = 1
                  dimz(1:ndim) = dims
                  cells = mxGetData( mx )
                  call MatlabAPI_COM_Ppx5(%VAL(cells), dimz)
                  fp => Ppx5
              endif
          endif
      endif
      end function fpGetCells5
!----------------------------------------------------------------------
      function fpGetCells6( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx6(:,:,:,:,:,:)
      common /MatlabAPI_COMP6/ Ppx6
!-LOC
      mwSize, pointer :: dims(:)
      mwSize :: ndim
      mwSize :: dimz(6)
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          if( mxGetNumberOfElements(mx) > 0 ) then
              dims => fpGetDimensions(mx)
              ndim = size(dims)
              if( ndim <= 6 ) then
                  dimz = 1
                  dimz(1:ndim) = dims
                  cells = mxGetData( mx )
                  call MatlabAPI_COM_Ppx6(%VAL(cells), dimz)
                  fp => Ppx6
              endif
          endif
      endif
      end function fpGetCells6
!----------------------------------------------------------------------
      function fpGetCells7( mx ) result(fp)
      implicit none
      mwPointer, pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      mwPointer, intent(in) :: mx
!-COM
      mwPointer, pointer :: Ppx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMP7/ Ppx7
!-LOC
      mwSize, pointer :: dims(:)
      mwSize :: ndim
      mwSize :: dimz(7)
      mwPointer :: cells
!-----
      nullify(fp)
      if( mxIsCell(mx) /= 0 ) then
          if( mxGetNumberOfElements(mx) > 0 ) then
              dims => fpGetDimensions(mx)
              ndim = size(dims)
              if( ndim <= 7 ) then
                  dimz = 1
                  dimz(1:ndim) = dims
                  cells = mxGetData( mx )
                  call MatlabAPI_COM_Ppx7(%VAL(cells), dimz)
                  fp => Ppx7
              endif
          endif
      endif
      end function fpGetCells7
      
      
!----------------------------------------------------------------------
      function fpAllocate0Double( ) result(fp)
      implicit none
      real(8), pointer :: fp
!-COM
      real(8), pointer :: Apx0
      common /MatlabAPI_COMA0/ Apx0
!-LOC
      mwPointer :: mxmemory
!-----
      mxmemory = mxMalloc(8)
      call MatlabAPI_COM_Apx0( %VAL(mxmemory) )
      fp => Apx0
      return
      end function fpAllocate0Double
!----------------------------------------------------------------------
      subroutine fpDeallocate0Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate0Double
!----------------------------------------------------------------------
      function fpAllocate1Double( n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n < 0 ) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n*8)
      call MatlabAPI_COM_Apx1( %VAL(mxmemory), stride, n )
      fp => Apx1
      return
      end function fpAllocate1Double
!----------------------------------------------------------------------
      subroutine fpDeallocate1Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate1Double
!----------------------------------------------------------------------
      function fpAllocate2Double( n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 ) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n1*n2*8)
      call MatlabAPI_COM_Apx2( %VAL(mxmemory), stride, (/n1,n2/) )
      fp => Apx2
      return
      end function fpAllocate2Double
!----------------------------------------------------------------------
      subroutine fpDeallocate2Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate2Double
!----------------------------------------------------------------------
      function fpAllocate3Double( n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 ) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*8)
      call MatlabAPI_COM_Apx3( %VAL(mxmemory), stride, (/n1,n2,n3/) )
      fp => Apx3
      return
      end function fpAllocate3Double
!----------------------------------------------------------------------
      subroutine fpDeallocate3Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate3Double
!----------------------------------------------------------------------
      function fpAllocate4Double( n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*8)
      call MatlabAPI_COM_Apx4( %VAL(mxmemory), stride, (/n1,n2,n3,n4/) )
      fp => Apx4
      return
      end function fpAllocate4Double
!----------------------------------------------------------------------
      subroutine fpDeallocate4Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate4Double
!----------------------------------------------------------------------
      function fpAllocate5Double( n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*n5*8)
      call MatlabAPI_COM_Apx5(%VAL(mxmemory),stride,(/n1,n2,n3,n4,n5/))
      fp => Apx5
      return
      end function fpAllocate5Double
!----------------------------------------------------------------------
      subroutine fpDeallocate5Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:,:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate5Double
!----------------------------------------------------------------------
      function fpAllocate6Double( n1, n2, n3, n4, n5, n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 ) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*n5*n6*8)
      call MatlabAPI_COM_Apx6( %VAL(mxmemory), stride,                  &
     &                        (/n1,n2,n3,n4,n5,n6/) )
      fp => Apx6
      return
      end function fpAllocate6Double
!----------------------------------------------------------------------
      subroutine fpDeallocate6Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:,:,:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate6Double
!----------------------------------------------------------------------
      function fpAllocate7Double( n1, n2, n3, n4, n5, n6, n7 )result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if(n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0) then
          nullify(fp)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*n5*n6*n7*8)
      call MatlabAPI_COM_Apx7( %VAL(mxmemory), stride,                  &
     &                        (/n1,n2,n3,n4,n5,n6,n7/))
      fp => Apx7
      return
      end function fpAllocate7Double
!----------------------------------------------------------------------
      subroutine fpDeallocate7Double( fp )
      implicit none
!-ARG
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fp) ) then
          mxmemory = loc(fp)
          call mxFree(mxmemory)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate7Double
!----------------------------------------------------------------------
      function fpAllocateZ0Double( ) result(fz)
      implicit none
      complex(8), pointer :: fz
!-COM
      complex(8), pointer :: Zpx0
      common /MatlabAPI_COMZ0/ Zpx0
!-LOC
      mwPointer :: mxmemory
!-----
      mxmemory = mxMalloc(16)
      call MatlabAPI_COM_Zpx0( %VAL(mxmemory) )
      fz => Zpx0
      return
      end function fpAllocateZ0Double
!----------------------------------------------------------------------
      subroutine fzDeallocate0Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate0Double
!----------------------------------------------------------------------
      function fpAllocateZ1Double( n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n<0 ) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n*16)
      call MatlabAPI_COM_Zpx1( %VAL(mxmemory), stride, n )
      fz => Zpx1
      return
      end function fpAllocateZ1Double
!----------------------------------------------------------------------
      subroutine fzDeallocate1Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate1Double
!----------------------------------------------------------------------
      function fpAllocateZ2Double( n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 ) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n1*n2*16)
      call MatlabAPI_COM_Zpx2( %VAL(mxmemory), stride, (/n1,n2/) )
      fz => Zpx2
      return
      end function fpAllocateZ2Double
!----------------------------------------------------------------------
      subroutine fzDeallocate2Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate2Double
!----------------------------------------------------------------------
      function fpAllocateZ3Double( n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 ) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*16)
      call MatlabAPI_COM_Zpx3( %VAL(mxmemory), stride, (/n1,n2,n3/) )
      fz => Zpx3
      return
      end function fpAllocateZ3Double
!----------------------------------------------------------------------
      subroutine fzDeallocate3Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate3Double
!----------------------------------------------------------------------
      function fpAllocateZ4Double( n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*16)
      call MatlabAPI_COM_Zpx4( %VAL(mxmemory), stride, (/n1,n2,n3,n4/) )
      fz => Zpx4
      return
      end function fpAllocateZ4Double
!----------------------------------------------------------------------
      subroutine fzDeallocate4Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate4Double
!----------------------------------------------------------------------
      function fpAllocateZ5Double( n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*n5*16)
      call MatlabAPI_COM_Zpx5(%VAL(mxmemory),stride,(/n1,n2,n3,n4,n5/))
      fz => Zpx5
      return
      end function fpAllocateZ5Double
!----------------------------------------------------------------------
      subroutine fzDeallocate5Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:,:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate5Double
!----------------------------------------------------------------------
      function fpAllocateZ6Double( n1, n2, n3, n4, n5, n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 ) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*n5*n6*16)
      call MatlabAPI_COM_Zpx6( %VAL(mxmemory), stride,                  &
     &                        (/n1,n2,n3,n4,n5,n6/) )
      fz => Zpx6
      return
      end function fpAllocateZ6Double
!----------------------------------------------------------------------
      subroutine fzDeallocate6Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate6Double
!----------------------------------------------------------------------
      function fpAllocateZ7Double(n1, n2, n3, n4, n5, n6, n7 )result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: mxmemory
      mwSize, parameter :: stride = 1
!-----
      if(n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0) then
          nullify(fz)
          return
      endif
      mxmemory = mxMalloc(n1*n2*n3*n4*n5*n6*n7*16)
      call MatlabAPI_COM_Zpx7( %VAL(mxmemory), stride,                  &
     &                        (/n1,n2,n3,n4,n5,n6,n7/))
      fz => Zpx7
      return
      end function fpAllocateZ7Double
!----------------------------------------------------------------------
      subroutine fzDeallocate7Double( fz )
      implicit none
!-ARG
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-LOC
      mwPointer :: mxmemory
!-----
      if( associated(fz) ) then
          mxmemory = loc(fz)
          call mxFree(mxmemory)
          nullify(fz)
      endif
      return
      end subroutine fzDeallocate7Double
!----------------------------------------------------------------------
      subroutine fpDeallocate1Character(fp)
      implicit none
!-ARG
      character(len=63), pointer :: fp(:)
!-LOC
      mwPointer ptr
!-----
      if( associated(fp) ) then
          ptr = loc(fp)
          call mxFree(ptr)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate1Character
!----------------------------------------------------------------------
      function fpReshape11Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), target, intent(in) :: ip(:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape11Double
!----------------------------------------------------------------------
      function fpReshape12Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape12Double
!----------------------------------------------------------------------
      function fpReshape13Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape13Double
!----------------------------------------------------------------------
      function fpReshape14Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape14Double
!----------------------------------------------------------------------
      function fpReshape15Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape15Double
!----------------------------------------------------------------------
      function fpReshape16Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape16Double
!----------------------------------------------------------------------
      function fpReshape17Double( ip, n ) result(fp)
      implicit none
      real(8), pointer :: fp(:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fp)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx1( %VAL(ipaddress), stride, n )
                  fp => Apx1
              endif
          endif
      endif
      return
      end function fpReshape17Double
!----------------------------------------------------------------------
      function fpReshape21Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape21Double
!----------------------------------------------------------------------
      function fpReshape22Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape22Double
!----------------------------------------------------------------------
      function fpReshape23Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape23Double
!----------------------------------------------------------------------
      function fpReshape24Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape24Double
!----------------------------------------------------------------------
      function fpReshape25Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape25Double
!----------------------------------------------------------------------
      function fpReshape26Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape26Double
!----------------------------------------------------------------------
      function fpReshape27Double( ip, n1, n2 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fp)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fp => Apx2
              endif
          endif
      endif
      return
      end function fpReshape27Double
!----------------------------------------------------------------------
      function fpReshape31Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape31Double
!----------------------------------------------------------------------
      function fpReshape32Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape32Double
!----------------------------------------------------------------------
      function fpReshape33Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape33Double
!----------------------------------------------------------------------
      function fpReshape34Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape34Double
!----------------------------------------------------------------------
      function fpReshape35Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape35Double
!----------------------------------------------------------------------
      function fpReshape36Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape36Double
!----------------------------------------------------------------------
      function fpReshape37Double( ip, n1, n2, n3 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fp => Apx3
              endif
          endif
      endif
      return
      end function fpReshape37Double
!----------------------------------------------------------------------
      function fpReshape41Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape41Double
!----------------------------------------------------------------------
      function fpReshape42Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape42Double
!----------------------------------------------------------------------
      function fpReshape43Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape43Double
!----------------------------------------------------------------------
      function fpReshape44Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape44Double
!----------------------------------------------------------------------
      function fpReshape45Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape45Double
!----------------------------------------------------------------------
      function fpReshape46Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape46Double
!----------------------------------------------------------------------
      function fpReshape47Double( ip, n1, n2, n3, n4 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fp => Apx4
              endif
          endif
      endif
      return
      end function fpReshape47Double
!----------------------------------------------------------------------
      function fpReshape51Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape51Double
!----------------------------------------------------------------------
      function fpReshape52Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape52Double
!----------------------------------------------------------------------
      function fpReshape53Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape53Double
!----------------------------------------------------------------------
      function fpReshape54Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape54Double
!----------------------------------------------------------------------
      function fpReshape55Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape55Double
!----------------------------------------------------------------------
      function fpReshape56Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape56Double
!----------------------------------------------------------------------
      function fpReshape57Double( ip, n1, n2, n3, n4, n5 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fp => Apx5
              endif
          endif
      endif
      return
      end function fpReshape57Double
!----------------------------------------------------------------------
      function fpReshape61Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape61Double
!----------------------------------------------------------------------
      function fpReshape62Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape62Double
!----------------------------------------------------------------------
      function fpReshape63Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape63Double
!----------------------------------------------------------------------
      function fpReshape64Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape64Double
!----------------------------------------------------------------------
      function fpReshape65Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape65Double
!----------------------------------------------------------------------
      function fpReshape66Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape66Double
!----------------------------------------------------------------------
      function fpReshape67Double( ip, n1,n2,n3,n4,n5,n6 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fp)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fp => Apx6
              endif
          endif
      endif
      return
      end function fpReshape67Double
!----------------------------------------------------------------------
      function fpReshape71Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape71Double
!----------------------------------------------------------------------
      function fpReshape72Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape72Double
!----------------------------------------------------------------------
      function fpReshape73Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape73Double
!----------------------------------------------------------------------
      function fpReshape74Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape74Double
!----------------------------------------------------------------------
      function fpReshape75Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape75Double
!----------------------------------------------------------------------
      function fpReshape76Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape76Double
!----------------------------------------------------------------------
      function fpReshape77Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fp)
      implicit none
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-ARG
      real(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fp)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Apx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fp => Apx7
              endif
          endif
      endif
      return
      end function fpReshape77Double
!----------------------------------------------------------------------
      function fzReshape11Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), target, intent(in) :: ip(:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape11Double
!----------------------------------------------------------------------
      function fzReshape12Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape12Double
!----------------------------------------------------------------------
      function fzReshape13Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape13Double
!----------------------------------------------------------------------
      function fzReshape14Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape14Double
!----------------------------------------------------------------------
      function fzReshape15Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape15Double
!----------------------------------------------------------------------
      function fzReshape16Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape16Double
!----------------------------------------------------------------------
      function fzReshape17Double( ip, n ) result(fz)
      implicit none
      complex(8), pointer :: fz(:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
!-----
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx1( %VAL(ipaddress), stride, n )
                  fz => Zpx1
              endif
          endif
      endif
      return
      end function fzReshape17Double
!----------------------------------------------------------------------
      function fzReshape21Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape21Double
!----------------------------------------------------------------------
      function fzReshape22Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape22Double
!----------------------------------------------------------------------
      function fzReshape23Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape23Double
!----------------------------------------------------------------------
      function fzReshape24Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape24Double
!----------------------------------------------------------------------
      function fzReshape25Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape25Double
!----------------------------------------------------------------------
      function fzReshape26Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape26Double
!----------------------------------------------------------------------
      function fzReshape27Double( ip, n1, n2 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2
      nullify(fz)
      if( n1<0 .or. n2<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx2( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2/) )
                  fz => Zpx2
              endif
          endif
      endif
      return
      end function fzReshape27Double
!----------------------------------------------------------------------
      function fzReshape31Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape31Double
!----------------------------------------------------------------------
      function fzReshape32Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape32Double
!----------------------------------------------------------------------
      function fzReshape33Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape33Double
!----------------------------------------------------------------------
      function fzReshape34Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape34Double
!----------------------------------------------------------------------
      function fzReshape35Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape35Double
!----------------------------------------------------------------------
      function fzReshape36Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape36Double
!----------------------------------------------------------------------
      function fzReshape37Double( ip, n1, n2, n3 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx3( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3/) )
                  fz => Zpx3
              endif
          endif
      endif
      return
      end function fzReshape37Double
!----------------------------------------------------------------------
      function fzReshape41Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape41Double
!----------------------------------------------------------------------
      function fzReshape42Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape42Double
!----------------------------------------------------------------------
      function fzReshape43Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape43Double
!----------------------------------------------------------------------
      function fzReshape44Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape44Double
!----------------------------------------------------------------------
      function fzReshape45Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape45Double
!----------------------------------------------------------------------
      function fzReshape46Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape46Double
!----------------------------------------------------------------------
      function fzReshape47Double( ip, n1, n2, n3, n4 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx4( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4/) )
                  fz => Zpx4
              endif
          endif
      endif
      return
      end function fzReshape47Double
!----------------------------------------------------------------------
      function fzReshape51Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape51Double
!----------------------------------------------------------------------
      function fzReshape52Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape52Double
!----------------------------------------------------------------------
      function fzReshape53Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape53Double
!----------------------------------------------------------------------
      function fzReshape54Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape54Double
!----------------------------------------------------------------------
      function fzReshape55Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape55Double
!----------------------------------------------------------------------
      function fzReshape56Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape56Double
!----------------------------------------------------------------------
      function fzReshape57Double( ip, n1, n2, n3, n4, n5 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 ) return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx5( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5/) )
                  fz => Zpx5
              endif
          endif
      endif
      return
      end function fzReshape57Double
!----------------------------------------------------------------------
      function fzReshape61Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress

      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape61Double
!----------------------------------------------------------------------
      function fzReshape62Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape62Double
!----------------------------------------------------------------------
      function fzReshape63Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape63Double
!----------------------------------------------------------------------
      function fzReshape64Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape64Double
!----------------------------------------------------------------------
      function fzReshape65Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape65Double
!----------------------------------------------------------------------
      function fzReshape66Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape66Double
!----------------------------------------------------------------------
      function fzReshape67Double( ip, n1,n2,n3,n4,n5,n6 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6
      nullify(fz)
      if( n1<0 .or. n2<0 .or. n3<0 .or. n4<0 .or. n5<0 .or. n6<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx6( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6/) )
                  fz => Zpx6
              endif
          endif
      endif
      return
      end function fzReshape67Double
!----------------------------------------------------------------------
      function fzReshape71Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape71Double
!----------------------------------------------------------------------
      function fzReshape72Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape72Double
!----------------------------------------------------------------------
      function fzReshape73Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape73Double
!----------------------------------------------------------------------
      function fzReshape74Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape74Double
!----------------------------------------------------------------------
      function fzReshape75Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape75Double
!----------------------------------------------------------------------
      function fzReshape76Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape76Double
!----------------------------------------------------------------------
      function fzReshape77Double( ip, n1,n2,n3,n4,n5,n6,n7 ) result(fz)
      implicit none
      complex(8), pointer :: fz(:,:,:,:,:,:,:)
!-ARG
      complex(8), intent(in) :: ip(:,:,:,:,:,:,:)
      mwSize, intent(in) :: n1, n2, n3, n4, n5, n6, n7
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-LOC
      mwPointer :: ipaddress
      mwSize :: stride
      mwSize :: n
!-----
      n = n1 * n2 * n3 * n4 * n5 * n6 * n7
      nullify(fz)
      if( n1<0.or.n2<0.or.n3<0.or.n4<0.or.n5<0.or.n6<0.or.n7<0 )return
      ipaddress = loc(ip)
      if( ipaddress /= 0 ) then
          if( size(ip) == n ) then
              stride = fpStride(ip)
              if( stride /= 0 ) then
                  call MatlabAPI_COM_Zpx7( %VAL(ipaddress), stride,     &
     &                                    (/n1,n2,n3,n4,n5,n6,n7/) )
                  fz => Zpx7
              endif
          endif
      endif
      return
      end function fzReshape77Double
      
!-------------------------------------------------------------------------------

      function uppercase(string) result(upper)
      character(len=*), intent(in) :: string
      character(len=len(string)) :: upper
      integer :: j
      do j = 1,len(string)
          if(string(j:j) >= "a" .and. string(j:j) <= "z") then
              upper(j:j) = achar(iachar(string(j:j)) - 32)
          else
              upper(j:j) = string(j:j)
          end if
      enddo
      end function uppercase
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader0double(A, B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A
      real(8), optional, intent(in) :: B
!-LOC
      mwPointer address
      mwSize, parameter :: N = 1, Zero = 0
!-----
      mx = 0
      if( present(B) ) then
          mx = mxCreateDoubleMatrix(Zero, Zero, mxCOMPLEX)
      else
          mx = mxCreateDoubleMatrix(Zero, Zero, mxREAL)
      endif
      if( mx == 0 ) return
      call mxSetM(mx, N)
      call mxSetN(mx, N)
      address = loc(A)
      call mxSetPr(mx, address)
      if( present(B)) then
          address = loc(B)
          call mxSetPi(mx, address)
      endif
      return
      end function mxArrayHeader0double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader1double(A, B, orient) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:)
      real(8), optional, intent(in) :: B(:)
      character(len=*), optional, intent(in) :: orient
!-LOC
      character(len=6) rowcol
      mwPointer address
      mwSize M, N
      mwSize :: Zero = 0
!-----
      mx = 0
      if( present(orient) ) then
            rowcol = uppercase(orient)
            if( rowcol == 'ROW' ) then
                M = 1
                N = size(A)
            elseif( rowcol == 'COLUMN' ) then
                M = size(A)
                N = 1
            else
                return
            endif
      else
            M = size(A)
            N = 1
      endif
      if( present(B) ) then
          if( size(A) == size(B) ) then
              mx = mxCreateDoubleMatrix(Zero, Zero, mxCOMPLEX)
          else
              return
          endif
      else
          mx = mxCreateDoubleMatrix(Zero, Zero, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      call mxSetM(mx, M)
      call mxSetN(mx, N)
      address = loc(A)
      call mxSetPr(mx, address)
      if( present(B)) then
          address = loc(B)
          call mxSetPi(mx, address)
      endif
      return
      end function mxArrayHeader1double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader2double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:)
      real(8), optional, intent(in) :: B(:,:)
!-LOC
      mwPointer address
      mwSize M, N
      mwSize :: Zero = 0
!-----
      if( present(B) ) then
          if( size(A,1) /= size(B,1) .or. size(A,2) /= size(B,1) ) then
              mx = 0
              return
          endif
          mx = mxCreateDoubleMatrix(Zero, Zero, mxCOMPLEX)
      else
          mx = mxCreateDoubleMatrix(Zero, Zero, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      M = size(A,1)
      N = size(A,2)
      call mxSetM(mx, M)
      call mxSetN(mx, N)
      address = loc(A)
      call mxSetPr(mx, address)
      if( present(B)) then
          address = loc(B)
          call mxSetPi(mx, address)
      endif
      return
      end function mxArrayHeader2double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader3double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:)
      real(8), optional, intent(in) :: B(:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 3
      mwSize dims(ndim)
      integer*4 i
!-----
      dims = 0
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( mxSetDimensions(mx, dims, ndim) /= 0 ) then
          call mxDestroyArray(mx)
          mx = 0
          return
      endif
      address = loc(A)
      call mxSetPr(mx, address )
      if( present(B) ) then
          address = loc(B)
          call mxSetPi(mx, address )
      endif
      return
      end function mxArrayHeader3double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader4double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 4
      mwSize dims(ndim)
      integer*4 i
!-----
      dims = 0
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( mxSetDimensions(mx, dims, ndim) /= 0 ) then
          call mxDestroyArray(mx)
          mx = 0
          return
      endif
      address = loc(A)
      call mxSetPr(mx, address )
      if( present(B) ) then
          address = loc(B)
          call mxSetPi(mx, address )
      endif
      return
      end function mxArrayHeader4double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader5double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 5
      mwSize dims(ndim)
      integer*4 i
!-----
      dims = 0
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( mxSetDimensions(mx, dims, ndim) /= 0 ) then
          call mxDestroyArray(mx)
          mx = 0
          return
      endif
      address = loc(A)
      call mxSetPr(mx, address )
      if( present(B) ) then
          address = loc(B)
          call mxSetPi(mx, address )
      endif
      return
      end function mxArrayHeader5double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader6double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 6
      mwSize dims(ndim)
      integer*4 i
!-----
      dims = 0
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( mxSetDimensions(mx, dims, ndim) /= 0 ) then
          call mxDestroyArray(mx)
          mx = 0
          return
      endif
      address = loc(A)
      call mxSetPr(mx, address )
      if( present(B) ) then
          address = loc(B)
          call mxSetPi(mx, address )
      endif
      return
      end function mxArrayHeader6double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArrayHeader7double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 7
      mwSize dims(ndim)
      integer*4 i
!-----
      dims = 0
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( mxSetDimensions(mx, dims, ndim) /= 0 ) then
          call mxDestroyArray(mx)
          mx = 0
          return
      endif
      address = loc(A)
      call mxSetPr(mx, address )
      if( present(B) ) then
          address = loc(B)
          call mxSetPi(mx, address )
      endif
      return
      end function mxArrayHeader7double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray0double(A, B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A
      real(8), optional, intent(in) :: B
!-LOC
      mwPointer address
      real(8), pointer :: fp
      mwSize, parameter :: N = 1
!-----
      mx = 0
      if( present(B) ) then
          mx = mxCreateDoubleMatrix(N, N, mxCOMPLEX)
      else
          mx = mxCreateDoubleMatrix(N, N, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr0(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi0(mx)
          fp = B
      endif
      return
      end function mxArray0double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray1double(A, B, orient) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:)
      real(8), optional, intent(in) :: B(:)
      character(len=*), optional, intent(in) :: orient
!-LOC
      character(len=6) rowcol
      mwPointer address
      real(8), pointer :: fp(:)
      mwSize M, N
!-----
      mx = 0
      if( present(orient) ) then
            rowcol = uppercase(orient)
            if( rowcol == 'ROW' ) then
                M = 1
                N = size(A)
            elseif( rowcol == 'COLUMN' ) then
                M = size(A)
                N = 1
            else
                return
            endif
      else
            M = size(A)
            N = 1
      endif
      if( present(B) ) then
          if( size(A) == size(B) ) then
              mx = mxCreateDoubleMatrix(M, N, mxCOMPLEX)
          else
              return
          endif
      else
          mx = mxCreateDoubleMatrix(M, N, mxREAL)
      endif
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr1(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi1(mx)
          fp = B
      endif
      return
      end function mxArray1double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray2double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:)
      real(8), optional, intent(in) :: B(:,:)
!-LOC
      mwPointer address
      mwSize M, N
      real(8), pointer :: fp(:,:)
!-----
      M = size(A,1)
      N = size(A,2)
      if( present(B) ) then
          if( size(B,1) /= M .or. size(B,2) /= N ) then
              mx = 0
              return
          endif
          mx = mxCreateDoubleMatrix(M, N, mxCOMPLEX)
      else
          mx = mxCreateDoubleMatrix(M, N, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr2(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi2(mx)
          fp = B
      endif
      return
      end function mxArray2double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray3double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:)
      real(8), optional, intent(in) :: B(:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 3
      mwSize dims(ndim)
      integer*4 i
      real(8), pointer :: fp(:,:,:)
!-----
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr3(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi3(mx)
          fp = B
      endif
      return
      end function mxArray3double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray4double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 4
      mwSize dims(ndim)
      integer*4 i
      real(8), pointer :: fp(:,:,:,:)
!-----
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr4(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi4(mx)
          fp = B
      endif
      return
      end function mxArray4double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray5double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 5
      mwSize dims(ndim)
      integer*4 i
      real(8), pointer :: fp(:,:,:,:,:)
!-----
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr5(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi5(mx)
          fp = B
      endif
      return
      end function mxArray5double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray6double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 6
      mwSize dims(ndim)
      integer*4 i
      real(8), pointer :: fp(:,:,:,:,:,:)
!-----
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr6(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi6(mx)
          fp = B
      endif
      return
      end function mxArray6double
      
!-------------------------------------------------------------------------------

      mwPointer function mxArray7double(A,B) result(mx)
      implicit none
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:,:,:)
      real(8), optional, intent(in) :: B(:,:,:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 7
      mwSize dims(ndim)
      integer*4 i
      real(8), pointer :: fp(:,:,:,:,:,:,:)
!-----
      do i=1,ndim
          dims(i) = size(A,i)
      enddo
      if( present(B) ) then
          do i=1,ndim
              if( size(A,i) /= size(B,i) ) then
                  mx = 0
                  return
              endif
          enddo
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxCOMPLEX)
      else
          mx = mxCreateNumericArray                                     &
     &        (ndim, dims, mxDOUBLE_CLASS, mxREAL)
      endif
      if( mx == 0 ) return
      fp => fpGetPr7(mx)
      fp = A
      if( present(B) ) then
          fp => fpGetPi7(mx)
          fp = B
      endif
      return
      end function mxArray7double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray0double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A
!-LOC
      mwPointer address
      mwSize, parameter :: N = 1
      real(8), pointer :: fp, fi
!-----
      mx = mxCreateDoubleMatrix(N, N, mxCOMPLEX)
      if( mx == 0 ) return
      fp => fpGetPr0(mx)
      fi => fpGetPi0(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, N)
      return
      end function mzArray0double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray1double(A, orient) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:)
      character(len=*), optional, intent(in) :: orient
!-LOC
      character(len=6) rowcol
      mwPointer address
      mwSize M, N
      real(8), pointer :: fp(:), fi(:)
!-----
      if( present(orient) ) then
            rowcol = uppercase(orient)
            if( rowcol == 'ROW' ) then
                M = 1
                N = size(A)
            elseif( rowcol == 'COLUMN' ) then
                M = size(A)
                N = 1
            else
                mx = 0
                return
            endif
      else
            M = size(A)
            N = 1
      endif
      mx = mxCreateDoubleMatrix(M, N, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr1(mx)
      fi => fpGetPi1(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, M*N)
      return
      end function mzArray1double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray2double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:,:)
!-LOC
      mwPointer address
      mwSize M, N
      real(8), pointer :: fp(:,:), fi(:,:)
!-----
      M = size(A,1)
      N = size(A,2)
      mx = mxCreateDoubleMatrix(M, N, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr2(mx)
      fi => fpGetPi2(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, M*N)
      return
      end function mzArray2double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray3double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 3
      mwSize dims(ndim)
      integer*4 i
      mwSize N
      real(8), pointer :: fp(:,:,:), fi(:,:,:)
!-----
      N = 1
      do i=1,ndim
          dims(i) = size(A,i)
          N = N * dims(i)
      enddo
      mx = mxCreateNumericArray(ndim, dims,                             &
     &                          mxDOUBLE_CLASS, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr3(mx)
      fi => fpGetPi3(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, N)
      return
      end function mzArray3double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray4double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 4
      mwSize dims(ndim)
      integer*4 i
      mwSize N
      real(8), pointer :: fp(:,:,:,:), fi(:,:,:,:)
!-----
      N = 1
      do i=1,ndim
          dims(i) = size(A,i)
          N = N * dims(i)
      enddo
      mx = mxCreateNumericArray(ndim, dims,                             &
     &                          mxDOUBLE_CLASS, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr4(mx)
      fi => fpGetPi4(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, N)
      return
      end function mzArray4double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray5double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 5
      mwSize dims(ndim)
      integer*4 i
      mwSize N
      real(8), pointer :: fp(:,:,:,:,:), fi(:,:,:,:,:)
!-----
      N = 1
      do i=1,ndim
          dims(i) = size(A,i)
          N = N * dims(i)
      enddo
      mx = mxCreateNumericArray(ndim, dims,                             &
     &                          mxDOUBLE_CLASS, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr5(mx)
      fi => fpGetPi5(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, N)
      return
      end function mzArray5double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray6double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:,:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 6
      mwSize dims(ndim)
      integer*4 i
      mwSize N
      real(8), pointer :: fp(:,:,:,:,:,:), fi(:,:,:,:,:,:)
!-----
      N = 1
      do i=1,ndim
          dims(i) = size(A,i)
          N = N * dims(i)
      enddo
      mx = mxCreateNumericArray(ndim, dims,                             &
     &                          mxDOUBLE_CLASS, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr6(mx)
      fi => fpGetPi6(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, N)
      return
      end function mzArray6double
      
!-------------------------------------------------------------------------------

      mwPointer function mzArray7double(A) result(mx)
      implicit none
!-ARG
      complex(8), intent(in) :: A(:,:,:,:,:,:,:)
!-LOC
      mwPointer address
      mwSize, parameter :: ndim = 7
      mwSize dims(ndim)
      integer*4 i
      mwSize N
      real(8), pointer :: fp(:,:,:,:,:,:,:), fi(:,:,:,:,:,:,:)
!-----
      N = 1
      do i=1,ndim
          dims(i) = size(A,i)
          N = N * dims(i)
      enddo
      mx = mxCreateNumericArray(ndim, dims,                             &
     &                          mxDOUBLE_CLASS, mxCOMPLEX)
      if( mx == 0 .or. size(A) == 0 ) return
      fp => fpGetPr7(mx)
      fi => fpGetPi7(mx)
      call mxCopyComplex16ToReal88(A, fp, fi, N)
      return
      end function mzArray7double
      
!-------------------------------------------------------------------------------

      subroutine mxDestroyArrayHeader(mx)
      implicit none
!-ARG
      mwPointer, intent(in) :: mx
!-LOC
      mwPointer, parameter :: p = 0
!-----
      call mxSetPr(mx, p)
      call mxSetPi(mx, p)
      call mxDestroyArray(mx)
      return
      end subroutine mxDestroyArrayHeader
      
!-------------------------------------------------------------------------------

      function fpStride1Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:)
!-----
      if( size(A) > 1 ) then
          stride = (loc(A(2)) - loc(A(1))) / 8
      else
          stride = 1
      endif
      return
      end function fpStride1Double
      
!-------------------------------------------------------------------------------

      function fpStride2Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:,:)
!-LOC
      integer, parameter :: ndim = 2
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2)))                                    &
     &           - loc(A(q(1),q(2)))) / 8
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fpStride2Double
      
!-------------------------------------------------------------------------------

      function fpStride3Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:,:,:)
!-LOC
      integer, parameter :: ndim = 3
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3)))                               &
     &           - loc(A(q(1),q(2),q(3)))) / 8
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fpStride3Double
      
!-------------------------------------------------------------------------------

      function fpStride4Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:,:,:,:)
!-LOC
      integer, parameter :: ndim = 4
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4)))                          &
     &           - loc(A(q(1),q(2),q(3),q(4)))) / 8
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fpStride4Double
      
!-------------------------------------------------------------------------------

      function fpStride5Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:)
!-LOC
      integer, parameter :: ndim = 5
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4),p(5)))                     &
     &           - loc(A(q(1),q(2),q(3),q(4),q(5)))) / 8
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fpStride5Double
      
!-------------------------------------------------------------------------------

      function fpStride6Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:,:)
!-LOC
      integer, parameter :: ndim = 6
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4),p(5),p(6)))                &
     &           - loc(A(q(1),q(2),q(3),q(4),q(5),q(6)))) / 8
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fpStride6Double
      
!-------------------------------------------------------------------------------

      function fpStride7Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      real(8), intent(in) :: A(:,:,:,:,:,:,:)
!-LOC
      integer, parameter :: ndim = 7
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4),p(5),p(6),p(7)))           &
     &           - loc(A(q(1),q(2),q(3),q(4),q(5),q(6),q(7)))) / 8
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fpStride7Double

!-------------------------------------------------------------------------------

      function fzStride1Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:)
!-----
      if( size(A) > 1 ) then
          stride = (loc(A(2)) - loc(A(1))) / 16
      else
          stride = 1
      endif
      return
      end function fzStride1Double
      
!-------------------------------------------------------------------------------

      function fzStride2Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:,:)
!-LOC
      integer, parameter :: ndim = 2
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2)))                                    &
     &           - loc(A(q(1),q(2)))) / 16
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fzStride2Double
      
!-------------------------------------------------------------------------------

      function fzStride3Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:,:,:)
!-LOC
      integer, parameter :: ndim = 3
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3)))                               &
     &           - loc(A(q(1),q(2),q(3)))) / 16
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fzStride3Double
      
!-------------------------------------------------------------------------------

      function fzStride4Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:,:,:,:)
!-LOC
      integer, parameter :: ndim = 4
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4)))                          &
     &           - loc(A(q(1),q(2),q(3),q(4)))) / 16
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fzStride4Double
      
!-------------------------------------------------------------------------------

      function fzStride5Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:,:,:,:,:)
!-LOC
      integer, parameter :: ndim = 5
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4),p(5)))                     &
     &           - loc(A(q(1),q(2),q(3),q(4),q(5)))) / 16
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fzStride5Double
      
!-------------------------------------------------------------------------------

      function fzStride6Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:,:,:,:,:,:)
!-LOC
      integer, parameter :: ndim = 6
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4),p(5),p(6)))                &
     &           - loc(A(q(1),q(2),q(3),q(4),q(5),q(6)))) / 16
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fzStride6Double
      
!-------------------------------------------------------------------------------

      function fzStride7Double(A) result(stride)
      implicit none
      mwSize :: stride
!-ARG
      complex(8), intent(in) :: A(:,:,:,:,:,:,:)
!-LOC
      integer, parameter :: ndim = 7
      mwSize i, p(ndim), q(ndim), s
!-----
      p = 1
      q = 1
      stride = 0
      do i=1,ndim
          if( size(A) > 1 ) then
              p(i) = 2
              s = (loc(A(p(1),p(2),p(3),p(4),p(5),p(6),p(7)))           &
     &           - loc(A(q(1),q(2),q(3),q(4),q(5),q(6),q(7)))) / 16
              if( stride == 0 ) stride = s
              if( s /= stride ) then
                  stride = 0
                  return
              endif
              p(i) = 1
         endif
         q(i) = size(A,i)
      enddo
      if( stride == 0 ) stride = 1
      return
      end function fzStride7Double

!-------------------------------------------------------------------------------

      subroutine random_number0double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z
!-LOC
      mwSize s
!-----
      s = 2
      call random_numberz(z,s)
      return
      end subroutine random_number0double

!-------------------------------------------------------------------------------

      subroutine random_number1double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number1double

!-------------------------------------------------------------------------------

      subroutine random_number2double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:,:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number2double

!-------------------------------------------------------------------------------

      subroutine random_number3double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:,:,:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number3double

!-------------------------------------------------------------------------------

      subroutine random_number4double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:,:,:,:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number4double

!-------------------------------------------------------------------------------

      subroutine random_number5double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:,:,:,:,:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number5double

!-------------------------------------------------------------------------------

      subroutine random_number6double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:,:,:,:,:,:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number6double

!-------------------------------------------------------------------------------

      subroutine random_number7double(z)
      implicit none
!-ARG
      complex(8), intent(out) :: z(:,:,:,:,:,:,:)
!-LOC
      mwSize s
!-----
      s = 2 * size(z)
      call random_numberz(z,s)
      return
      end subroutine random_number7double

!-------------------------------------------------------------------------------

      end module MatlabAPImx

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!\
! Routines below here have implicit interfaces. This is necessary because they
! are called using the %VAL() construct.
!/
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

      subroutine mxCopyC1toI2( C1, I2, k )
      implicit none
!-ARG
      mwSize, intent(in) :: k
      character(len=*), intent(in) :: C1 ! Input, the Fortran character data stored as 1-byte elements
      integer*2, intent(out) :: I2(k)    ! Output, the mxArray char data stored as 2-byte elements
!-LOC
      mwSize i
!-----
      do i=1,k
          I2(i) = iachar(C1(i:i))
      enddo
      return
      end subroutine mxCopyC1toI2

!-------------------------------------------------------------------------------

      subroutine mxCopyI2toC1( I2, C1, k )
      implicit none
!-ARG
      mwSize, intent(in) :: k
      integer*2, intent(in) :: I2(k)      ! Input, the mxArray char data stored as 2-byte elements
      character(len=*), intent(out) :: C1 ! Output, the Fortran character data stored as 1-byte elements
!-LOC
      mwSize i
!-----
      do i=1,k
          C1(i:i) = achar(I2(i))
      enddo
      return
      end subroutine mxCopyI2toC1

!-------------------------------------------------------------------------------
#ifndef NOCOMPLEX16

      subroutine mxCopyComplex32ToReal1616(y, xr, xi, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(16), intent(in) :: y(2*n)
      real(16), intent(out) :: xr(n), xi(n)
!-LOC
      mwSize i
!-----
      do i=1,n
          xr(i) = y(2*i-1)
          xi(i) = y(2*i  )
      enddo
      return
      end subroutine mxCopyComplex32ToReal1616

#endif
!-------------------------------------------------------------------------------
#ifndef NOCOMPLEX16

      subroutine mxCopyReal1616ToComplex32(xr, xi, y, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(16), intent(in) :: xr(n), xi(n)
      real(16), intent(out) :: y(2*n)
!-LOC
      mwSize i
!-----
      do i=1,n
          y(2*i-1) = xr(i)
          y(2*i)   = xi(i)
      enddo
      return
      end subroutine mxCopyReal1616ToComplex32

#endif
!-------------------------------------------------------------------------------
#ifndef NOREAL16

      subroutine mxCopyReal16ToReal16(x, y, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(16), intent(in) :: x(n)
      real(16), intent(out) :: y(n)
!-LOC
      mwSize i
!-----
      y = x
      return
      end subroutine mxCopyReal16ToReal16

#endif
!-------------------------------------------------------------------------------
#ifndef NOINTEGER8

      subroutine mxCopyInteger8ToInteger8(y, x, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      integer(8), intent(in) :: y(n)
      integer(8), intent(out) :: x(n)
!-----
      x = y
      return
      end subroutine mxCopyInteger8ToInteger8

#endif

!------------------------------------------------------------------------------

      subroutine mxCopyReal88ToComplex16(xr, xi, y, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(8), intent(in) :: xr(n), xi(n)
      real(8), intent(out) :: y(2*n)
!-LOC
      mwSize i
!-----
      do i=1,n
          y(2*i-1) = xr(i)
          y(2*i)   = xi(i)
      enddo
      return
      end subroutine mxCopyReal88ToComplex16

!------------------------------------------------------------------------------

      subroutine mxCopyReal80ToComplex16(xr, y, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(8), intent(in) :: xr(n)
      real(8), intent(out) :: y(2*n)
!-LOC
      mwSize i
!-----
      do i=1,n
          y(2*i-1) = xr(i)
          y(2*i)   = 0.d0
      enddo
      return
      end subroutine mxCopyReal80ToComplex16

!------------------------------------------------------------------------------

      subroutine mxCopyComplex16ToReal88(y, xr, xi, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      real(8), intent(in) :: y(2*n)
      real(8), intent(out) :: xr(n), xi(n)
!-LOC
      mwSize i
!-----
      do i=1,n
          xr(i) = y(2*i-1)
          xi(i) = y(2*i)
      enddo
      return
      end subroutine mxCopyComplex16ToReal88
      
!------------------------------------------------------------------------------

      subroutine mxCopyInteger1ToLogical( logicaldata, fortran, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      integer*1, intent(in) :: logicaldata(n)
      logical, intent(out) :: fortran(n)
!-----
      fortran = (logicaldata /= 0)
      return
      end subroutine mxCopyInteger1ToLogical


!------------------------------------------------------------------------------

      subroutine mxCopyLogicalToInteger1( fortran, logicaldata, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical, intent(in) :: fortran(n)
      integer*1, intent(out) :: logicaldata(n)
!-LOC
      integer k
!-----
      logicaldata = 0
      forall( k = 1:n, fortran(k) ) logicaldata(k) = 1
      return
      end subroutine mxCopyLogicalToInteger1

!------------------------------------------------------------------------------

      logical function mxDataToLogical( integer1value )
      implicit none
!-ARG
      integer*1, intent(in) :: integer1value
!-----
      mxDataToLogical = (integer1value /= 0)
      return
      end function mxDataToLogical

!------------------------------------------------------------------------------

      logical function mxLSDataToLogical( integer1value )
      implicit none
!-ARG
      integer*1, intent(in) :: integer1value
!-----
      mxLSDataToLogical = (integer1value /= 0)
      return
      end function mxLSDataToLogical

!------------------------------------------------------------------------------

      subroutine mxCopyInteger1ToLogical1( logicaldata, fortran, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      integer*1, intent(in) :: logicaldata(n)
      logical*1, intent(out) :: fortran(n)
!-----
      fortran = (logicaldata /= 0)
      return
      end subroutine mxCopyInteger1ToLogical1

!------------------------------------------------------------------------------

      subroutine mxCopyLogical1ToInteger1( fortran, logicaldata, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical*1, intent(in) :: fortran(n)
      integer*1, intent(out) :: logicaldata(n)
!-LOC
      integer k
!-----
      logicaldata = 0
      forall( k = 1:n, fortran(k) ) logicaldata(k) = 1
      return
      end subroutine mxCopyLogical1ToInteger1

!------------------------------------------------------------------------------

      logical*1 function mxDataToLogical1( integer1value )
      implicit none
!-ARG
      integer*1, intent(in) :: integer1value
!-----
      mxDataToLogical1 = (integer1value /= 0)
      return
      end function mxDataToLogical1

!------------------------------------------------------------------------------

      subroutine mxCopyInteger1ToLogical2( logicaldata, fortran, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      integer*1, intent(in) :: logicaldata(n)
      logical*2, intent(out) :: fortran(n)
!-----
      fortran = (logicaldata /= 0)
      return
      end subroutine mxCopyInteger1ToLogical2

!------------------------------------------------------------------------------

      subroutine mxCopyLogical2ToInteger1( fortran, logicaldata, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical*2, intent(in) :: fortran(n)
      integer*1, intent(out) :: logicaldata(n)
!-LOC
      integer k
!-----
      logicaldata = 0
      forall( k = 1:n, fortran(k) ) logicaldata(k) = 1
      return
      end subroutine mxCopyLogical2ToInteger1

!------------------------------------------------------------------------------

      logical*2 function mxDataToLogical2( integer1value )
      implicit none
!-ARG

      integer*1, intent(in) :: integer1value
!-----
      mxDataToLogical2 = (integer1value /= 0)
      return
      end function mxDataToLogical2


!------------------------------------------------------------------------------

      subroutine mxCopyInteger1ToLogical4( logicaldata, fortran, n)
      implicit none
!-ARG
      mwSize, intent(in) :: n
      integer*1, intent(in) :: logicaldata(n)
      logical*4, intent(out) :: fortran(n)
!-----
      fortran = (logicaldata /= 0)
      return
      end subroutine mxCopyInteger1ToLogical4

!------------------------------------------------------------------------------

      subroutine mxCopyLogical4ToInteger1( fortran, logicaldata, n )
      implicit none
!-ARG
      mwSize, intent(in) :: n
      logical*4, intent(in) :: fortran(n)
      integer*1, intent(out) :: logicaldata(n)
!-LOC
      integer k
!-----
      logicaldata = 0
      forall( k = 1:n, fortran(k) ) logicaldata(k) = 1
      return
      end subroutine mxCopyLogical4ToInteger1

!------------------------------------------------------------------------------

      logical*4 function mxDataToLogical4( integer1value )
      implicit none
!-ARG
      integer*1, intent(in) :: integer1value
!-----
      mxDataToLogical4 = (integer1value /= 0)
      return
      end function mxDataToLogical4
      
!----------------------------------------------------------------------
! Specific Fortan Pointer Helper functions. Not contained in the module
! becausewe need an implicit interface to get the %VAL() construct to
! work properly in the calling routine. Passing the appropriate pointer
! back in a COMMON block. Looks awkward, but works beautifully.
!----------------------------------------------------------------------

      subroutine MatlabAPI_COM_Apx0( A )
      implicit none
!-ARG
      real(8), target, intent(in) :: A
!-COM
      real(8), pointer :: Apx0
      common /MatlabAPI_COMA0/ Apx0
!-----
      Apx0 => A
      return
      end subroutine MatlabAPI_COM_Apx0
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Apx1( A, stride, N )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, N
      real(8), target, intent(in) :: A(stride, N)
!-COM
      real(8), pointer :: Apx1(:)
      common /MatlabAPI_COMA1/ Apx1
!-----
      Apx1 => A(1,:)
      return
      end subroutine MatlabAPI_COM_Apx1

      subroutine MatlabAPI_COM_SingleApx1( A, stride, N )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, N
      real(4), target, intent(in) :: A(stride, N)
!-COM
      real(4), pointer :: Apx1(:)
      common /MatlabAPI_COMSingleA1/ Apx1
!-----
      Apx1 => A(1,:)
      return
      end subroutine MatlabAPI_COM_SingleApx1







!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Apx2( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(2)
      real(8), target, intent(in) :: A(stride,DIMS(1),DIMS(2))
!-COM
      real(8), pointer :: Apx2(:,:)
      common /MatlabAPI_COMA2/ Apx2
!-----
      Apx2 => A(1,:,:)
      return
      end subroutine MatlabAPI_COM_Apx2


      subroutine MatlabAPI_COM_SingleApx2( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(2)
      real(4), target, intent(in) :: A(stride,DIMS(1),DIMS(2))
!-COM
      real(4), pointer :: Apx2(:,:)
      common /MatlabAPI_COMSingleA2/ Apx2
!-----
      Apx2 => A(1,:,:)
      return
      end subroutine MatlabAPI_COM_SingleApx2





!----------------------------------------------------------------------     

   
      subroutine MatlabAPI_COM_Apx3( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(3)
      real(8), target, intent(in) :: A(stride,DIMS(1),DIMS(2),DIMS(3))
!-COM
      real(8), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMA3/ Apx3
!-----
      Apx3 => A(1,:,:,:)
      return
      end subroutine MatlabAPI_COM_Apx3

 
      subroutine MatlabAPI_COM_SingleApx3( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(3)
      real(4), target, intent(in) :: A(stride,DIMS(1),DIMS(2),DIMS(3))
!-COM
      real(4), pointer :: Apx3(:,:,:)
      common /MatlabAPI_COMSingleA3/ Apx3
!-----
      Apx3 => A(1,:,:,:)
      return
      end subroutine MatlabAPI_COM_SingleApx3







!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Apx4( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(4)
      real(8), target, intent(in) :: A(stride,DIMS(1),DIMS(2),DIMS(3),  &
     &                                 DIMS(4))
!-COM
      real(8), pointer :: Apx4(:,:,:,:)
      common /MatlabAPI_COMA4/ Apx4
!-----
      Apx4 => A(1,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Apx4
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Apx5( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(5)
      real(8), target, intent(in) :: A(stride,DIMS(1),DIMS(2),DIMS(3),  &
     &                                 DIMS(4),DIMS(5))
!-COM
      real(8), pointer :: Apx5(:,:,:,:,:)
      common /MatlabAPI_COMA5/ Apx5
!-----
      Apx5 => A(1,:,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Apx5
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Apx6( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(6)
      real(8), target, intent(in) :: A(stride,DIMS(1),DIMS(2),DIMS(3),  &
     &                                 DIMS(4),DIMS(5),DIMS(6))
!-COM
      real(8), pointer :: Apx6(:,:,:,:,:,:)
      common /MatlabAPI_COMA6/ Apx6
!-----
      Apx6 => A(1,:,:,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Apx6
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Apx7( A, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(7)
      real(8), target, intent(in) :: A(stride*DIMS(1),DIMS(2),DIMS(3),  &
     &                                 DIMS(4),DIMS(5),DIMS(6),DIMS(7))
!-COM
      real(8), pointer :: Apx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMA7/ Apx7
!-----
      Apx7 => A(::stride,:,:,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Apx7
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx0( Z )
      implicit none
!-ARG
      complex(8), target, intent(in) :: Z
!-COM
      complex(8), pointer :: Zpx0
      common /MatlabAPI_COMZ0/ Zpx0
!-----
      Zpx0 => Z
      return
      end subroutine MatlabAPI_COM_Zpx0
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx1( Z, stride, N )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, N
      complex(8), target, intent(in) :: Z(stride,N)
!-COM
      complex(8), pointer :: Zpx1(:)
      common /MatlabAPI_COMZ1/ Zpx1
!-----
      Zpx1 => Z(1,:)
      return
      end subroutine MatlabAPI_COM_Zpx1
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx2( Z, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(2)
      complex(8), target, intent(in) :: Z(stride,DIMS(1),DIMS(2))
!-COM
      complex(8), pointer :: Zpx2(:,:)
      common /MatlabAPI_COMZ2/ Zpx2
!-----
      Zpx2 => Z(1,:,:)
      return
      end subroutine MatlabAPI_COM_Zpx2
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx3( Z, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(3)
      complex(8), target, intent(in) ::Z(stride,DIMS(1),DIMS(2),DIMS(3))
!-COM
      complex(8), pointer :: Zpx3(:,:,:)
      common /MatlabAPI_COMZ3/ Zpx3
!-----
      Zpx3 => Z(1,:,:,:)
      return
      end subroutine MatlabAPI_COM_Zpx3
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx4( Z, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(4)
      complex(8), target, intent(in) ::Z(stride,DIMS(1),DIMS(2),DIMS(3) &
     &                                  ,DIMS(4))
!-COM
      complex(8), pointer :: Zpx4(:,:,:,:)
      common /MatlabAPI_COMZ4/ Zpx4
!-----
      Zpx4 => Z(1,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Zpx4
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx5( Z, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(5)
      complex(8), target, intent(in) ::Z(stride,DIMS(1),DIMS(2),DIMS(3) &
     &                                  ,DIMS(4),DIMS(5))
!-COM
      complex(8), pointer :: Zpx5(:,:,:,:,:)
      common /MatlabAPI_COMZ5/ Zpx5
!-----
      Zpx5 => Z(1,:,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Zpx5
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx6( Z, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(6)
      complex(8), target, intent(in) ::Z(stride,DIMS(1),DIMS(2),DIMS(3) &
     &                                  ,DIMS(4),DIMS(5),DIMS(6))
!-COM
      complex(8), pointer :: Zpx6(:,:,:,:,:,:)
      common /MatlabAPI_COMZ6/ Zpx6
!-----
      Zpx6 => Z(1,:,:,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Zpx6
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Zpx7( Z, stride, DIMS )
      implicit none
!-ARG
      mwSize, intent(in) :: stride, DIMS(7)
      complex(8), target, intent(in) ::Z(stride*DIMS(1),DIMS(2),DIMS(3) &
     &                                 ,DIMS(4),DIMS(5),DIMS(6),DIMS(7))
!-COM
      complex(8), pointer :: Zpx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMZ7/ Zpx7
!-----
      Zpx7 => Z(::stride,:,:,:,:,:,:)
      return
      end subroutine MatlabAPI_COM_Zpx7
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Dpx(dims, ndim)
      implicit none
!-ARG
      mwSize, intent(in) :: ndim
      mwSize, target, intent(in) :: dims(ndim)
!-COM
      mwSize, pointer :: Dpx(:)
      common /MatlabAPI_COMD/ Dpx
!-----
      Dpx => dims
      return
      end subroutine MatlabAPI_COM_Dpx
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ipx(ir, nzmax)
      implicit none
!-ARG
      mwSize, intent(in) :: nzmax
      mwIndex, target, intent(in) :: ir(nzmax)
!-COM
      mwIndex, pointer :: Ipx(:)
      common /MatlabAPI_COMI/ Ipx
!-----
      Ipx => ir
      return
      end subroutine MatlabAPI_COM_Ipx
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx1(cells, N)
      implicit none
!-ARG
      mwSize, intent(in) :: N
      mwPointer, target, intent(in) :: cells(N)
!-COM
      mwPointer, pointer :: Ppx1(:)
      common /MatlabAPI_COMP1/ Ppx1
!-----
      Ppx1 => cells
      return
      end subroutine MatlabAPI_COM_Ppx1
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx2(cells, DIMS)
      implicit none
!-ARG
      mwSize, intent(in) :: DIMS(2)
      mwPointer, target, intent(in) :: cells(DIMS(1),DIMS(2))
!-COM
      mwPointer, pointer :: Ppx2(:,:)
      common /MatlabAPI_COMP2/ Ppx2
!-----
      Ppx2 => cells
      return
      end subroutine MatlabAPI_COM_Ppx2
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx3(cells, DIMS)
      implicit none
!-ARG
      mwSize, intent(in) :: DIMS(3)
      mwPointer, target, intent(in) :: cells(DIMS(1),DIMS(2),DIMS(3))
!-COM
      mwPointer, pointer :: Ppx3(:,:,:)
      common /MatlabAPI_COMP3/ Ppx3
!-----
      Ppx3 => cells
      return
      end subroutine MatlabAPI_COM_Ppx3
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx4(cells, DIMS)
      implicit none
!-ARG
      mwSize, intent(in) :: DIMS(4)
      mwPointer, target, intent(in) :: cells(DIMS(1),DIMS(2),DIMS(3),   &
     &                                       DIMS(4))
!-COM
      mwPointer, pointer :: Ppx4(:,:,:,:)
      common /MatlabAPI_COMP4/ Ppx4
!-----
      Ppx4 => cells
      return
      end subroutine MatlabAPI_COM_Ppx4
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx5(cells, DIMS)
      implicit none
!-ARG
      mwSize, intent(in) :: DIMS(5)
      mwPointer, target, intent(in) :: cells(DIMS(1),DIMS(2),DIMS(3),   &
     &                                       DIMS(4),DIMS(5))
!-COM
      mwPointer, pointer :: Ppx5(:,:,:,:,:)
      common /MatlabAPI_COMP5/ Ppx5
!-----
      Ppx5 => cells
      return
      end subroutine MatlabAPI_COM_Ppx5
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx6(cells, DIMS)
      implicit none
!-ARG
      mwSize, intent(in) :: DIMS(6)
      mwPointer, target, intent(in) :: cells(DIMS(1),DIMS(2),DIMS(3),   &
     &                                       DIMS(4),DIMS(5),DIMS(6))
!-COM
      mwPointer, pointer :: Ppx6(:,:,:,:,:,:)
      common /MatlabAPI_COMP6/ Ppx6
!-----
      Ppx6 => cells
      return
      end subroutine MatlabAPI_COM_Ppx6
!----------------------------------------------------------------------      
      subroutine MatlabAPI_COM_Ppx7(cells, DIMS)
      implicit none
!-ARG
      mwSize, intent(in) :: DIMS(7)
      mwPointer, target, intent(in) :: cells(DIMS(1),DIMS(2),DIMS(3),   &
     &                                       DIMS(4),DIMS(5),DIMS(6),   &
     &                                       DIMS(7))
!-COM
      mwPointer, pointer :: Ppx7(:,:,:,:,:,:,:)
      common /MatlabAPI_COMP7/ Ppx7
!-----
      Ppx7 => cells
      return
      end subroutine MatlabAPI_COM_Ppx7
!----------------------------------------------------------------------      
      subroutine random_numberz(z,s)
      implicit none
!-ARG
      mwSize, intent(in) :: s
      real(8), intent(out) :: z(s)
!-----
      call random_number(z)
      return
      end subroutine random_numberz
