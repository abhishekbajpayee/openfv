!*************************************************************************************
! 
!  MATLAB (R) is a trademark of The Mathworks (R) Corporation
! 
!  Filename:    MatlabAPImex.f
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
!  Interfaces included (see MATLAB doc for use):
!
!     mexCallMATLAB
!     mexCallMATLABWithTrap
!     mexErrMsgIdAndTxt
!     mexErrMsgTxt
!     mexEvalString
!     mexEvalStringWithTrap
!     mexFunctionName
!     mexGetVariable
!     mexGetVariablePtr
!     mexIsGlobal
!     mexIsLocked
!     mexLock
!     mexMakeArrayPersistent
!     mexMakeMemoryPersistent
!     mexPrintf
!     mexPutVariable
!     mexSetTrapFlag
!     mexUnlock
!     mexWarnMsgIdAndTxt
!     mexWarnMsgTxt
!
!  Routines included (see MATLAB doc for use):
!
!     mexCreateSparseLogicalMatrix (works same as mxCreateSparseLogicalMatrix)
!     mexGet
!     mexPrint (same as mexPrintf except trims and appends a newline at end)
!     mexSet
!
!  Routine included using Fortran Poiners (see below for use)
!
!     fpMexGetNames (gets all variable names from the workplace)
!
!  Change Log:
!  2009/Oct/27 --> Initial Release
!  2009/Dec/11 --> Changed default address function to LOC instead of %LOC
!
!*************************************************************************************

#ifndef mwPointer
#define mwPointer integer(4)
#endif

#ifndef mwSize
#define mwSize integer(4)
#endif

#ifndef mwIndex
#define mwIndex integer(4)
#endif

#ifdef PERCENTLOC
#define loc %LOC
#endif

      module MatlabAPIcharMex
      
      integer, parameter :: NameLengthMaxMex = 63
      
      contains

!----------------------------------------------------------------------
      function fpAllocate1CharacterMex(n) result(fp)
      implicit none
      character(len=NameLengthMaxMex), pointer :: fp(:)
!-ARG
      mwSize, intent(in) :: n
!-FUN
      mwPointer, external :: mxMalloc
!-COM
      character(len=NameLengthMaxMex), pointer :: Cpx1(:)
      common /MatlabAPI_COMCmex/ Cpx1
!-LOC
      mwPointer ptr
      mwSize, parameter :: NLMM = NameLengthMaxMex
!-----
      ptr = mxMalloc(NameLengthMaxMex * n)
      call MatlabAPI_COM_CpxMex(n, %val(ptr), %val(NLMM))
      fp => Cpx1
      return
      end function fpAllocate1CharacterMex
!----------------------------------------------------------------------
      subroutine fpDeallocate1CharacterMex(fp)
      implicit none
!-ARG
      character(len=NameLengthMaxMex), pointer :: fp(:)
!-LOC
      mwPointer ptr
!-----
      if( associated(fp) ) then
          ptr = loc(fp)
          call mxFree(ptr)
          nullify(fp)
      endif
      return
      end subroutine fpDeallocate1CharacterMex

!----------------------------------------------------------------------

      end module MatlabAPIcharMex

!----------------------------------------------------------------------

      subroutine MatlabAPI_COM_CpxMex(n, C)
      implicit none
!-PAR
      integer, parameter :: NameLengthMaxMex = 63
!-ARG
      mwSize, intent(in) :: n
      character(len=NameLengthMaxMex), target :: C(n)
!-COM
      character(len=NameLengthMaxMex), pointer :: Cpx1(:)
      common /MatlabAPI_COMCmex/ Cpx1
!-----
      Cpx1 => C
      return
      end subroutine MatlabAPI_COM_CpxMex

!----------------------------------------------------------------------
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      module MatlabAPImex
      
      use MatlabAPIcharMex
      
!-----------------------------------------------------      
! Interface definitions for MATLAB mex API functions
!-----------------------------------------------------
      
      interface

      integer(4) function mexCallMATLAB(nlhs, plhs, nrhs, prhs,         &
     &                                  functionName)
      integer(4), intent(in) :: nlhs, nrhs
      mwPointer, intent(out) :: plhs(*)
      mwPointer, intent(in) :: prhs(*)
      character(len=*), intent(in) :: functionName
      end function mexCallMATLAB
!-----
      mwPointer function mexCallMATLABWithTrap(nlhs, plhs, nrhs, prhs,  &
     &                                         functionName)
      integer(4), intent(in) :: nlhs, nrhs
      mwPointer, intent(out) :: plhs(*)
      mwPointer, intent(in) :: prhs(*)
      character(len=*), intent(in) :: functionName
      end 
!-----
      subroutine mexErrMsgIdAndTxt(errorid, errormsg)
      character(len=*), intent(in) :: errorid, errormsg
      end subroutine mexErrMsgIdAndTxt
!-----
      subroutine mexErrMsgTxt(errormsg)
      character(len=*), intent(in) :: errormsg
      end subroutine mexErrMsgTxt
!-----
      integer(4) function mexEvalString(command)
      character(len=*), intent(in) :: command
      end function mexEvalString
!-----
      mwPointer function mexEvalStringWithTrap(command)
      character(len=*), intent(in) :: command
      end function mexEvalStringWithTrap
!-----
      character*63 function mexFunctionName() ! Altered
      end function mexFunctionName
!-----
!      mwPointer function mexGet(handle, property)  ! New code below
!      real(8), intent(in) :: handle
!      character(len=*), intent(in) :: property
!      end function mexGet
!-----
      mwPointer function mexGetVariable(workspace, varname)
      character(len=*), intent(in) :: workspace, varname
      end function mexGetVariable
!-----
      mwPointer function mexGetVariablePtr(workspace, varname)
      character(len=*), intent(in) :: workspace, varname
      end function mexGetVariablePtr
!-----
      integer(4) function mexIsGlobal(pm)
      mwPointer, intent(in) :: pm
      end function mexIsGlobal
!-----
      integer(4) function mexIsLocked()
      end function mexIsLocked
!-----
      subroutine mexLock
      end subroutine mexLock
!-----
      subroutine mexMakeArrayPersistent(pm)
      mwPointer, intent(in) :: pm
      end subroutine mexMakeArrayPersistent
!-----
      subroutine mexMakeMemoryPersistent(ptr)
      mwPointer, intent(in) :: ptr
      end subroutine mexMakeMemoryPersistent
!-----
!      integer(4) function mexPrintf(message)
!      character(len=*), intent(in) :: message
!      end function mexPrintf
!-----
      integer(4) function mexPutVariable(workspace, varname, pm)
      character(len=*), intent(in) :: workspace, varname
      mwPointer, intent(in) :: pm
      end function mexPutVariable
!-----
!      integer(4) function mexSet(handle, property, value)  ! New code below
!      real(8), intent(in) :: handle
!      character(len=*), intent(in) :: property
!      mwPointer, intent(in) :: value
!      end function mexSet
!-----
      subroutine mexSetTrapFlag(trapflag)
      integer(4), intent(in) :: trapflag
      end subroutine mexSetTrapFlag
!-----
      subroutine mexUnlock
      end subroutine mexUnlock
!-----
      subroutine mexWarnMsgIdAndTxt(warningid, warningmsg)
      character(len=*), intent(in) :: warningid, warningmsg
      end subroutine mexWarnMsgIdAndTxt
!-----
      subroutine mexWarnMsgTxt(warningmsg)
      character(len=*), intent(in) :: warningmsg
      end subroutine mexWarnMsgTxt

      end interface
      
      contains

!-----------------------------------------------------      
! Definitions for extra functions
!-----------------------------------------------------

!------------------------------------------------------------------------------
!
! Function:    mexCreateSparseLogicalMatrix
!
! Arguments
!
!  m
!
!    The desired number of rows
!
!  n
!
!    The desired number of columns
!
!  nzmax
!
!    The number of elements that mexCreateSparseLogicalMatrix should allocate to
!    hold the data. Set the value of nzmax to be greater than or equal to the
!    number of nonzero elements you plan to put into the mxArray, but make sure
!    that nzmax is less than or equal to m*n.
!
!  Returns
!
!    A pointer to the created mxArray, if successful. If unsuccessful in a
!    MEX-file, the MEX-file terminates and control returns to the MATLAB
!    prompt. mexCreateSparseLogicalMatrix is unsuccessful when there is not
!    enough free heap space to create the mxArray.
!
!  Description
!
!    Use mexCreateSparseLogicalMatrix to create an m-by-n mxArray of mxLogical
!    elements. mexCreateSparseLogicalMatrix initializes each element in the array
!    to logical 0.
!
!  Call mxDestroyArray when you finish using the mxArray. mxDestroyArray
!  deallocates the mxArray and its elements.
!
!******************************************************************************

!-----------------------------------------------------------------------------

      mwPointer function mexCreateSparseLogicalMatrix( m, n, nzmax )
      implicit none
!-ARG
      mwSize, intent(in) :: m, n, nzmax
!-PAR
      mwSize, parameter :: zero = 0
      mwSize, parameter :: sizeoflogical = 1   ! logicals in MATLAB are 1 byte
      integer(4), parameter :: ComplexFlag = 0  ! real
      integer(4), parameter :: nlhs = 1
      integer(4), parameter :: nrhs = 1
!-FUN
      mwPointer, external :: mxCalloc
      mwPointer, external :: mxCreateDoubleMatrix
      mwPointer, external :: mxCreateDoubleScalar
      mwPointer, external :: mxCreateString
      mwPointer, external :: mxGetData
      mwPointer, external :: mxGetIr
      mwPointer, external :: mxGetJc
      mwPointer, external :: mxMalloc
!-LOC
      integer(4) k
      mwPointer plhs(1), prhs(1)  ! really mxArray*
      mwPointer pr                ! really mxLogical*
      mwPointer jc, ir            ! really mwIndex*
      mwIndex mwx(2)              ! used for sizeof calculation
      mwSize sizeofindex
!-----
!\
! Default return value is failure
!/
      mexCreateSparseLogicalMatrix = 0
!\
! Get size of the MATLAB type mwIndex
!/
      sizeofindex = loc(mwx(2)) - loc(mwx(1))
!\
! First create an empty full double matrix
!/
      prhs(1) = mxCreateDoubleMatrix( zero, zero, ComplexFlag )
!\
! Now turn it into an empty sparse double matrix
!/
      k = mexCallMATLAB( nlhs, plhs, nrhs, prhs, "sparse" )
      call mxDestroyArray( prhs(1) )
      if( k /= 0 ) return
!\
! Then turn it into an empty sparse logical matrix
!/
      prhs(1) = plhs(1)
      k = mexCallMATLAB( nlhs, plhs, nrhs, prhs, "logical" )
      call mxDestroyArray( prhs(1) )
      if( k /= 0 ) return
!\
! Allocate new memory for data and indexes based on nzmax.
!/
      pr = mxMalloc( nzmax * sizeoflogical )  ! Not initialized
      ir = mxMalloc( nzmax * sizeofindex )    ! Not initialized
      jc = mxCalloc( n+1, sizeofindex )       ! Initialized to all zero
!\
! Free the current data and index memory
!/
      call mxFree( mxGetData( plhs(1) ) )
      call mxFree( mxGetIr( plhs(1) ) )
      call mxFree( mxGetJc( plhs(1) ) )
!\
! Reset the pointers to the new memory
!/
      call mxSetM( plhs(1), m )
      call mxSetN( plhs(1), n )
      call mxSetData( plhs(1), pr )
      call mxSetIr( plhs(1), ir )
      call mxSetJc( plhs(1), jc )
      call mxSetNzmax( plhs(1), nzmax )
!\
! Everything went ok, so set return value
!/
      mexCreateSparseLogicalMatrix = plhs(1)

      return
      end function

!----------------------------------------------------------------------
! Print a trimmed character string with a newline at the end. The
! returned function value does not count the newline character.
!----------------------------------------------------------------------
!      integer(4) function mexPrint(message)
!      implicit none
!!-ARG
!      character(len=*), intent(in) :: message
!!-LOC
!      integer(4) k
!!-----
!      mexPrint = mexPrintf(message)
!      k = mexPrintf(achar(10))
!      return
!      end function mexPrint
      
!----------------------------------------------------------------------
 
      mwPointer function mexGet(handle, property)
      implicit none
!-ARG
      real(8), intent(in) :: handle
      character(len=*), intent(in) :: property
!-FUN
      mwPointer, external :: mxCreateDoubleScalar
      mwPointer, external :: mxCreateString
!-LOC
      mwPointer mxproperty
      mwPointer mx(2), my(1)
      integer(4) trapflag
!-----
      mx(1) = mxCreateDoubleScalar(handle)
      mx(2) = mxCreateString(property)
      call mexSetTrapFlag(1)
      trapflag = mexCallMATLAB(1, my, 2, mx, "get")
      call mxDestroyArray(mx(2))
      call mxDestroyArray(mx(1))
      if( trapflag == 0 ) then
          mexGet = my(1)
      else
          mexGet = 0
      endif
      return
      end function mexGet
      
!----------------------------------------------------------------------
 
      integer(4) function mexSet(handle, property, value)
      implicit none
!-ARG
      real(8), intent(in) :: handle
      character(len=*), intent(in) :: property
      mwPointer, intent(in) :: value
!-FUN
      mwPointer, external :: mxCreateDoubleScalar
      mwPointer, external :: mxCreateString
!-LOC
      mwPointer mx(3)
      mwPointer answer(1)
!-----
      mx(1) = mxCreateDoubleScalar(handle)
      mx(2) = mxCreateString(property)
      mx(3) = value
      call mexSetTrapFlag(1)
      mexSet = mexCallMATLAB(0, answer, 3, mx, "set")
      call mxDestroyArray(mx(2))
      call mxDestroyArray(mx(1))
      return
      end function mexSet
      
!----------------------------------------------------------------------

      function fpMexGetNames( ) result(fp)
      implicit none
      character(len=NameLengthMaxMex), pointer :: fp(:)
!-FUN
      mwPointer, external :: mxGetField
      mwSize, external :: mxGetNumberOfElements
      integer*4, external :: mxGetString
!-LOC
      mwPointer rhs(1), lhs(1)
      mwPointer mx
      integer(4) k
      mwIndex i, n
!-----
      nullify(fp)
      k = mexCallMATLAB(1, lhs, 0, rhs, "whos") ! Get list of variables
      if( k == 0 ) then
          n = mxGetNumberOfElements(lhs(1)) ! Get number of variables
          if( n /= 0 ) then
              fp => fpAllocate1CharacterMex(n) ! Allocate space for array
              if( associated(fp) ) then
                  fp = ' '
                  do i=1,n
                      mx = mxGetField(lhs(1), i, "name") ! Get the name
                      k = mxGetString(mx, fp(i), NameLengthMaxMex) ! Copy into our array
                  enddo
              endif
          endif
          call mxDestroyArray(lhs(1)) ! Free the result of the whos call
      endif
      return
      end function fpMexGetNames
      
!----------------------------------------------------------------------
 
      end module MatlabAPImex
