!_______________________________________________________________________
!
! MODULE mod_piv3d
!
!   Contains subroutines used within libpiv3d which we don't 
!   necessarily want to be exposed in the library
!_______________________________________________________________________

MODULE mod_piv3d

  IMPLICIT NONE

  ! There are no shared arrays within this module - as I've no idea 
  ! what the scoping rules are (and no documentation lists it) when
  ! using OMP within a subroutine contained in a module. 
  ! All variables are passed around like it's 1977 still... but we 
  ! keep our subroutines here so that we have explicit interfaces, 
  ! allowing more errors to be caught in the compilation stages 
  ! (useful for code development)


CONTAINS


  !_______________________________________________________________________
  !
  !	  SUBROUTINE WINDOWCOORDSPAR
  !	  Computes coordinates of window elements in list form, derived from 
  !     corner coordinates of each window (simple trilinear interpolation
  !     to determine coordinates of mapped voxel positions in real voxels 
  !     array space).
  !
  !     SUBROUTINE WHITCARDINTERP5, WHITCARDINTERP7
  !     Applies the Whittaker Cardinal Function to deform and interpolate
  !     the input intensity arrays (A and B) to regular window arrays. 
  !     Functions available for 5^3 (faster) or 7^3 (more accurate)
  !     cardinal interpolation kernels.
  include 'sub_whitCardInterp.F90'


  !_______________________________________________________________________
  !
  !	  SUBROUTINE INDEXEDFETCH
  !	  Fetches regular cube windows directly from the field arrays. Much
  !     quicker than cardinal interpolation for cases where no window
  !     deformation is required
  include 'sub_indexedFetch.F90'

  !_______________________________________________________________________
  !
  !	  SUBROUTINE TRILINTERP
  !	  Interpolates windows from the field arrays using linear 
  !     interpolation. Much quicker but also much less accurate than 
  !     cardinal interpolation
  include 'sub_triLInterp.F90'

  !_______________________________________________________________________
  !
  !     SUBROUTINE SETUPCROSSCORR
  !     Creates descriptors, sets up lengths and scaling values, commits 
  !     FFTs (i.e. computes twiddle factors) in preparation for looped 
  !     invocation of forward and inverse FFTS. This removes the 
  !     costly allocation and setup from the (1,NWINDOWS) loop.
  !       - NOT CURRENTLY IMPLEMENTED (couldn't make it work!!)
  !_______________________________________________________________________

  !     SUBROUTINE CROSSCORRPAR
  !     Performs FFTs based cross correlation on Windows A and B. 
  !     To be executed within the (1,NWINDOWS) loop i.e. within a 
  !     thread of a parallel construct.

  !     SUBROUTINE CROSSCORRPARCTE
  !     Performs FFTs based CTE cross correlation on Windows A, B, C and D.
  !     To be executed within the (1,NWINDOWS) loop i.e. within a 
  !     thread of a parallel construct.
  include 'sub_crossCorr.F90'


  !_______________________________________________________________________
  !
  !     SUBROUTINE GETSMOOTHINGKERNEL
  !     Gets a 3x3x3 smoothing kernel for smoothing the correlation plane
  !	  include 'fMexPIV_mod_GetSmoothingKernel.F90'


  !_______________________________________________________________________
  !
  !     SUBROUTINE GETWEIGHT
  !     Gets weighting array for debiasing the correlation plane
  include 'sub_getWeight.F90'


  !_______________________________________________________________________
  !
  !     SUBROUTINE FREEALL
  !     Frees FFT descriptors, frees memory buffer allocated by Intel MKL 
  !     and Deallocates all allocated arrays shared in the module. 
  !     Useful before exit and in case of error.
  !
  !	  SUBROUTINE HANDLEFFTERROR
  !     Handles errors in the FFT modules and exits safely to MATLAB 
  !	  include 'fMexPIV_mod_MemoryHandling.F90'



  !_______________________________________________________________________
  !
  !                        END MODULE
  !
END MODULE mod_piv3d
!_______________________________________________________________________
