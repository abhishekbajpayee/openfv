!_______________________________________________________________________
!
!	SUBROUTINE GETWEIGHT
!
!	Computes the weighting array used to debias a correlation peak.
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs:
!
!       WSIZE           [1 x 1] integer *1
!                               dimension of the correlation cube array
!   
!   Outputs:
!       WEIGHT          [WSIZE x WSIZE x WSIZE] real*4
!                               Debiasing array for multiplication with
!                               the cross correlation plane                            
!
! Author:               T.H. Clark
! Work address:         Fluids Laboratory
!                       Cambridge University Engineering Department
!                       2 Trumpington Street
!                       Cambridge
!                       CB21PZ
! Email:                t.clark@cantab.net
! Website:              http://www2.eng.cam.ac.uk/~thc29
!
! Revision History:     02 July 2011        Created
!                       04 July 2011        Improved documentation
!                       17 September 2011   Updated interface for
!                                           libpiv3d
!                       June 2015           Reformatted and edited to
!                                           compile without MKL and ifort

SUBROUTINE getWeight(WSIZE, WEIGHT)


  !     Safer to have no implicit variables
  IMPLICIT NONE


  !	  Input
  INTEGER*4, INTENT(in) :: WSIZE

  !     Output
  REAL*4, DIMENSION(WSIZE,WSIZE,WSIZE), INTENT(OUT) :: WEIGHT

  !     Internal admin variables for computing the array
  REAL*4 :: wdisttocen, wdistfromcen
  INTEGER*4 :: ii, jj, kk, n, m, p



  !     Use Nick's code to calculate the weighting matrix
  m = WSIZE
  n = WSIZE
  p = WSIZE
  wdisttocen = (    	(((REAL(m)+1.)/2.)**2.)	+                   &
       (((REAL(n)+1.)/2.)**2.)	+                   &
       (((REAL(p)+1.)/2.)**2.) )**0.5 
  DO ii=1,m
     DO jj=1,n
        DO kk=1,p
           wdistfromcen = (((REAL(ii)-((REAL(m)+1.)/2.))**2.)  +  &
                ((REAL(jj)-((REAL(n)+1.)/2.))**2.)  +  &
                ((REAL(kk)-((REAL(p)+1.)/2.))**2.))**0.5
           WEIGHT(ii,jj,kk) = 1 - (wdistfromcen/wdisttocen)
           IF (WEIGHT(ii,jj,kk).LE.0) THEN
              WEIGHT(ii,jj,kk) = 0.01
           ENDIF
        ENDDO
     ENDDO
  ENDDO


  !	  End of subroutine GetWeight
END SUBROUTINE getWeight

