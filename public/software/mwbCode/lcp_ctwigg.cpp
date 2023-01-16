// LCP solver, originally based on the one in ODE but heavily modified
// by Christopher D. Twigg.  Original code Copyright Russell L. Smith;
// modifications copyright Christopher D.  Twigg.  Released under the 
// same BSD license as the rest of ODE.  

/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001,2002 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of EITHER:                                  *
 *   (1) The GNU Lesser General Public License as published by the Free  *
 *       Software Foundation; either version 2.1 of the License, or (at  *
 *       your option) any later version. The text of the GNU Lesser      *
 *       General Public License is included with this library in the     *
 *       file LICENSE.TXT.                                               *
 *   (2) The BSD-style license that is included with this library in     *
 *       the file LICENSE-BSD.TXT.                                       *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

/*


THE ALGORITHM
-------------

solve A*x = b+w, with x and w subject to certain LCP conditions.
each x(i),w(i) must lie on one of the three line segments in the following
diagram. each line segment corresponds to one index set :

     w(i)
     /|\      |           :
      |       |           :
      |       |i in N     :
  w>0 |       |state[i]=0 :
      |       |           :
      |       |           :  i in C
  w=0 +       +-----------------------+
      |                   :           |
      |                   :           |
  w<0 |                   :           |i in N
      |                   :           |state[i]=1
      |                   :           |
      |                   :           |
      +-------|-----------|-----------|----------> x(i)
             lo           0           hi

the Dantzig algorithm proceeds as follows:
  for i=1:n
    * if (x(i),w(i)) is not on the line, push x(i) and w(i) positive or
      negative towards the line. as this is done, the other (x(j),w(j))
      for j<i are constrained to be on the line. if any (x,w) reaches the
      end of a line segment then it is switched between index sets.
    * i is added to the appropriate index set depending on what line segment
      it hits.

we restrict lo(i) <= 0 and hi(i) >= 0. this makes the algorithm a bit
simpler, because the starting point for x(i),w(i) is always on the dotted
line x=0 and x will only ever increase in one direction, so it can only hit
two out of the three line segments.


NOTES
-----

this is an implementation of "lcp_dantzig2_ldlt.m" and "lcp_dantzig_lohi.m".
the implementation is split into an LCP problem object (dLCP) and an LCP
driver function. most optimization occurs in the dLCP object.

a naive implementation of the algorithm requires either a lot of data motion
or a lot of permutation-array lookup, because we are constantly re-ordering
rows and columns. to avoid this and make a more optimized algorithm, a
non-trivial data structure is used to represent the matrix A (this is
implemented in the fast version of the dLCP object).

during execution of this algorithm, some indexes in A are clamped (set C),
some are non-clamped (set N), and some are "don't care" (where x=0).
A,x,b,w (and other problem vectors) are permuted such that the clamped
indexes are first, the unclamped indexes are next, and the don't-care
indexes are last. this permutation is recorded in the array `p'.
initially p = 0..n-1, and as the rows and columns of A,x,b,w are swapped,
the corresponding elements of p are swapped.

because the C and N elements are grouped together in the rows of A, we can do
lots of work with a fast dot product function. if A,x,etc were not permuted
and we only had a permutation array, then those dot products would be much
slower as we would have a permutation array lookup in some inner loops.

A is accessed through an array of row pointers, so that element (i,j) of the
permuted matrix is A[i][j]. this makes row swapping fast. for column swapping
we still have to actually move the data.

during execution of this algorithm we maintain an L*D*L' factorization of
the clamped submatrix of A (call it `AC') which is the top left nC*nC
submatrix of A. there are two ways we could arrange the rows/columns in AC.

(1) AC is always permuted such that L*D*L' = AC. this causes a problem
    when a row/column is removed from C, because then all the rows/columns of A
    between the deleted index and the end of C need to be rotated downward.
    this results in a lot of data motion and slows things down.
(2) L*D*L' is actually a factorization of a *permutation* of AC (which is
    itself a permutation of the underlying A). this is what we do - the
    permutation is recorded in the vector C. call this permutation A[C,C].
    when a row/column is removed from C, all we have to do is swap two
    rows/columns and manipulate C.

*/

#include <ode/common.h>
#include "lcp.h"
#include <ode/matrix.h>
#include <ode/misc.h>
#include "mat.h"		// for testing
#include <ode/timer.h>		// for testing
#include <cassert>
#include <vector>
#include <mkl_lapack.h>
#include <mkl_blas.h>

#include "twigg/linalg.h"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/scoped_array.hpp>

//***************************************************************************
// code generation parameters

// LCP debugging (mosty for fast dLCP) - this slows things down a lot
/*
#ifdef _DEBUG
#define DEBUG_FACTORIZATION
#define DEBUG_LCP
#endif
*/

#define CheckAndDump(x) verifyCondition(x, #x )

void dSaveLCP (std::ostream& ofs, int n, const dReal* A, const dReal* x, const dReal* b, const dReal* w, int nub, const dReal* lo, const dReal* hi, const int* findex)
{
    int nskip = dPAD(n);

    assert( ofs );
    ofs.write( reinterpret_cast<const char*>( &n ), sizeof( int ) );
    ofs.write( reinterpret_cast<const char*>( &nub ), sizeof( int ) );
    ofs.write( reinterpret_cast<const char*>( A ), n*nskip*sizeof( dReal ) );
    ofs.write( reinterpret_cast<const char*>( x ), n*sizeof( dReal ) );
    ofs.write( reinterpret_cast<const char*>( b ), n*sizeof( dReal ) );
    ofs.write( reinterpret_cast<const char*>( w ), n*sizeof( dReal ) );
    ofs.write( reinterpret_cast<const char*>( lo ), n*sizeof( dReal ) );
    ofs.write( reinterpret_cast<const char*>( hi ), n*sizeof( dReal ) );
    ofs.write( reinterpret_cast<const char*>( findex ), n*sizeof( int ) );
}


void dSaveLCP (const char* filename, int n, const dReal* A, const dReal* x, const dReal* b, const dReal* w, int nub, const dReal* lo, const dReal* hi, const int* findex)
{
    std::ofstream ofs( filename, std::ios::binary );

    dSaveLCP( ofs, n, A, x, b, w, nub, lo, hi, findex );
}

bool isLoadedFromDisk = false;

void dLoadLCP( const char* filename )
{
    std::ifstream ifs( filename, std::ios::binary );
    assert( ifs );

    int n, nub;
    ifs.read( reinterpret_cast<char*>( &n ), sizeof( int ) );
    ifs.read( reinterpret_cast<char*>( &nub ), sizeof( int ) );

    int nskip = dPAD(n);

    std::vector<dReal> A( n*nskip );
    std::vector<dReal> x( n );
    std::vector<dReal> b( n );
    std::vector<dReal> w( n );
    std::vector<dReal> lo( n );
    std::vector<dReal> hi( n );
    std::vector<int> findex( n );

    ifs.read( reinterpret_cast<char*>( &A[0] ), A.size()*sizeof(dReal) );
    ifs.read( reinterpret_cast<char*>( &x[0] ), x.size()*sizeof(dReal) );
    ifs.read( reinterpret_cast<char*>( &b[0] ), b.size()*sizeof(dReal) );
    ifs.read( reinterpret_cast<char*>( &w[0] ), w.size()*sizeof(dReal) );
    ifs.read( reinterpret_cast<char*>( &lo[0] ), lo.size()*sizeof(dReal) );
    ifs.read( reinterpret_cast<char*>( &hi[0] ), hi.size()*sizeof(dReal) );
    ifs.read( reinterpret_cast<char*>( &findex[0] ), findex.size()*sizeof(int) );

    isLoadedFromDisk = true;
    dSolveLCP( n, &A[0], &x[0], &b[0], &w[0], nub, &lo[0], &hi[0], &findex[0] );
    isLoadedFromDisk = false;
}

void saveProblem (const char* prefix, int n, const dReal* A, const dReal* x, const dReal* b, const dReal* w, int nub, const dReal* lo, const dReal* hi, const int* findex)
{
    int i = 1;
    while( true )
    {
        std::ostringstream oss;
        oss << prefix << i++ << ".bin";
        std::ifstream ifs( oss.str().c_str() );
        if( !ifs )
        {
            std::ofstream ofs( oss.str().c_str(), std::ios::binary );
            dSaveLCP( ofs, n, A, x, b, w, nub, lo, hi, findex );
            return;
        }
    }
}

void checkVector( const dReal* vec, int n )
{
#ifdef _DEBUG
    for( int i = 0; i < n; ++i )
        assert( vec[i] > -1e10 && vec[i] < 1e10 );
#endif
}

bool withinEpsilon( dReal a, dReal b, dReal scale )
{
    scale = std::max( std::max( std::max( std::abs(a), std::abs(b) ), std::abs(scale) ), 1.0 );
    const dReal diff = std::abs<dReal>( a - b );
    const dReal epsilon = std::numeric_limits<dReal>::epsilon();
    return ( diff < 1e-7 || diff <= 1e6*scale*epsilon );
}

bool withinEpsilon( dReal a, dReal b )
{
    return withinEpsilon( a, b, 1.0 );
}

// option 1 : matrix row pointers (less data copying)
#define ROWPTRS
#define ATYPE dReal **
#define AROW(i) (A[i])

// option 2 : no matrix row pointers (slightly faster inner loops)
//#define NOROWPTRS
//#define ATYPE dReal *
//#define AROW(i) (A+(i)*nskip)

// use protected, non-stack memory allocation system

#ifdef dUSE_MALLOC_FOR_ALLOCA
extern unsigned int dMemoryFlag;

#define ALLOCA(t,v,s) t* v = (t*) malloc(s)
#define UNALLOCA(t)  free(t)

#else

#define ALLOCA(t,v,s) t* v =(t*)dALLOCA16(s)
#define UNALLOCA(t)  /* nothing */

#endif

#define NUB_OPTIMIZATIONS

//***************************************************************************

// Apply the rotation [ cs sn; -sn cs ] to the packed matrix R, 
// rows i:i+1 and columns [startColumn, endColumn)
void drotPacked( dReal* R, 
                 const size_t row, 
                 const size_t startColumn, 
                 const size_t endColumn, 
                 dReal cs,
                 dReal sn )
{
    // Apply update to R
    for( size_t jCol = startColumn, jEntry = (jCol*(jCol+1))/2 + row; 
        jCol < endColumn; 
        ++jCol )
    {
        const dReal r1 = R[ jEntry   ];
        const dReal r2 = R[ jEntry+1 ];
        R[ jEntry   ] = cs*r1 + sn*r2;
        R[ jEntry+1 ] = -sn*r1 + cs*r2;

        jEntry += jCol + 1;
    }
}

// swap row/column i1 with i2 in the n*n matrix A. the leading dimension of
// A is nskip. this only references and swaps the lower triangle.
// if `do_fast_row_swaps' is nonzero and row pointers are being used, then
// rows will be swapped by exchanging row pointers. otherwise the data will
// be copied.

static void swapRowsAndCols (ATYPE A, int n, int i1, int i2, int nskip,
			     int do_fast_row_swaps)
{
  int i;
  dAASSERT (A && n > 0 && i1 >= 0 && i2 >= 0 && i1 < n && i2 < n &&
	    nskip >= n && i1 < i2);

# ifdef ROWPTRS
  for (i=i1+1; i<i2; i++) A[i1][i] = A[i][i1];
  for (i=i1+1; i<i2; i++) A[i][i1] = A[i2][i];
  A[i1][i2] = A[i1][i1];
  A[i1][i1] = A[i2][i1];
  A[i2][i1] = A[i2][i2];
  // swap rows, by swapping row pointers
  if (do_fast_row_swaps) {
    dReal *tmpp;
    tmpp = A[i1];
    A[i1] = A[i2];
    A[i2] = tmpp;
  }
  else {
    ALLOCA (dReal,tmprow,n * sizeof(dReal));

#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (tmprow == NULL) {
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;      
      return;
    }
#endif

    memcpy (tmprow,A[i1],n * sizeof(dReal));
    memcpy (A[i1],A[i2],n * sizeof(dReal));
    memcpy (A[i2],tmprow,n * sizeof(dReal));
    UNALLOCA(tmprow);
  }
  // swap columns the hard way
  for (i=i2+1; i<n; i++) {
    dReal tmp = A[i][i1];
    A[i][i1] = A[i][i2];
    A[i][i2] = tmp;
  }
# else
  dReal tmp;
  ALLOCA (dReal,tmprow,n * sizeof(dReal));

#ifdef dUSE_MALLOC_FOR_ALLOCA
  if (tmprow == NULL) {
    return;
  }
#endif

  if (i1 > 0) {
    memcpy (tmprow,A+i1*nskip,i1*sizeof(dReal));
    memcpy (A+i1*nskip,A+i2*nskip,i1*sizeof(dReal));
    memcpy (A+i2*nskip,tmprow,i1*sizeof(dReal));
  }
  for (i=i1+1; i<i2; i++) {
    tmp = A[i2*nskip+i];
    A[i2*nskip+i] = A[i*nskip+i1];
    A[i*nskip+i1] = tmp;
  }
  tmp = A[i1*nskip+i1];
  A[i1*nskip+i1] = A[i2*nskip+i2];
  A[i2*nskip+i2] = tmp;
  for (i=i2+1; i<n; i++) {
    tmp = A[i*nskip+i1];
    A[i*nskip+i1] = A[i*nskip+i2];
    A[i*nskip+i2] = tmp;
  }
  UNALLOCA(tmprow);
# endif

}


// swap two indexes in the n*n LCP problem. i1 must be <= i2.

static void swapProblem (ATYPE A, dReal *x, dReal *b, dReal *w, dReal *lo,
			 dReal *hi, int *p, int* pinv, int *state, int *findex,
			 int n, int i1, int i2, int nskip,
			 int do_fast_row_swaps)
{
  dIASSERT (n>0 && i1 >=0 && i2 >= 0 && i1 < n && i2 < n && nskip >= n &&
	    i1 <= i2);
  if (i1==i2) return;
  swapRowsAndCols (A,n,i1,i2,nskip,do_fast_row_swaps);
#ifdef dUSE_MALLOC_FOR_ALLOCA
  if (dMemoryFlag == d_MEMORY_OUT_OF_MEMORY)
    return;
#endif
  std::swap( x[i1], x[i2] );
  std::swap( b[i1], b[i2] );
  std::swap( w[i1], w[i2] );
  std::swap( lo[i1], lo[i2] );
  std::swap( hi[i1], hi[i2] );
  std::swap( p[i1], p[i2] );
  std::swap( state[i1], state[i2] );
  if (findex) {
      std::swap( findex[i1], findex[i2] );
  }

  pinv[ p[i1] ] = i1;
  pinv[ p[i2] ] = i2;
}


// for debugging - check that L,d is the factorization of A[C,C].
// A[C,C] has size nC*nC and leading dimension nskip.
// L has size nC*nC and leading dimension nskip.
// d has size nC.


void addMatrix( MATFile& file, const dMatrix& mat, const char* name )
{
    fortran_matrix m( mat.nrows(), mat.ncols() );
    for( size_t i = 0; i < m.nrows(); ++i )
        for( size_t j = 0; j < m.ncols(); ++j )
            m(i, j) = mat(i, j);

    file.add( name, m );
}

// for debugging

static void checkPermutations (int i, int n, int nC, int nN, int *p, int* pinv)
{
  dIASSERT (nC>=0 && nN>=0 && (nC+nN)==i && i < n);
  for (int k=0; k<n; k++) dIASSERT (p[k] >= 0 && p[k] < n);
  for (int k=0; k<n; k++) dIASSERT (pinv[k] >= 0 && pinv[k] < n && p[ pinv[k] ] == k );
  // this is not true because the first thing we do is to permute the findex entries to the back
  //  for (k=i; k<n; k++) dIASSERT (p[k] == k);
}

//***************************************************************************
// dLCP manipulator object. this represents an n*n LCP problem.
//
// two index sets C and N are kept. each set holds a subset of
// the variable indexes 0..n-1. an index can only be in one set.
// initially both sets are empty.
//
// the index set C is special: solutions to A(C,C)\A(C,i) can be generated.

//***************************************************************************
// fast implementation of dLCP. see the above definition of dLCP for
// interface comments.
//
// `p' records the permutation of A,x,b,w,etc. p is initially 1:n and is
// permuted as the other vectors/matrices are permuted.
//
// A,x,b,w,lo,hi,state,findex,p,c are permuted such that sets C,N have
// contiguous indexes. the don't-care indexes follow N.
//
// an L*D*L' factorization is maintained of A(C,C), and whenever indexes are
// added or removed from the set C the factorization is updated.
// thus L*D*L'=A[C,C], i.e. a permuted top left nC*nC submatrix of A.
// the leading dimension of the matrix L is always `nskip'.
//
// at the start there may be other indexes that are unbounded but are not
// included in `nub'. dLCP will permute the matrix so that absolutely all
// unbounded vectors are at the start. thus there may be some initial
// permutation.
//
// the algorithms here assume certain patterns, particularly with respect to
// index transfer.

struct dLCP {
  int n,nskip,nub;
  ATYPE A;				// A rows
  dReal *Adata,*x,*b,*w,*lo,*hi;	// permuted LCP problem data
  int *state,*findex,*p;
  std::vector<int> pinv;    // need to be know where any given index has gone
  int nC,nN;				// size of each index set

  dLCP (int _n, int _nub, dReal *_Adata, dReal *_x, dReal *_b, dReal *_w,
	dReal *_lo, dReal *_hi, 
	int *_state, int *_findex, int *_p, dReal **Arows);
  int getNub() const { return nub; }
  void transfer_i_to_C (int i);
  void transfer_i_to_N (int i);
  void transfer_i_from_N_to_C (int i);
  void transfer_i_from_C_to_N (int i);
  void transfer_i_from_N_to_end (int i);
  int numC() const { return nC; }
  int numN() const { return nN; }
  int indexC (int i) const { return i; }
  int indexN (int i) const { return i+nC; }
  dReal Aii (int i) const { return AROW(i)[i]; }
  dReal Aij (int i, int j) const;
  dReal AiC_times_qC (int i, dReal *q) const;
  dReal AiN_times_qN (int i, dReal *q) const;
  dReal Ai_times_q (int i, dReal *q) const; // use the whole matrix
  void pN_equals_ANC_times_qC (dReal *p, dReal *q) const;
  void pN_plusequals_ANi (dReal *p, int i, int sign=1) const;
  void pN_plusequals_ANi_scaled (dReal *p, int i, dReal scale) const;
  void pC_plusequals_s_times_qC (dReal *p, dReal s, dReal *q) const
    { for (int i=0; i<nC; i++) p[i] += s*q[i]; }
  void pN_plusequals_s_times_qN (dReal *p, dReal s, dReal *q) const
    { for (int i=0; i<nN; i++) p[i+nC] += s*q[i+nC]; }
  void solve1 (dReal *a, int i, int dir=1);
  void unpermute();
  void checkInvariants() const;

  bool redundantConstraint( const dReal* newRow, const dReal* newColumn ) const;

  dReal effectiveLo(int index) const;
  dReal effectiveHi(int index) const;

  int inverse_permute(int i) const { return pinv[i]; }

  void verifyCondition( bool condition, const char* name ) const;

#ifdef DEBUG_FACTORIZATION
  void checkFactorization() const;
#endif

private:
  void rotate_i_to_end( int i );

  // normally we would store Q and R in the same nxn matrix.
  // unfortunately, for the QR updates it _appears_ that we
  // need to store the full Q matrix instead.  
  // @todo check on this!
  boost::scoped_array<dReal> _Q;

  // Multiply x by Q^T and put the result in out
  void QtransTimesVec( const dReal* x, dReal* out ) const;

  boost::scoped_array<dReal> _R;

  dReal& R_element( size_t row, size_t col )
  {
    assert( row <= col );
    return _R[ col*(col+1)/2 + row ];
  }

  const dReal& R_element( size_t row, size_t col ) const
  {
    assert( row <= col );
    return _R[ col*(col+1)/2 + row ];
  }

  // This is the size of both _Q and _R.  It will grow
  // exponentially to accommodate growth in nC.
  // Because _R is packed, its size is _nrows*(_nrows+1)/2,
  // while _Q is _nrows*_nrows.
  size_t _nrows;

  // Compute a complete QR factorization of A(1:nC, 1:nC)
  void factorize();

  // Update the factorization by adding to a particular column
  void addToColumn( size_t iColumn, const dReal* values );

  // Update the factorization by adding a row and column at the end
  void addRowAndColumn( const dReal* row, const dReal* column );

  // Update the factorization by removing a row and column
  void removeRowAndColumn( size_t i );

  void solve(const dReal* b, dReal* x) const;

  void constructMatrix( dReal* matrix, size_t stride ) const;

#ifdef DEBUG_FACTORIZATION
  void checkFactorization( 
    const fortran_matrix& fullA,
    const dReal* updateToQ,
    size_t iUpdateColumn,
    const std::vector<dReal>& belowDiagonal ) const;
  void checkFactorization( 
    const fortran_matrix& fullA,
    const dReal* rowUpdate,
    const dReal* columnUpdate,
    const dReal* Qpacked,
    const dReal* Rpacked,
    const std::vector<dReal>& belowDiagonal ) const;
void checkFactorization( 
    const fortran_matrix& fullA,
    const dReal* Qpacked,
    const dReal* Rpacked,
    const dReal* spikeColumn,
    size_t iSpikeColumn,
    const std::vector<dReal>& belowDiagonal ) const;
#endif

  fortran_matrix QRproduct() const;

  void dumpMatrices( MATFile& matFile ) const;

#ifdef DEBUG_LCP
  std::vector<dReal> A_backup;
  std::vector<dReal> x_backup;
  std::vector<dReal> b_backup;
  std::vector<dReal> w_backup;
  std::vector<dReal> lo_backup;
  std::vector<dReal> hi_backup;
  std::vector<int> findex_backup;
  int nub_backup;

  mutable bool saved;
#endif
};

fortran_matrix QRproduct( const dReal* Q, const dReal* R, int nC, int stride );


dLCP::dLCP (int _n, int _nub, dReal *_Adata, dReal *_x, dReal *_b, dReal *_w,
	    dReal *_lo, dReal *_hi, 
	    int *_state, int *_findex, int *_p, dReal **Arows)
        : _nrows(0)
{
  n = _n;
  nub = _nub;
  nskip = dPAD(n);

#ifdef DEBUG_LCP
  // back it up:
    A_backup.resize( n*nskip );     std::copy( _Adata, _Adata+n*nskip, A_backup.begin() );
    x_backup.resize( n );           std::copy( _x, _x+n, x_backup.begin() );
    b_backup.resize( n );           std::copy( _b, _b+n, b_backup.begin() );
    w_backup.resize( n );           std::copy( _w, _w+n, w_backup.begin() );
    lo_backup.resize( n );          std::copy( _lo, _lo+n, lo_backup.begin() );
    hi_backup.resize( n );          std::copy( _hi, _hi+n, hi_backup.begin() );
    findex_backup.resize( n );      std::copy( _findex, _findex+n, findex_backup.begin() );
    nub_backup = _nub;

    saved = false;
#endif

  Adata = _Adata;
  A = 0;
  x = _x;
  b = _b;
  w = _w;
  lo = _lo;
  hi = _hi;
  state = _state;
  findex = _findex;
  p = _p;
  dSetZero (x,n);

  int k;

# ifdef ROWPTRS
  // make matrix row pointers
  A = Arows;
  for (k=0; k<n; k++) A[k] = Adata + k*nskip;
# else
  A = Adata;
# endif

  nC = 0;
  nN = 0;
  for (k=0; k<n; k++) p[k]=k;		// initially unpermuted

  pinv.resize(n);
  for( int i = 0; i < n; ++i )
      pinv[i] = i;

  /*
  // for testing, we can do some random swaps in the area i > nub
  if (nub < n) {
    for (k=0; k<100; k++) {
      int i1,i2;
      do {
	i1 = dRandInt(n-nub)+nub;
	i2 = dRandInt(n-nub)+nub;
      }
      while (i1 > i2); 
      //printf ("--> %d %d\n",i1,i2);
      swapProblem (A,x,b,w,lo,hi,p,state,findex,n,i1,i2,nskip,0);
    }
  }
  */

  // permute the problem so that *all* the unbounded variables are at the
  // start, i.e. look for unbounded variables not included in `nub'. we can
  // potentially push up `nub' this way and get a bigger initial factorization.
  // note that when we swap rows/cols here we must not just swap row pointers,
  // as the initial factorization relies on the data being all in one chunk.
  // variables that have findex >= 0 are *not* considered to be unbounded even
  // if lo=-inf and hi=inf - this is because these limits may change during the
  // solution process.

  for (k=0; k < nub; k++)
  {
    assert( lo[k]==-dInfinity && hi[k]==dInfinity );
  }

  for (k=nub; k<n; k++) {
    if (findex && findex[k] >= 0) continue;
    if (lo[k]==-dInfinity && hi[k]==dInfinity) {
      swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,nub,k,nskip,0);
      nub++;
    }
  }

  // permute the indexes > nub such that all findex variables are at the end
  if (findex) {
    int num_at_end = 0;
    for (k=n-1; k >= nub; k--) {
      if (findex[k] >= 0) {
	swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,k,n-1-num_at_end,nskip,1);
	num_at_end++;
      }
    }
  }

  // if there are unbounded variables at the start, factorize A up to that
  // point and solve for x. this puts all indexes 0..nub-1 into C.
  if( nub > 0 )
  {
      nC = nub;
      this->factorize();
      this->solve( b, x );
      dSetZero (w,nub);
  }

  // print info about indexes
  /*
  for (k=0; k<n; k++) {
    if (k<nub) printf ("C");
    else if (lo[k]==-dInfinity && hi[k]==dInfinity) printf ("c");
    else printf (".");
  }
  printf ("\n");
  */
}

void dLCP::factorize()
{
    // MKL will get grouchy if we pass in 0s
    if( nC == 0 )
        return;

    if( this->_nrows < nC )
    {
        this->_nrows = std::max<size_t>( nC, std::min<size_t>( _nrows*2, n ) );
        _Q.reset( new dReal[_nrows*_nrows] );
        _R.reset( new dReal[_nrows*(_nrows+1)/2] );
    }

    int m = nC;
    int n = nC;
    int lda = nC;
    int info;

    std::vector<dReal> tau( nC );

    int lwork = 64*nC;

    this->constructMatrix( &_Q[0], nC );

    // work array query
    dReal* work = (dReal*) alloca( lwork*sizeof(dReal) );
    dgeqrf(
        &m,
        &n,
        &_Q[0],
        &lda,
        &tau[0],
        work,
        &lwork,
        &info );

    // sthg smarter here?
    assert( info == 0 );


    // pack in the upper triangular portion
    for( int i = 0; i < nC; ++i )
        std::copy( _Q.get() + i*nC, _Q.get() + i*nC + nC, 
                   _R.get() + (i*(i+1))/2 );

    // Now need to construct the actual Q matrix:
    int k = nC;
    dorgqr(
        &m,
        &n,
        &k,
        &_Q[0],
        &lda,
        &tau[0],
        work,
        &lwork,
        &info );
    assert( info == 0 );

#ifdef DEBUG_FACTORIZATION
    this->checkFactorization();
#endif
}

void dLCP::dumpMatrices( MATFile& matFile ) const
{
    fortran_matrix A( nC, nC );
    this->constructMatrix( A.data(), nC );
    matFile.add( "A", A );

    fortran_matrix R( nC, nC );
    std::fill( R.data(), R.data() + nC*nC, 0 );
    for( size_t i = 0; i < nC; ++i )
        for( size_t j = 0; j <= i; ++j )
            R(j, i) = R_element(j, i);
    matFile.add( "R", R );

    fortran_matrix Q( nC, nC );
    for( size_t i = 0; i < nC; ++i )
        for( size_t j = 0; j < nC; ++j )
            Q(j, i) = _Q[i*nC + j];
    matFile.add( "Q", Q );
}

#ifdef DEBUG_FACTORIZATION
void dLCP::checkFactorization( 
    const fortran_matrix& fullA,
    const dReal* updateToR,
    size_t iUpdateColumn,
    const std::vector<dReal>& belowDiagonal ) const
{   
    fortran_matrix R(nC, nC);
    std::fill( R.data(), R.data() + nC*nC, 0.0 );
    for( size_t i = 0; i < nC; ++i )
        for( size_t j = 0; j <= i; ++j )
            R(j, i) = R_element(j, i);

    if( updateToR )
    {
        for( size_t i = 0; i < nC; ++i )
            R(i, iUpdateColumn) = updateToR[i];
    }

    if( !belowDiagonal.empty() )
    {
        for( size_t i = 1; i < belowDiagonal.size(); ++i )
            R(i+iUpdateColumn+1, i+iUpdateColumn) = belowDiagonal[i];
    }

    fortran_matrix Q( nC, nC );
    for( size_t i = 0; i < nC; ++i )
        for( size_t j = 0; j < nC; ++j )
            Q(j, i) = _Q[i*nC + j];

    fortran_matrix QR = prod( Q, R );

    /*
    {
        MATFile matFile( "QRdebug2.mat", "testing" );
        matFile.add( "A", fullA );
        matFile.add( "Q", Q );
        matFile.add( "R", R );
    }
    */

    for( size_t iCol = 0; iCol < nC; ++iCol )
        for( size_t iRow = 0; iRow < nC; ++iRow )
            assert( withinEpsilon( QR(iRow, iCol), fullA(iRow, iCol) ) );
}
#endif

#ifdef DEBUG_FACTORIZATION
void dLCP::checkFactorization( 
    const fortran_matrix& fullA,
    const dReal* rowUpdate,
    const dReal* columnUpdate,
    const dReal* Qpacked,
    const dReal* Rpacked,
    const std::vector<dReal>& belowDiagonal ) const
{   
    fortran_matrix R(nC+1, nC+1);
    std::fill( R.data(), R.data() + (nC+1)*(nC+1), 0.0 );
    for( size_t i = 0; i < nC+1; ++i )
        for( size_t j = 0; j <= i; ++j )
            R(j, i) = Rpacked[ i*(i+1)/2 + j ];

    if( !belowDiagonal.empty() )
    {
        assert( belowDiagonal.size() == nC );
        for( size_t i = 0; i < nC; ++i )
            R(i+1, i) = belowDiagonal[i];
    }

    fortran_matrix Q( nC+1, nC+1 );
    for( size_t i = 0; i < nC+1; ++i )
        for( size_t j = 0; j < nC+1; ++j )
            Q(j, i) = Qpacked[i*(nC+1) + j];

    fortran_matrix QR = prod( Q, R );

    /*
    {
        MATFile matFile( "QRdebug2.mat", "testing" );
        matFile.add( "A", fullA );
        matFile.add( "Q", Q );
        matFile.add( "R", R );
    }
    */

    for( size_t iCol = 0; iCol < nC; ++iCol )
        for( size_t iRow = 0; iRow < nC; ++iRow )
            assert( withinEpsilon( QR(iRow+1, iCol), fullA(iRow, iCol) ) );

    for( size_t iRow = 0; iRow < nC; ++iRow )
        assert( withinEpsilon( QR(iRow+1, nC), columnUpdate[iRow] ) );

    for( size_t iCol = 0; iCol <= nC; ++iCol )
        assert( withinEpsilon( QR(0, iCol), rowUpdate[iCol] ) );
}
#endif


// Adds a row and column to the end of the A matrix
void dLCP::addRowAndColumn( const dReal* rowUpdate, const dReal* columnUpdate )
{
    // We are going to add a new row and column to the matrix.  Although
    // the row is eventually going to go at the end, we will add it at
    // the top; this will produce a matrix that looks like this:
    //
    // [ x x x x x x ]
    // [ * * * * * x ]
    // [   * * * * x ]
    // [     * * * x ]
    // [       * * x ]
    // [         * x ]
    //
    // where the x's are the new entries.  We can convert this to 
    // upper triangular by rotating away the diagonals and then
    // permute the Q matrix to compensate for the row ordering.

#ifdef DEBUG_FACTORIZATION
    fortran_matrix fullA = this->QRproduct();
#endif

    const size_t newSize = nC+1;

    boost::scoped_array<dReal> columnUpdateInBasis( new dReal[nC] );
    if( nC > 0 )
        this->QtransTimesVec( columnUpdate, &columnUpdateInBasis[0] );

    std::vector<dReal> belowDiagonal( nC );
    for( size_t row = 0; row < nC; ++row )
        belowDiagonal[row] = R_element( row, row );

    {
        dReal* newQ = _Q.get();
        dReal* newR = _R.get();

        if( newSize > this->_nrows )
        {
            this->_nrows = std::max<size_t>( newSize, std::min<size_t>( _nrows*2, n ) );
            newQ = new dReal[_nrows*_nrows];
            newR = new dReal[_nrows*(_nrows+1)/2];
        }

        // Build the updated Q matrix of [ 1 0 ]
        //                               [ 0 Q ]
        for( int i = nC-1; i >= 0; --i )
            std::copy( _Q.get() + i*nC, _Q.get() + (i+1)*nC, newQ + (i+1)*newSize + 1 );

        newQ[0] = 1;
        for( size_t i = 1; i < newSize; ++i )
            newQ[i*newSize] = 0;
        std::fill( newQ + 1, newQ + newSize, 0 );

        // Now update the R matrix
        for( int col = nC - 1; col >= 0; --col )
            std::copy( _R.get() + col*(col+1)/2, _R.get() + (col+1)*(col+2)/2 - 1, 
                       newR + col*(col+1)/2 + 1 );

        for( size_t col = 0; col < newSize; ++col )
            newR[ col*(col+1)/2 ] = rowUpdate[col];

        for( size_t row = 0; row < nC; ++row )
            newR[ (nC*(nC+1))/2 + row + 1 ] = columnUpdateInBasis[row];
        newR[ (nC*(nC+1))/2 ] = columnUpdate[nC];

        if( newR != _R.get() )
            _R.reset( newR );
        if( newQ != _Q.get() )
            _Q.reset( newQ );
    }

#ifdef DEBUG_FACTORIZATION
    this->checkFactorization( fullA, rowUpdate, columnUpdate, _Q.get(), _R.get(), belowDiagonal );
#endif

    // Now, need to rotate out all the below-diagonal entries
    for( int i = 0; i < nC; ++i )
    {
        dReal cs, sn, r;
        dlartg( &_R[(i*(i+1))/2 + i], &belowDiagonal[i], &cs, &sn, &r );
        _R[(i*(i+1))/2 + i] = r;
        belowDiagonal[i] = 0;

        // Apply update to R
        drotPacked( &_R[0], i, i+1, newSize, cs, sn );

        // Apply update to Q
        int incx = 1, incy = 1, newSizeInt = newSize;
        drot( &newSizeInt, &_Q[ (i)*newSize ], &incx, &_Q[ (i+1)*newSize ], &incy, &cs, &sn );

#ifdef DEBUG_FACTORIZATION
        this->checkFactorization( fullA, rowUpdate, columnUpdate, _Q.get(), _R.get(), belowDiagonal );
#endif
    }

    // Now need to rotate the top row of Q down to the bottom
    if( nC > 0 )
    {
        for( size_t iCol = 0; iCol < (nC+1); ++iCol )
        {
            std::rotate( _Q.get() + iCol*(nC+1), 
                         _Q.get() + iCol*(nC+1) + 1, 
                         _Q.get() + (iCol+1)*(nC+1) );
        }
    }

#ifdef DEBUG_FACTORIZATION
    {
        // vertify that permutation did the right thing
        fortran_matrix QR = ::QRproduct(_Q.get(), _R.get(), nC+1, nC+1);
        for( size_t i = 0; i < nC; ++i )
            for( size_t j = 0; j < nC; ++j )
                assert( withinEpsilon( QR(i, j), fullA(i, j), fullA(i, j) ) );

        for( size_t i = 0; i < nC+1; ++i )
            assert( withinEpsilon( QR(i, nC), columnUpdate[i], columnUpdate[i] ) );

        for( size_t i = 0; i < nC+1; ++i )
            assert( withinEpsilon( QR(nC, i), rowUpdate[i], rowUpdate[i] ) );
    }
#endif
}

#ifdef DEBUG_FACTORIZATION
void dLCP::checkFactorization( 
    const fortran_matrix& fullA,
    const dReal* Qpacked,
    const dReal* Rpacked,
    const dReal* spikeColumn,
    size_t iSpikeColumn,
    const std::vector<dReal>& belowDiagonal ) const
{
    fortran_matrix R(nC, nC);
    std::fill( R.data(), R.data() + nC*nC, 0.0 );
    for( size_t i = 0; i < nC; ++i )
        for( size_t j = 0; j <= i; ++j )
            R(j, i) = Rpacked[ i*(i+1)/2 + j ];

    if( spikeColumn )
    {
        for( size_t i = 0; i < nC; ++i )
            R(i, iSpikeColumn) = spikeColumn[i];
    }

    for( size_t i = 0; i < belowDiagonal.size(); ++i )
        R(i+1, i) = belowDiagonal[i];

    fortran_matrix Q( nC, nC );
    for( size_t i = 0; i < nC; ++i )
        for( size_t j = 0; j < nC; ++j )
            Q(j, i) = Qpacked[i*nC + j];

    fortran_matrix QR = prod( Q, R );

    /*
    {
        MATFile matFile( "QRdebug2.mat", "testing" );
        matFile.add( "A", fullA );
        matFile.add( "Q", Q );
        matFile.add( "R", R );
    }
    */

    for( size_t iCol = 0; iCol < nC; ++iCol )
        for( size_t iRow = 0; iRow < nC; ++iRow )
            assert( withinEpsilon( QR(iRow, iCol), fullA(iRow, iCol) ) );
}
#endif

void dLCP::removeRowAndColumn( size_t iRemoved )
{
    assert( iRemoved < nC );
    if( nC == 1 )
        return;

    // We will remove a row and column from the matrix.  
    const size_t newSize = nC - 1;

#ifdef DEBUG_FACTORIZATION
    fortran_matrix beforeRotate = this->QRproduct();
#endif

    // The first step is to permute the removed row up to the
    // top, since we can only really remove rows up there.
    for( size_t iCol = 0; iCol < nC; ++iCol )
    {
        std::rotate( _Q.get() + iCol*nC, 
                     _Q.get() + iCol*nC + iRemoved,
                     _Q.get() + iCol*nC + iRemoved + 1 );
    }

#ifdef DEBUG_FACTORIZATION
    fortran_matrix fullA = this->QRproduct();
#endif

    // Because we want to remove the column, the first step is
    // to zero out entries along the diagonal; that is, we will turn
    // this:
    //       i
    // [ * * * * * ]      [ * * x * * ]
    // [   * * * * ]      [   * x * * ]
    // [     * * * ] ---> [     x * * ]
    // [       * * ]      [     x   * ]
    // [         * ]      [     x     ]
    //
    boost::scoped_array<dReal> removedColumn( new dReal[nC] );
    std::copy( _R.get() + iRemoved*(iRemoved+1)/2, _R.get() + (iRemoved+1)*(iRemoved+2)/2, removedColumn.get() );
    std::fill( removedColumn.get() + iRemoved + 1, removedColumn.get() + nC, 0 );

    std::vector<dReal> belowDiagonal( iRemoved, 0 );

#ifdef DEBUG_FACTORIZATION
    /*
    {
        MATFile dump( "QRdebug.mat", "debugging QR" );
        dumpMatrices( dump );
        dump.add( "iCol", iRemoved+1 );
        dump.add( "beforeRotate", beforeRotate );
    }
    */
    this->checkFactorization( fullA, _Q.get(), _R.get(), removedColumn.get(), iRemoved, belowDiagonal );
#endif

    for( size_t i = iRemoved + 1; i < nC; ++i )
    {
        // compute Givens rotation parameters for Givens matrix G = [ cos sin; -sin cos ]
        dReal cs, sn, r;
        dlartg( &R_element(i-1, i), &R_element(i, i), &cs, &sn, &r );
        R_element(i-1, i) = r;
        R_element(i, i) = 0;

        // Apply update to R
        drotPacked( &_R[0], i-1, i+1, nC, cs, sn );
        
        // Apply update to spike column
        {
            dReal r1 = removedColumn[i-1];
            dReal r2 = removedColumn[i];
            removedColumn[i-1] = cs*r1 + sn*r2;
            removedColumn[i] = -sn*r1 + cs*r2;
        }

        // Apply update to Q
        int incx = 1, incy = 1;
        drot( &nC, &_Q[ (i-1)*nC ], &incx, &_Q[ (i)*nC ], &incy, &cs, &sn );

#ifdef DEBUG_FACTORIZATION
        this->checkFactorization( fullA, _Q.get(), _R.get(), removedColumn.get(), iRemoved, belowDiagonal );
#endif
    }

    // Now, we need to apply rotations to convert the first row of Q to
    //    [1 0 0 0 ... ]
    for( size_t i = newSize; i > 0; --i )
    {
        dReal cs, sn, r;
        dlartg( &_Q[(i-1)*nC], &_Q[i*nC], &cs, &sn, &r );
        _Q[(i-1)*nC] = r;
        _Q[(i)  *nC] = 0;

        // Apply update to Q
        int incx = 1, incy = 1, newSizeInt = nC-1;
        drot( &newSizeInt, &_Q[ (i-1)*nC + 1 ], &incx, &_Q[ (i)*nC + 1 ], &incy, &cs, &sn );

        // Apply update to R
        drotPacked( &_R[0], i-1, i, nC, cs, sn );
        
        // Apply update to spike column
        {
            dReal r1 = removedColumn[i-1];
            dReal r2 = removedColumn[i];
            removedColumn[i-1] = cs*r1 + sn*r2;
            removedColumn[i] = -sn*r1 + cs*r2;
        }

        if( i <= iRemoved )
            belowDiagonal[i-1] = -sn*R_element( i-1, i-1 );
        R_element( i-1, i-1 ) = cs*R_element( i-1, i-1 );

#ifdef DEBUG_FACTORIZATION
        this->checkFactorization( fullA, _Q.get(), _R.get(), removedColumn.get(), iRemoved, belowDiagonal );
#endif                
    }

    // Now we need to reconstruct the final matrices
    for( size_t iCol = 1; iCol < nC; ++iCol )
        std::copy( _Q.get() + (iCol)*nC + 1, _Q.get() + (iCol+1)*nC, _Q.get() + (iCol-1)*newSize );

    for( size_t iCol = 0; iCol < iRemoved; ++iCol )
    {
        std::copy( _R.get() + (iCol*(iCol+1))/2 + 1,
                   _R.get() + (iCol+1)*(iCol+2)/2,
                   _R.get() + (iCol*(iCol+1))/2 );
        _R[ (iCol*(iCol+1))/2 + iCol ] = belowDiagonal[iCol];
    }

    for( size_t iCol = iRemoved + 1; iCol < nC; ++iCol )
    {
        std::copy( _R.get() + iCol*(iCol+1)/2 + 1,
                   _R.get() + (iCol+1)*(iCol+2)/2,
                   _R.get() + (iCol-1)*iCol/2 );
    }

#ifdef DEBUG_FACTORIZATION
    fortran_matrix QRprod = ::QRproduct(_Q.get(), _R.get(), nC-1, nC-1);
    for( size_t i = 0; i < nC; ++i )
    {
        for( size_t j = 0; j < nC; ++j )
        {
            if( i == iRemoved || j == iRemoved )
                continue;

            size_t i2 = (i > iRemoved) ? (i-1) : (i);
            size_t j2 = (j > iRemoved) ? (j-1) : (j);
            assert( withinEpsilon( beforeRotate(i, j), QRprod(i2, j2), beforeRotate(i, j) ) ); 
        }
    }
#endif
}

void dLCP::addToColumn( size_t iColumn, const dReal* update )
{
    assert( iColumn < nC );

#ifdef DEBUG_FACTORIZATION
    /*
    {
        MATFile dump( "QRdebug.mat", "debugging QR" );
        dumpMatrices( dump );
        dump.add( "update", update, nC );
    }
    */

    fortran_matrix fullA = this->QRproduct();
    for( size_t i = 0; i < nC; ++i )
        fullA(i, iColumn) += update[i];
#endif

    // So we have our factorization A = QR, and we want to update
    // this to reflect our new A = [ A_1 A_2 ... A_i + update ... A_n ]
    // First, we can express this update in the basis provided by Q.
    // That will change R to look like this:
    //       i
    // [ * * x * * ]
    // [   * x * * ]
    // [     x * * ]
    // [     x * * ]
    // [     x   * ]
    // with this "spike" in the ith column.  We can correct this with
    // a bunch of givens rotations:
    //    G_1 * G_2 * R
    // which then have to get post-multiplied onto the Q matrix (okay
    // since Givens rotations are orthogonal).

    boost::scoped_array<dReal> updateInBasis( new dReal[nC] );
    this->QtransTimesVec( update, &updateInBasis[0] );

    // Okay, now we know what the update looks like, we need to
    // compute the Givens rotations to clear it out.
    // Before we do that, though, we should actually apply the update
    // to the appropriate spots of R (the ones above the diagonal)

    for( int j = 0; j <= iColumn; ++j )
        updateInBasis[j] += this->R_element(j, iColumn);

    // When we apply Givens rotations to elements on the diagonal,
    // we'll end up adding elements just below them, which will 
    // need to get rotated out later.  We'll go ahead and store these
    // even for the bottom right element, even though it is obviously
    // not going to have one in the final reckoning.
    std::vector<dReal> belowDiagonal( nC - iColumn - 1, 0 );
#ifdef DEBUG_FACTORIZATION
    this->checkFactorization( fullA, updateInBasis.get(), iColumn, belowDiagonal );
#endif
    for( int i = nC-1; i > iColumn; --i )
    {
        assert( i > 0 );

        // compute Givens rotation parameters for Givens matrix G = [ cos sin; -sin cos ]
        dReal cs, sn, r;
        dlartg( &updateInBasis[i-1], &updateInBasis[i], &cs, &sn, &r );
        updateInBasis[i] = 0;
        updateInBasis[i-1] = r;

        // Apply update to R
        drotPacked( &_R[0], i-1, i, nC, cs, sn );
        
        {
            dReal r1 = R_element( i-1, i-1 );
            R_element( i-1, i-1 ) = r1*cs;
            belowDiagonal[i - iColumn - 1] = -sn*r1;
        }

        // Apply update to Q
        int incx = 1, incy = 1;
        drot( &nC, &_Q[ (i-1)*nC ], &incx, &_Q[ (i)*nC ], &incy, &cs, &sn );

#ifdef DEBUG_FACTORIZATION
        this->checkFactorization( fullA, updateInBasis.get(), iColumn, belowDiagonal );
#endif
    }

    for( int j = 0; j <= iColumn; ++j )
        this->R_element(j, iColumn) = updateInBasis[j];
    updateInBasis.reset();

    // Now, need to rotate out all the below-diagonal entries
    for( int i = iColumn+1; i < nC-1; ++i )
    {
        dReal cs, sn, r;
        dlartg( &R_element(i, i), &belowDiagonal[i-iColumn], &cs, &sn, &r );
        R_element(i, i) = r;
        belowDiagonal[i-iColumn] = 0;

        // Apply update to R
        drotPacked( &_R[0], i, i+1, nC, cs, sn );

        // Apply update to Q
        int incx = 1, incy = 1;
        drot( &nC, &_Q[ i*nC ], &incx, &_Q[ (i+1)*nC ], &incy, &cs, &sn );

#ifdef DEBUG_FACTORIZATION
        this->checkFactorization( fullA, updateInBasis.get(), iColumn, belowDiagonal );
#endif
    }

    belowDiagonal.clear();
#ifdef DEBUG_FACTORIZATION
    this->checkFactorization( fullA, updateInBasis.get(), iColumn, belowDiagonal );
#endif
}

void dLCP::QtransTimesVec( const dReal* x, dReal* out ) const
{
    char trans = 'T';
    int m = nC;
    int n = nC;
    int lda = nC;
    int incx = 1;
    int incy = 1;

    dReal alpha = 1;
    dReal beta = 0;

    // multiply b by Q^T
    dgemv( 
        &trans,
        &m,
        &n,
        &alpha,
        &this->_Q[0],
        &lda,
        x,
        &incx,
        &beta,
        out,
        &incy );
}

void dLCP::solve( const dReal* b, dReal* x ) const
{
    // We are solving Ax = b here
    // Since A + QR, this becomes
    //      Q*R*x = b
    // or
    //        R*x = Q^T * b
    // which can be solved with simple back substitution.

    this->QtransTimesVec( b, x );

    {
        // now back-substitute
        char uplo = 'U';    // upper triangular
        char trans = 'N';   // not transposed
        char diag = 'N';    // not unit diagonal
        int n = nC;
        int incx = 1;

        dtpsv( 
            &uplo,
            &trans,
            &diag,
            &n,
            &_R[0],
            x,
            &incx );
    }
}

void dLCP::verifyCondition (bool condition, const char* value) const
{
#ifdef DEBUG_LCP
    if( condition )
        return;

    if( isLoadedFromDisk )
    {
        assert( condition );
    }
    else if( !saved )
    {
      saveProblem( "lcpBad", n, &A_backup[0], &x_backup[0], &b_backup[0], &w_backup[0], nub_backup, &lo_backup[0], &hi_backup[0], &findex_backup[0] );
      saved = true;
    }
#endif
}

bool dLCP::redundantConstraint( const dReal* newRow, const dReal* newColumn ) const
{
  if( nC == 0 )
      return false;

  assert( newRow[nC] == newColumn[nC] );

  // We are testing whether adding the row/column to the matrix
  // makes it singular; that is, we have our matrix A and we want to
  // know whether our new column -- with its extra row -- is in the
  // columnspace of A:
  //
  // [          ] 
  // [   A   w  ] 
  // [          ] 
  // [  v^T  u  ] 
  // [          ] 
  //
  // We will do this by solving Ax = w, which is guaranteed to have a 
  // single solution (under the assumption that A is not singular).  
  // We will then check whether v^t*x == u (within machine epsilon).
  std::vector<dReal> tmp( nC );
  this->solve( newColumn, &tmp[0] );
  dReal accum = 0;
  for( size_t i = 0; i < nC; ++i )
      accum += tmp[i]*newRow[i];

  return ( std::abs(newRow[nC] - accum) < 1e-10 );
}

void dLCP::transfer_i_to_C (int i)
{
#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif
  assert( i == (nC+nN) );

  swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,nC,i,nskip,1);
 
  std::vector<dReal> newColumn( nC+1 );

  const dReal* aRow = AROW(nC);
  std::copy( aRow, aRow + (nC+1), newColumn.begin() );

  std::vector<dReal> newRow = newColumn;

  if( findex != 0 )
  {
    // In this case, may need to add to column
    for( int j = 0; j < nN; ++j )
    {
        if( findex[ nC+1+j ] < 0 )
            continue;

        int normalIndex = inverse_permute( findex[ nC+1+j ] );
        if( normalIndex <= nC )
        {
            dReal alpha;
            if( state[ nC+1+j ] == 0 )
                alpha = lo[ nC+1+j ];
            else
                alpha = hi[ nC+1+j ];

            newRow[ normalIndex ] += alpha*AROW( nC+1+j )[nC];

            if( normalIndex == nC )
            {
                for( int k = 0; k < nC+1; ++k )
                    newColumn[ k ] += alpha*AROW( nC+1+j )[k];
            }
        }
    }
  }

  if( this->redundantConstraint(&newRow[0], &newColumn[0]) )
  {
    swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,nC,i,nskip,1);
    state[i] = 2;
    transfer_i_to_N(i);
    return;
  }

  this->addRowAndColumn( &newRow[0], &newColumn[0] );
  nC++;

# ifdef DEBUG_LCP
  if (i < (n-1)) checkPermutations (i+1,n,nC,nN,p,&pinv[0]);
# endif
#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif
}

void dLCP::transfer_i_to_N (int i)
{
  assert( i == (nC+nN) );
#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif

  if( findex != 0 && findex[i] >= 0 )
  {
    // In this case, need to update the factorization
    int normalIndex = inverse_permute( findex[ i ] );
    if( normalIndex < nC )
    {
        dReal alpha;
        if( state[ i ] == 0 )
            alpha = lo[ i ];
        else
            alpha = hi[ i ];

        std::vector<dReal> toAdd( this->nC );
        for( int k = 0; k < nC; ++k )
        {
            assert( i >= k );
            toAdd[ k ] = alpha*AROW( i )[k];
        }
        this->addToColumn( normalIndex, &toAdd[0] );
    }
  }

  // because we can assume C and N span 1:i-1
  nN++;

#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif
}

dReal dLCP::Aij (int i, int j) const
{
  if( i > j )
    return AROW(i)[j];
  else
    return AROW(j)[i];
}

dReal dLCP::effectiveLo( int index ) const
{
    if( findex && findex[index] >= 0 )
    {
        int normalIndex = inverse_permute( findex[index] );
        if( x[normalIndex] < 0 )
            return lo[index];
        else
            return lo[index] * x[normalIndex];
    }

    return lo[index];
}

dReal dLCP::effectiveHi( int index ) const
{
    if( findex && findex[index] >= 0 )
    {
        int normalIndex = inverse_permute( findex[index] );
        if( x[normalIndex] < 0 )
            return hi[index];
        else
            return hi[index] * x[normalIndex];
    }

    return hi[index];
}


void dLCP::transfer_i_from_N_to_C (int i)
{
#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif

  // Need to update any columns that this one affects
  if( findex != 0 && findex[i] >= 0 )
  {
    // In this case, need to update the factorization
    int normalIndex = inverse_permute( findex[ i ] );
    if( normalIndex < nC )
    {
        dReal alpha;
        if( state[ i ] == 0 )
            alpha = lo[ i ];
        else
            alpha = hi[ i ];

        std::vector<dReal> toAdd( this->nC );
        for( int k = 0; k < nC; ++k )
        {
            assert( i >= k );
            toAdd[ k ] = -alpha*AROW( i )[k];
        }
        this->addToColumn( normalIndex, &toAdd[0] );
    }
  }
    
  swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,nC,i,nskip,1);
  nN--;

  std::vector<dReal> newColumn( nC+1 );

  const dReal* aRow = AROW(nC);
  std::copy( aRow, aRow + (nC+1), newColumn.begin() );

  std::vector<dReal> newRow = newColumn;

  if( findex != 0 )
  {
    // In this case, may need to add to column
    for( int j = 0; j < nN; ++j )
    {
        if( findex[ nC+1+j ] < 0 )
            continue;

        int normalIndex = inverse_permute( findex[ nC+1+j ] );
        if( normalIndex <= nC )
        {
            dReal alpha;
            if( state[ nC+1+j ] == 0 )
                alpha = lo[ nC+1+j ];
            else
                alpha = hi[ nC+1+j ];

            newRow[ normalIndex ] += alpha*AROW( nC+1+j )[nC];

            if( normalIndex == nC )
            {
                for( int k = 0; k < nC+1; ++k )
                    newColumn[ k ] += alpha*AROW( nC+1+j )[k];
            }
        }
    }
  }

  if( this->redundantConstraint(&newRow[0], &newColumn[0]) )
  {
    swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,nC,nC+nN,nskip,1);
    transfer_i_to_N(nC+nN);
    state[nC+nN] = 2;
    return;
  }

  this->addRowAndColumn( &newRow[0], &newColumn[0] );

  nC++;

#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif
}

template <typename T>
void rotate1( T* first, int i, int n )
{
    std::rotate( first + i, first + i + 1, first + n );
}

// Sometimes, if we get stuck in a loop, we'll need to move a constraint
// to the end of the list.  
void dLCP::rotate_i_to_end( int i )
{
#ifdef DEBUG_LCP
    fortran_matrix A_unpermuted( n, n );
    for( int j = 0; j < n; ++j )
        for( int k = 0; k < n; ++k )
            A_unpermuted(p[j], p[k]) = this->Aij(j, k);
#endif

    rotate1( x, i, n );
    rotate1( b, i, n );
    rotate1( w, i, n );
    rotate1( lo, i, n );
    rotate1( hi, i, n );
    rotate1( p, i, n );
    rotate1( state, i, n );
    rotate1( findex, i, n );

    // fix pinv
    for( int j = i; j < n; ++j )
        pinv[ p[j] ] = j;

    // Now, just need to fix the matrix

    // the last row is going to be messed up because it 
    // needs to have every entry filled in
    for( int j = i + 1; j < n; ++j )
        A[i][j] = A[j][i];

    // Rotate each row
    for( int j = 0; j < n; ++j )
        rotate1( A[j], i, n );

    // Now rotate the rows
    rotate1( A, i, n );

    assert( i >= nC );
    if( i < (nC+nN) )
        --nN;

#ifdef DEBUG_LCP
    for( int j = 0; j < n; ++j )
        for( int k = 0; k < n; ++k )
            assert( A_unpermuted(p[j], p[k]) == this->Aij(j, k) );
#endif
}

void dLCP::transfer_i_from_N_to_end (int i)
{
#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif

    assert( i >= nC && i < n );

    int source = p[i];

    // We only need to update the factorization for i;
    // the others can only affect i so those columns will
    // not be affected.
    if( i < (nC+nN) && findex != 0 && findex[i] >= 0 )
    {
        // In this case, need to update the factorization
        int normalIndex = inverse_permute( findex[ i ] );
        if( normalIndex < nC )
        {
            dReal alpha;
            if( state[ i ] == 0 )
                alpha = lo[ i ];
            else
                alpha = hi[ i ];

            std::vector<dReal> toAdd( this->nC );
            for( int k = 0; k < nC; ++k )
            {
                assert( i >= k );
                toAdd[ k ] = -alpha*AROW( i )[k];
            }
            this->addToColumn( normalIndex, &toAdd[0] );
        }
    }

    rotate_i_to_end( i );

#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif

    // any constraints with findex == i need to be rotated after i
    for( int j = 0; j < nC; )
    {
        if( findex[j] == source )
            transfer_i_from_C_to_N(j);
        else
            ++j;
    }

    int last = n - 1;
    for( int j = nC; j < last; )
    {
        if( findex[j] == source )
        {
            --last;
            rotate_i_to_end( j );
        }
        else
            ++j;
    }

#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif
}

void dLCP::transfer_i_from_C_to_N (int i)
{
#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif

  this->removeRowAndColumn( i );

  // this could probably be faster
  for( int j = i; j < (nC-1); ++j )
    swapProblem (A,x,b,w,lo,hi,p,&pinv[0],state,findex,n,j,j+1,nskip,1);

  nC--;
  nN++;

  // Now, since we've transferred it into N, we need to update the
  // matrix with those changes as well:
  if( findex != 0 && findex[nC] >= 0 )
  {
    // In this case, need to update the factorization
    int normalIndex = inverse_permute( findex[ nC ] );
    if( normalIndex < nC )
    {
        dReal alpha;
        if( state[ nC ] == 0 )
            alpha = lo[ nC ];
        else
            alpha = hi[ nC ];

        std::vector<dReal> toAdd( this->nC );
        for( int k = 0; k < nC; ++k )
        {
            assert( nC >= k );
            toAdd[ k ] = alpha*AROW( nC )[k];
        }
        this->addToColumn( normalIndex, &toAdd[0] );
    }
  }

#ifdef DEBUG_FACTORIZATION
  this->checkFactorization();
#endif
}

dReal dLCP::Ai_times_q (int i, dReal *q) const
{
    dReal result = dDot( AROW(i), q, std::min(n, i+1) );
    for( int j = i+1; j < n; ++j )
        result += AROW(j)[i] * q[j];
    return result;
}

dReal dLCP::AiC_times_qC (int i, dReal *q) const
{
    dReal result = dDot( AROW(i), q, std::min(nC, i+1) );
    for( int j = i+1; j < nC; ++j )
        result += AROW(j)[i] * q[j];
    return result;
}

dReal dLCP::AiN_times_qN (int i, dReal *q) const
{
    dReal result = 0;
    if( i >= nC )
        result = dDot( AROW(i)+nC, q+nC, std::min(nN, i+1-nC) );
    for( int j = std::max(i+1, nC); j < nC+nN; ++j )
        result += AROW(j)[i] * q[j];
    return result;
}

void dLCP::pN_equals_ANC_times_qC (dReal *p, dReal *q) const
{
  // we could try to make this matrix-vector multiplication faster using
  // outer product matrix tricks, e.g. with the dMultidotX() functions.
  // but i tried it and it actually made things slower on random 100x100
  // problems because of the overhead involved. so we'll stick with the
  // simple method for now.
  for (int i=0; i<nN; i++) p[i+nC] = dDot (AROW(i+nC),q,nC);
}


void dLCP::pN_plusequals_ANi (dReal *p, int i, int sign) const
{
  dReal *aptr = AROW(i)+nC;
  if (sign > 0) {
    for (int i=0; i<nN; i++) p[i+nC] += aptr[i];
  }
  else {
    for (int i=0; i<nN; i++) p[i+nC] -= aptr[i];
  }
}

void dLCP::pN_plusequals_ANi_scaled (dReal *p, int i, dReal scale) const
{
  dReal *aptr = AROW(i);
  for( int j = nC; j <= std::min(nC+nN, i); ++j )
      p[j] += scale*aptr[j];

  for( int j = std::max(i+1, nC); j < nC+nN; ++j )
      p[j] += scale*AROW(j)[i];
}

void dLCP::checkInvariants() const
{
    for( int j = 0; j < (this->numC() + this->numN()); ++j )
    {
        const double epsilon = 1e-10;

        // Verify that A*x = w + b
        dReal dotProd = this->Ai_times_q(j, x);

        dReal diff = dotProd - b[j];
        CheckAndDump( withinEpsilon( diff, w[j], b[j] ) );

        // should verify that A*x - rhs = w

        dReal effectiveLow = this->effectiveLo(j);
        dReal effectiveHigh = this->effectiveHi(j);
        if( j < this->numC() )
        {
            CheckAndDump( withinEpsilon(w[j], 0) );
            CheckAndDump( x[j] >= (effectiveLow + std::min(-epsilon, epsilon*effectiveLow)) && 
                x[j] <= (effectiveHigh + std::max(epsilon, epsilon*effectiveHigh)) );
        }
        else if( j < this->numC() + this->numN() )
        {
            assert( state[j] >= 0 && state[j] <= 2 );
            if( state[j] == 0 )
                CheckAndDump( withinEpsilon( x[j], effectiveLow )  && w[j] > -epsilon );
            else if( state[j] == 1 )
                CheckAndDump( withinEpsilon( x[j], effectiveHigh ) && w[j] < epsilon );
            else
            {
                assert( state[j] == 2 );
                //CheckAndDump( std::abs(w[j]) < 1 );
            }
        }
        else
        {
            // still fixing it
            ;
        }
    }

#ifdef DEBUG_FACTORIZATION
    this->checkFactorization();
#endif
}

void dLCP::constructMatrix( dReal* matrix, size_t stride ) const
{
    for (int j=0; j<nC; j++)
        for( int k = 0; k <= j; ++k )
            matrix[ k*stride + j ] = matrix[ j*stride + k ] = this->Aij(j,k);

    // If an element i of N has a nonzero findex, then its value must
    // depend on the value of the corresponding findex[i].  This is 
    // not a problem if findex[i] is in N since it won't be changing,
    // but if findex[i] is in C then we need to take into account the
    // fact that changing lambda for findex[i] will also change lambda
    // for i.  This necessitates adding an extra term to the equation
    // as follows:
    if( findex != 0 )
    {
        for (int j=0; j<nN; ++j)
        {
            if( findex[ indexN(j) ] < 0 )
                continue;
            int normalIndex = inverse_permute( findex[ indexN(j) ] );
            if( normalIndex >= nC )
                continue;

            dReal alpha;
            if( state[ indexN(j) ] == 0 )
                alpha = lo[ indexN(j) ];
            else
                alpha = hi[ indexN(j) ];

            for(int k = 0; k < nC; ++k )
                matrix[ normalIndex*stride + k ] += alpha*AROW( indexN(j) )[k];
        }
    }
}

fortran_matrix QRproduct( const dReal* Q, const dReal* R, int nC, int stride )
{
    if( nC == 0 )
        return fortran_matrix();

    fortran_matrix QRprod( nC, nC );
    std::fill( QRprod.data(), QRprod.data() + nC*nC, 0 );

    // multiply out to see if A = Q*R
    for( size_t iCol = 0; iCol < nC; ++iCol )
    {
        char trans = 'N';
        int m = nC;
        int n = iCol+1;
        int lda = stride;
        dReal alpha = 1;
        dReal beta = 0;
        int incx = 1;
        int incy = 1;

        dgemv(
            &trans,
            &m,
            &n,
            &alpha,
            &Q[0],
            &lda,
            &R[0] + (iCol*(iCol+1))/2,
            &incx,
            &beta,
            QRprod.data() + iCol*nC,
            &incy );
    }

    return QRprod;
}


fortran_matrix dLCP::QRproduct() const
{
    return ::QRproduct( _Q.get(), _R.get(), this->nC, this->nC );
}

#ifdef DEBUG_FACTORIZATION
void dLCP::checkFactorization() const
{
    if( nC == 0 )
        return;

    fortran_matrix fullA( this->nC, this->nC );
    this->constructMatrix( fullA.data(), this->nC );

    fortran_matrix QRprod = this->QRproduct();

    /*
    {
        MATFile output( "QRdebug3.mat", "Testing" );
        output.add( "A", fullA );
        output.add( "QR", QRprod );
    }
    */

    for( size_t iCol = 0; iCol < nC; ++iCol )
        for( size_t iRow = 0; iRow < nC; ++iRow )
            assert( withinEpsilon( fullA(iRow, iCol), QRprod(iRow, iCol), fullA(iRow, iCol) ) );
}
#endif

void dLCP::solve1 (dReal *a, int i, int dir)
{
    int j;
    if (nC == 0)
       return;

    dReal *aptr = AROW(i);

    // Check if there are 0s on the diagonal
    dReal minElement = std::numeric_limits<dReal>::max();
    for( size_t i = 0; i < nC; ++i )
        minElement = std::min( minElement, std::abs(_R[ (i*(i+1))/2 + i ]) );

    std::vector<double> tmp( nC );
    if( minElement < 1e-7 )
    {
        // Solve Ax = b using SVD, rather than QR decomposition
        // @todo make this a last resort somehow
        std::vector<double> AA( nC*nC );
        constructMatrix( &AA[0], nC );


        std::vector<double> AA_before = AA;

        for (j=0; j<nC; j++) tmp[j] = aptr[j];

        int m = nC;
        int n = nC;
        int nrhs = 1;
        int lda = nC;
        int ldb = nC;
        double rcond = -1.0;
        int lwork = 5*nC;
        std::vector<double> work( lwork );
        std::vector<double> s( nC );
        int rank;
        int info;

        dgelss(
            &m,
            &n,
            &nrhs,
            &AA[0],
            &lda,
            &tmp[0],
            &ldb,
            &s[0],
            &rcond,
            &rank,
            &work[0],
            &lwork,
            &info );

        // @todo throw some kind of error here
        assert( info == 0 );
    }
    else
    {
#ifdef DEBUG_FACTORIZATION
        this->checkFactorization();

        // Solve Ax = b using SVD, rather than QR decomposition
        // @todo make this a last resort somehow
        std::vector<double> AA( nC*nC );
        constructMatrix( &AA[0], nC );


        std::vector<double> AA_before = AA;

        std::vector<double> tmp1( nC );
        for (j=0; j<nC; j++) tmp1[j] = aptr[j];

        int m = nC;
        int n = nC;
        int nrhs = 1;
        int lda = nC;
        int ldb = nC;
        double rcond = -1.0;
        int lwork = 5*nC;
        std::vector<double> work( lwork );
        std::vector<double> s( nC );
        int info;

        std::vector<int> ipiv( n );
        dgesv(
            &n,
            &nrhs,
            &AA[0],
            &lda,
            &ipiv[0],
            &tmp1[0],
            &ldb,
            &info );

        // @todo throw some kind of error here
        assert( info == 0 );
#endif

        this->solve( aptr, &tmp[0] );

#ifdef DEBUG_FACTORIZATION
        for( size_t i = 0; i < tmp1.size(); ++i )
            assert( withinEpsilon(tmp[i], tmp1[i], tmp[i]) );
#endif
    }

    if (dir > 0) 
    {
        for (j=0; j<nC; j++) a[j] = -tmp[j];
    }
    else 
    {
        for (j=0; j<nC; j++) a[j] = tmp[j];
    }

    //checkVector( a, nC );
}


void dLCP::unpermute()
{
  // now we have to un-permute x and w
  int j;
  ALLOCA (dReal,tmp,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (tmp == NULL) {
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  memcpy (tmp,x,n*sizeof(dReal));
  for (j=0; j<n; j++) x[p[j]] = tmp[j];
  memcpy (tmp,w,n*sizeof(dReal));
  for (j=0; j<n; j++) w[p[j]] = tmp[j];

  UNALLOCA (tmp);
}

//***************************************************************************
// an unoptimized Dantzig LCP driver routine for the basic LCP problem.
// must have lo=0, hi=dInfinity, and nub=0.

void dSolveLCPBasic (int n, dReal *A, dReal *x, dReal *b,
		     dReal *w, int nub, dReal *lo, dReal *hi)
{
  dAASSERT (n>0 && A && x && b && w && nub == 0);

  int i,k;
  int nskip = dPAD(n);
  ALLOCA (dReal,delta_x,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (delta_x == NULL) {
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,delta_w,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (delta_w == NULL) {
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal*,Arows,n*sizeof(dReal*));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (Arows == NULL) {
      UNALLOCA(delta_w);
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (int,p,n*sizeof(int));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (p == NULL) {
      UNALLOCA(Arows);
      UNALLOCA(delta_w);
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (int,dummy,n*sizeof(int));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (dummy == NULL) {
      UNALLOCA(p);
      UNALLOCA(Arows);
      UNALLOCA(delta_w);
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif


  std::vector<double> tmp(n);
  dLCP lcp (n,0,A,x,b,w,&tmp[0],&tmp[0],dummy,dummy,p,Arows);
  nub = lcp.getNub();

  for (i=0; i<n; i++) {
    w[i] = lcp.AiC_times_qC (i,x) - b[i];
    if (w[i] >= 0) {
      lcp.transfer_i_to_N (i);
    }
    else {
      for (;;) {
	// compute: delta_x(C) = -A(C,C)\A(C,i)
	dSetZero (delta_x,n);
	lcp.solve1 (delta_x,i);
    checkVector( delta_x, n );
#ifdef dUSE_MALLOC_FOR_ALLOCA
	if (dMemoryFlag == d_MEMORY_OUT_OF_MEMORY) {
	  UNALLOCA(dummy);
	  UNALLOCA(p);
	  UNALLOCA(Arows);
	  UNALLOCA(delta_w);
	  UNALLOCA(delta_x);
	  return;
	}
#endif
	delta_x[i] = 1;

	// compute: delta_w = A*delta_x
	dSetZero (delta_w,n);
	lcp.pN_equals_ANC_times_qC (delta_w,delta_x);
	lcp.pN_plusequals_ANi (delta_w,i);
        delta_w[i] = lcp.AiC_times_qC (i,delta_x) + lcp.Aii(i);

	// find index to switch
	int si = i;		// si = switch index
	int si_in_N = 0;	// set to 1 if si in N
	dReal s = -w[i]/delta_w[i];

	if (s <= 0) {
	  dMessage (d_ERR_LCP, "LCP internal error, s <= 0 (s=%.4e)",s);
	  if (i < (n-1)) {
	    dSetZero (x+i,n-i);
	    dSetZero (w+i,n-i);
	  }
	  goto done;
	}

	for (k=0; k < lcp.numN(); k++) {
	  if (delta_w[lcp.indexN(k)] < 0) {
	    dReal s2 = -w[lcp.indexN(k)] / delta_w[lcp.indexN(k)];
	    if (s2 < s) {
	      s = s2;
	      si = lcp.indexN(k);
	      si_in_N = 1;
	    }
	  }
	}
	for (k=0; k < lcp.numC(); k++) {
	  if (delta_x[lcp.indexC(k)] < 0) {
	    dReal s2 = -x[lcp.indexC(k)] / delta_x[lcp.indexC(k)];
	    if (s2 < s) {
	      s = s2;
	      si = lcp.indexC(k);
	      si_in_N = 0;
	    }
	  }
	}

	// apply x = x + s * delta_x
	lcp.pC_plusequals_s_times_qC (x,s,delta_x);
	x[i] += s;
	lcp.pN_plusequals_s_times_qN (w,s,delta_w);
	w[i] += s * delta_w[i];

	// switch indexes between sets if necessary
	if (si==i) {
	  w[i] = 0;
	  lcp.transfer_i_to_C (i);
	  break;
	}
	if (si_in_N) {
          w[si] = 0;
	  lcp.transfer_i_from_N_to_C (si);
	}
	else {
          x[si] = 0;
	  lcp.transfer_i_from_C_to_N (si);
	}
      }
    }
  }

 done:
  lcp.unpermute();

  UNALLOCA (delta_x);
  UNALLOCA (delta_w);
  UNALLOCA (Arows);
  UNALLOCA (p);
  UNALLOCA (dummy);
}

//***************************************************************************
// an optimized Dantzig LCP driver routine for the lo-hi LCP problem.

void dSolveLCP (int n, dReal *A, dReal *x, dReal *b,
		dReal *w, int nub, dReal *lo, dReal *hi, int *findex)
{
  dAASSERT (n>0 && A && x && b && w && lo && hi && nub >= 0 && nub <= n);
  int nskip = dPAD(n);

  checkVector( b, n );

  int i,k,hit_first_friction_index = 0;

  // if all the variables are unbounded then we can just factor, solve,
  // and return
  if (nub >= n) {
    dFactorLDLT (A,w,n,nskip);		// use w for d
    dSolveLDLT (A,w,b,n,nskip);
    memcpy (x,b,n*sizeof(dReal));
    dSetZero (w,n);

    return;
  }
# ifndef dNODEBUG
  // check restrictions on lo and hi
  for (k=0; k<n; k++) dIASSERT (lo[k] <= 0 && hi[k] >= 0);
# endif
  ALLOCA (dReal,delta_x,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (delta_x == NULL) {
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,delta_w,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (delta_w == NULL) {
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal*,Arows,n*sizeof(dReal*));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (Arows == NULL) {
      UNALLOCA(delta_w);
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (int,p,n*sizeof(int));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (p == NULL) {
      UNALLOCA(Arows);
      UNALLOCA(delta_w);
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif

    std::vector<int> rotatedToEnd( n, 0 );

  int dir;
  dReal dirf;

  // for i in N, state[i] is 0 if x(i)==lo(i) or 1 if x(i)==hi(i)
  ALLOCA (int,state,n*sizeof(int));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (state == NULL) {
      UNALLOCA(p);
      UNALLOCA(Arows);
      UNALLOCA(delta_w);
      UNALLOCA(delta_x);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif

  // create LCP object. note that tmp is set to delta_w to save space, this
  // optimization relies on knowledge of how tmp is used, so be careful!
  dLCP *lcp=new dLCP(n,nub,A,x,b,w,lo,hi,state,findex,p,Arows);
  nub = lcp->getNub();

  // loop over all indexes nub..n-1. for index i, if x(i),w(i) satisfy the
  // LCP conditions then i is added to the appropriate index set. otherwise
  // x(i),w(i) is driven either +ve or -ve to force it to the valid region.
  // as we drive x(i), x(C) is also adjusted to keep w(C) at zero.
  // while driving x(i) we maintain the LCP conditions on the other variables
  // 0..i-1. we do this by watching out for other x(i),w(i) values going
  // outside the valid region, and then switching them between index sets
  // when that happens.

  const dReal epsilon = 1e-10;

  for (i=nub; i<n; i++) {
    // the index i is the driving index and indexes i+1..n-1 are "dont care",
    // i.e. when we make changes to the system those x's will be zero and we
    // don't care what happens to those w's. in other words, we only consider
    // an (i+1)*(i+1) sub-problem of A*x=b+w.

    // if we've hit the first friction index, we have to compute the lo and
    // hi values based on the values of x already computed. we have been
    // permuting the indexes, so the values stored in the findex vector are
    // no longer valid. thus we have to temporarily unpermute the x vector. 
    // for the purposes of this computation, 0*infinity = 0 ... so if the
    // contact constraint's normal force is 0, there should be no tangential
    // force applied.

      /*
    if (hit_first_friction_index == 0 && findex && findex[i] >= 0) {
      // un-permute x into delta_w, which is not being used at the moment
      for (k=0; k<n; k++) delta_w[p[k]] = x[k];

      // set lo and hi values
      for (k=i; k<n; k++) {
	dReal wfk = delta_w[findex[k]];
	if (wfk == 0) {
	  hi[k] = 0;
	  lo[k] = 0;
	}
	else {
	  hi[k] = dFabs (hi[k] * wfk);
	  lo[k] = -hi[k];
	}
      }
      hit_first_friction_index = 1;
    }
    */

    // thus far we have not even been computing the w values for indexes
    // greater than i, so compute w[i] now.
    w[i] = lcp->Ai_times_q (i,x) - b[i];

    // if lo=hi=0 (which can happen for tangential friction when normals are
    // 0) then the index will be assigned to set N with some state. however,
    // set C's line has zero size, so the index will always remain in set N.
    // with the "normal" switching logic, if w changed sign then the index
    // would have to switch to set C and then back to set N with an inverted
    // state. this is pointless, and also computationally expensive. to
    // prevent this from happening, we use the rule that indexes with lo=hi=0
    // will never be checked for set changes. this means that the state for
    // these indexes may be incorrect, but that doesn't matter.

    // see if x(i),w(i) is in a valid region
    if (x[i] == lcp->effectiveLo(i) && w[i] >= 0) {
      state[i] = 0;
      lcp->transfer_i_to_N (i);
      lcp->checkInvariants();
    }
    else if (x[i] == lcp->effectiveHi(i) && w[i] <= 0) {
      state[i] = 1;
      lcp->transfer_i_to_N (i);
      lcp->checkInvariants();
    }
    else {
      int prevCmd = -1;
      int prevIndex = -1;

      int iIter = 0;
      // we must push x(i) and w(i)
      for (;;)
      {
        if ( std::abs(w[i]) <= epsilon && 
              (lcp->effectiveLo(i)-epsilon) <= x[i] && 
              (lcp->effectiveHi(i)+epsilon) >= x[i] )
        {
          // this is a degenerate case. by the time we get to this test we know
          // that lo != 0, which means that lo < 0 as lo is not allowed to be +ve,
          // and similarly that hi > 0. this means that the line segment
          // corresponding to set C is at least finite in extent, and we are on it.
          // NOTE: we must call lcp->solve1() before lcp->transfer_i_to_C()

#ifdef dUSE_MALLOC_FOR_ALLOCA
          if (dMemoryFlag == d_MEMORY_OUT_OF_MEMORY) {
	        UNALLOCA(state);
	        UNALLOCA(p);
	        UNALLOCA(Arows);
	        UNALLOCA(delta_w);
	        UNALLOCA(delta_x);
	        return;
          }
#endif

          lcp->transfer_i_to_C (i);
          lcp->checkInvariants();

          break;
        }

          ++iIter;
	// find direction to push on x(i)
    dReal effLo = lcp->effectiveLo(i);
    dReal effHi = lcp->effectiveHi(i);
    if( x[i] > effHi )
    {
      dir = -1;
      dirf = REAL(-1.0);
    }
    else if( x[i] < effLo )
    {
      dir = 1;
      dirf = REAL(1.0);
    }
	else if (w[i] <= 0) {
	  dir = 1;
	  dirf = REAL(1.0);
	}
	else {
	  dir = -1;
	  dirf = REAL(-1.0);
	}

	// compute: delta_x(C) = -dir*A(C,C)\A(C,i)
#ifdef DEBUG_FACTORIZATION
    lcp->checkFactorization();
#endif
	lcp->solve1 (delta_x,i,dir);

#ifdef dUSE_MALLOC_FOR_ALLOCA
	if (dMemoryFlag == d_MEMORY_OUT_OF_MEMORY) {
	  UNALLOCA(state);
	  UNALLOCA(p);
	  UNALLOCA(Arows);
	  UNALLOCA(delta_w);
	  UNALLOCA(delta_x);
	  return;
	}
#endif

	// note that delta_x[i] = dirf, but we wont bother to set it

	// compute: delta_w = A*delta_x ... note we only care about
        // delta_w(N) and delta_w(i), the rest is ignored
	lcp->pN_equals_ANC_times_qC (delta_w,delta_x);
	lcp->pN_plusequals_ANi (delta_w,i,dir);
    delta_w[i] = lcp->AiC_times_qC (i,delta_x) + lcp->Aii(i)*dirf;

    // for any friction (findex) entries in NC, we need to maintain the friction
    // constraint that f_i = mu * f_N[ findex[i] ]
    for( int j = 0; j < lcp->numN(); ++j )
    {
        if( findex == 0 || findex[ lcp->indexN(j) ] < 0 )
        {
            delta_x[ lcp->indexN(j) ] = 0.0;
            continue;
        }

        int normalIndex = lcp->inverse_permute( findex[ lcp->indexN(j) ] );

        // if it's in the not-constrained set, then the change in
        // normal force is 0 which means the change in friction is 0 too
        if( normalIndex >= lcp->numC() )
        {
            delta_x[ lcp->indexN(j) ] = 0.0;
            continue;
        }

        if( state[ lcp->indexN(j) ] == 0 )
            delta_x[ lcp->indexN(j) ] = lo[ lcp->indexN(j) ] * delta_x[normalIndex];
        else
            delta_x[ lcp->indexN(j) ] = hi[ lcp->indexN(j) ] * delta_x[normalIndex];

        lcp->pN_plusequals_ANi_scaled(delta_w, lcp->indexN(j), delta_x[ lcp->indexN(j) ] );
        delta_w[i] += delta_x[ lcp->indexN(j) ] * lcp->Aij( lcp->indexN(j), i );
    }


	// find largest step we can take (size=s), either to drive x(i),w(i)
	// to the valid LCP region or to drive an already-valid variable
	// outside the valid region.

	int cmd = 1;		// index switching command
	int si = 0;		// si = index to switch if cmd>3

    dReal s = dInfinity;
    int normalIndex = -1;
    if( findex != 0 && findex[i] >= 0 )
        normalIndex = lcp->inverse_permute( findex[i] );
    if( x[normalIndex] < 0 )
        normalIndex = -1;

    if( std::abs(delta_w[i]) < epsilon && rotatedToEnd[ p[i] ] > 0 )
    {
        // transfer to "redundant constraints" set
        // We will only do this after we've tried to rotate it to the end once
        s = 0;
        cmd = 8;
    }
    else if( /* std::abs(delta_w[i]) < epsilon ||*/ 
             (delta_w[i] < 0 && dirf > 0) || (delta_w[i] > 0 && dirf < 0) )
    {
        s = 0;
        cmd = 7;        
    }
    else
    {
	    dReal s2 = -w[i]/delta_w[i];

        // resulting x
        dReal x2 = x[i] + s2*dirf;

        // check whether resulting x is in valid range
        dReal resLo = -dInfinity, resHi = dInfinity;
        if( normalIndex >= 0 )
        {
            dReal resNorm = x[normalIndex] + s2*delta_x[normalIndex];
            if( resNorm >= 0 )
            {
                resLo = lo[i]*resNorm - epsilon;
                resHi = hi[i]*resNorm + epsilon;
            }
        }
        else
        {
            resLo = lo[i] - epsilon;
            resHi = hi[i] + epsilon;
        }

        if( s2 >= 0 )
        {
            if( x2 >= resLo-epsilon && x2 <= resHi+epsilon )
                s = s2;
        }
        else if( std::abs(delta_w[i]) < epsilon && std::abs(w[i]) < epsilon )
        {
            // This does happen occasionally; in this case, we have conflicting
            // constraints and so some other constraint in the system is cancelling
            // us out.  We can either set s = infinity and let the other constraint
            // prevent us from walking off the edge or we can just assume that
            // we'll never satisfy the constraint completely and move our index i
            // into C as-is.
            s = 0;
        }
        else if( x[i] <= effHi && x[i] >= effLo )
        {
            // something has gone badly wrong here
            s = 0;
            cmd = 7;
        }
    }

    lcp->CheckAndDump( s >= 0 );

    if (hi[i] < dInfinity)
    {
        dReal s2;
        if( normalIndex < 0 )
        {
          s2 = (hi[i]-x[i])/dirf;		// step to x(i)=hi(i)
        }
        else
        {
          // We have to be careful here; the actual max. s that will
          // be allowed is related to how x[ findex[i] ] changes as well.
          s2 = (hi[i]*x[normalIndex] - x[i]) / (dirf - hi[i]*delta_x[normalIndex]);
        }

        // the w_res test here makes sure we aren't missing it, just in
        // case we're coming from the outside
        dReal w_res = w[i] + s2*delta_w[i];
        if( s2 >= 0 && s2 < s && w_res <= 0 )
        {
	      s = s2;
	      cmd = 3;
        }
	}

    if (lo[i] > -dInfinity)
    {
        dReal s2;
        if( normalIndex < 0 )
        {
          s2 = (lo[i]-x[i])/dirf;		// step to x(i)=hi(i)
        }
        else
        {
          // We have to be careful here; the actual max. s that will
          // be allowed is related to how x[ findex[i] ] changes as well.
          s2 = (lo[i]*x[normalIndex] - x[i]) / (dirf - lo[i]*delta_x[normalIndex]);
        }

        dReal w_res = w[i] + s2*delta_w[i];
        if( s2 >= 0 && s2 < s && w_res >= 0 )
        {
	      s = s2;
	      cmd = 2;
        }
    }

    lcp->CheckAndDump( s < dInfinity );

	for (k=0; k < lcp->numN(); k++) {
      if(state[lcp->indexN(k)] == 2 )
          continue;

      int index = lcp->indexN(k);
	  if ((state[lcp->indexN(k)]==0 && delta_w[lcp->indexN(k)] < 0) ||
	      (state[lcp->indexN(k)]!=0 && delta_w[lcp->indexN(k)] > 0)) {
	    // don't bother checking if lo=hi=0
	    if (lo[lcp->indexN(k)] == 0 && hi[lcp->indexN(k)] == 0) continue;
	    dReal s2 = -w[lcp->indexN(k)] / delta_w[lcp->indexN(k)];

        lcp->CheckAndDump( s2 >= 0 );
	    if (s2 >= 0 && s2 < s) {
	      s = s2;
	      cmd = 4;
	      si = lcp->indexN(k);
	    }
	  }
	}

	for (k=nub; k < lcp->numC(); k++) {
      int normalIndex = -1;
      if( findex != 0 && findex[lcp->indexC(k)] >= 0 )
          normalIndex = lcp->inverse_permute( findex[lcp->indexC(k)] );

      // We are going to have a problem with friction if x is negative
      if( normalIndex >= 0 && x[normalIndex] >= 0 )
      {
        dReal s2;
        if( lo[lcp->indexC(k)]*x[normalIndex] > x[lcp->indexC(k)] )
            s2 = 0;
        else
            s2 = (lo[lcp->indexC(k)]*x[normalIndex] - x[lcp->indexC(k)]) / 
                 (delta_x[lcp->indexC(k)] - lo[lcp->indexC(k)]*delta_x[normalIndex]);
        if( s2 >= 0 && s2 < s )
        {
	      s = s2;
	      cmd = 5;
	      si = lcp->indexC(k);
        }

        if( hi[lcp->indexC(k)]*x[normalIndex] < x[lcp->indexC(k)] )
            s2 = 0;
        else
            s2 = (hi[lcp->indexC(k)]*x[normalIndex] - x[lcp->indexC(k)]) / 
                 (delta_x[lcp->indexC(k)] - hi[lcp->indexC(k)]*delta_x[normalIndex]);
        if( s2 >= 0 && s2 < s )
        {
	      s = s2;
	      cmd = 6;
	      si = lcp->indexC(k);
        }

        continue;
      }

	  if (delta_x[lcp->indexC(k)] < 0 && lo[lcp->indexC(k)] > -dInfinity) {
        dReal s2 = (lo[lcp->indexC(k)]-x[lcp->indexC(k)]) / 
            delta_x[lcp->indexC(k)];		// step to x(i)=hi(i)
          
        lcp->CheckAndDump( s2 >= 0 );
	    if (s2 < s) {
	      s = s2;
	      cmd = 5;
	      si = lcp->indexC(k);
	    }
	  }

	  if (delta_x[lcp->indexC(k)] > 0 && hi[lcp->indexC(k)] < dInfinity) {
        dReal s2 = (hi[lcp->indexC(k)]-x[lcp->indexC(k)]) / 
              delta_x[lcp->indexC(k)];		// step to x(i)=hi(i)

        lcp->CheckAndDump( s2 >= 0 );
	    if (s2 < s) {
	      s = s2;
	      cmd = 6;
	      si = lcp->indexC(k);
	    }
	  }
	}

	//static char* cmdstring[8] = {0,"->C","->NL","->NH","N->C",
	//			     "C->NL","C->NH"};
	//printf ("cmd=%d (%s), si=%d\n",cmd,cmdstring[cmd],(cmd>3) ? si : i);

	// if s <= 0 then we've got a problem. if we just keep going then
	// we're going to get stuck in an infinite loop. instead, just cross
	// our fingers and exit with the current solution.
	if (s < 0) {
	  dMessage (d_ERR_LCP, "LCP internal error, s <= 0 (s=%.4e)",s);
      lcp->CheckAndDump( false );
	  if (i < (n-1)) {
	    dSetZero (x+i,n-i);
	    dSetZero (w+i,n-i);
	  }
	  goto done;
	}

	// apply x = x + s * delta_x
    for( int j = 0; j < i; ++j )
        x[j] += s*delta_x[j];
	x[i] += s * dirf;

	// apply w = w + s * delta_w
	lcp->pN_plusequals_s_times_qN (w,s,delta_w);
	w[i] += s * delta_w[i];

    // These might be violated due to precision loss
    for( int j = 0; j < lcp->numN(); ++j )
    {
        int index = lcp->indexN(j);
        if( (state[index] == 0 && w[index] < 0) ||
            (state[index] == 1 && w[index] > 0) )
        {
            lcp->CheckAndDump( std::abs(w[index]) < epsilon );
            w[index] = 0;
        }
    }

    /*
#ifdef DEBUG_LCP
    {
        MATFile dump( "lcpDump.mat", "LCP info" );

        fortran_matrix A( n, n );
        for( int i1 = 0; i1 < n; ++i1 )
            for( int j1 = 0; j1 < n; ++j1 )
                A( i1, j1 ) = lcp->Aij(i1, j1);

        dump.add( "A", A );

        dump.add( "x", x, n );
        dump.add( "w", w, n );

        std::vector<dReal> delta_x_tmp( n, 0 );
        std::copy( delta_x, delta_x + i, delta_x_tmp.begin() );
        delta_x_tmp[i] = dirf;

        std::vector<dReal> delta_w_tmp( n, 0 );
        std::copy( delta_w + lcp->numC(), delta_w + lcp->numC() + lcp->numN() + 1, delta_w_tmp.begin() + lcp->numC() );

        dump.add( "delta_x", delta_x_tmp );
        dump.add( "delta_w", delta_w_tmp );
        dump.add( "p", p, n );
        dump.add( "s", s );

        dump.add( "b", b, n );
        dump.add( "lo", lo, n );
        dump.add( "hi", hi, n );
        dump.add( "findex", findex, n );
    }
#endif
    */

	// switch indexes between sets if necessary
	switch (cmd) {
	case 1:		// done
      lcp->CheckAndDump( withinEpsilon(w[i], 0) );
      lcp->CheckAndDump( x[i] > lcp->effectiveLo(i)-epsilon );
      lcp->CheckAndDump( x[i] < lcp->effectiveHi(i)+epsilon );
	  w[i] = 0;
	  lcp->transfer_i_to_C (i);
	  break;
	case 2:		// done
      lcp->CheckAndDump( withinEpsilon(x[i], lcp->effectiveLo(i)) );
	  x[i] = lcp->effectiveLo(i);
	  state[i] = 0;
	  lcp->transfer_i_to_N (i);
	  break;
	case 3:		// done
      lcp->CheckAndDump( withinEpsilon(x[i], lcp->effectiveHi(i)) );
	  x[i] = lcp->effectiveHi(i);
	  state[i] = 1;
	  lcp->transfer_i_to_N (i);
	  break;
	case 4:		// keep going
      if( iIter > 20 ||
          ((prevCmd == 5 || prevCmd == 6) && prevIndex == p[si]) )
      {
          // don't want to loop forever!
          rotatedToEnd[ p[si] ]++;
          if( rotatedToEnd[ p[si] ] >= 4 )
              goto done;

          lcp->transfer_i_from_N_to_end(si);
          i = lcp->numN() + lcp->numC();
          w[i] = lcp->Ai_times_q (i,x) - b[i];
      }
      else
      {
          lcp->CheckAndDump( withinEpsilon(w[si], 0) );
    	  w[si] = 0;
          lcp->transfer_i_from_N_to_C (si);
      }
	  break;
	case 5:		// keep going
      lcp->CheckAndDump( withinEpsilon(x[si], lcp->effectiveLo(si)) );
	  x[si] = lcp->effectiveLo(si);
      if( w[si] < 0 )
      {
          lcp->CheckAndDump( std::abs(w[si]) < epsilon );
          w[si] = 0;
      }
	  state[si] = 0;
	  lcp->transfer_i_from_C_to_N (si);
	  break;
	case 6:		// keep going
      lcp->CheckAndDump( withinEpsilon(x[si], lcp->effectiveHi(si)) );
	  x[si] = lcp->effectiveHi(si);
      if( w[si] > 0 )
      {
          lcp->CheckAndDump( std::abs(w[si]) < epsilon );
          w[si] = 0;
      }
	  state[si] = 1;
	  lcp->transfer_i_from_C_to_N (si);
	  break;
    case 7:
          rotatedToEnd[ p[i] ]++;
          if( rotatedToEnd[ p[i] ] >= 4 )
          {
              lcp->verifyCondition( false, "failure to converge" );
              goto done;
          }

          lcp->transfer_i_from_N_to_end(i);
          i = lcp->numN() + lcp->numC();
          w[i] = lcp->Ai_times_q (i,x) - b[i];
          break;
    case 8:
        state[i] = 2;
        lcp->transfer_i_to_N (i);
        break;
	}

    prevCmd = cmd;
    prevIndex = p[si];

#ifdef DEBUG_LCP
    // verify that the constraints are satisfied for everything
    // up to and including the current i:
    if( cmd <= 3 || cmd == 8 )
        lcp->CheckAndDump( (lcp->numC() + lcp->numN()) == (i+1) );
    else
        lcp->CheckAndDump( (lcp->numC() + lcp->numN()) == i );

    lcp->checkInvariants();

#endif // DEBUG_LCP

    if (cmd <= 3 || cmd == 8) break;
      }
    }
  }

 done:
  lcp->unpermute();
  delete lcp;

  UNALLOCA (delta_x);
  UNALLOCA (delta_w);
  UNALLOCA (Arows);
  UNALLOCA (p);
  UNALLOCA (state);
}

//***************************************************************************
// accuracy and timing test

extern "C" ODE_API void dTestSolveLCP()
{
  int n = 100;
  int i,nskip = dPAD(n);
#ifdef dDOUBLE
  const dReal tol = REAL(1e-9);
#endif
#ifdef dSINGLE
  const dReal tol = REAL(1e-4);
#endif
  printf ("dTestSolveLCP()\n");

  ALLOCA (dReal,A,n*nskip*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (A == NULL) {
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,x,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (x == NULL) {
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,b,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (b == NULL) {
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,w,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (w == NULL) {
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,lo,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (lo == NULL) {
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,hi,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (hi == NULL) {
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif

  ALLOCA (dReal,A2,n*nskip*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (A2 == NULL) {
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,b2,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (b2 == NULL) {
      UNALLOCA (A2);
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,lo2,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (lo2 == NULL) {
      UNALLOCA (b2);
      UNALLOCA (A2);
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,hi2,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (hi2 == NULL) {
      UNALLOCA (lo2);
      UNALLOCA (b2);
      UNALLOCA (A2);
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,tmp1,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (tmp1 == NULL) {
      UNALLOCA (hi2);
      UNALLOCA (lo2);
      UNALLOCA (b2);
      UNALLOCA (A2);
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif
  ALLOCA (dReal,tmp2,n*sizeof(dReal));
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (tmp2 == NULL) {
      UNALLOCA (tmp1);
      UNALLOCA (hi2);
      UNALLOCA (lo2);
      UNALLOCA (b2);
      UNALLOCA (A2);
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      dMemoryFlag = d_MEMORY_OUT_OF_MEMORY;
      return;
    }
#endif

  double total_time = 0;
  for (int count=0; count < 1000; count++) {

    // form (A,b) = a random positive definite LCP problem
    dMakeRandomMatrix (A2,n,n,1.0);
    dMultiply2 (A,A2,A2,n,n,n);
    dMakeRandomMatrix (x,n,1,1.0);
    dMultiply0 (b,A,x,n,n,1);
    for (i=0; i<n; i++) b[i] += (dRandReal()*REAL(0.2))-REAL(0.1);

    // choose `nub' in the range 0..n-1
    int nub = 50; //dRandInt (n);

    // make limits
    for (i=0; i<nub; i++) lo[i] = -dInfinity;
    for (i=0; i<nub; i++) hi[i] = dInfinity;
    //for (i=nub; i<n; i++) lo[i] = 0;
    //for (i=nub; i<n; i++) hi[i] = dInfinity;
    //for (i=nub; i<n; i++) lo[i] = -dInfinity;
    //for (i=nub; i<n; i++) hi[i] = 0;
    for (i=nub; i<n; i++) lo[i] = -(dRandReal()*REAL(1.0))-REAL(0.01);
    for (i=nub; i<n; i++) hi[i] =  (dRandReal()*REAL(1.0))+REAL(0.01);

    // set a few limits to lo=hi=0
    /*
    for (i=0; i<10; i++) {
      int j = dRandInt (n-nub) + nub;
      lo[j] = 0;
      hi[j] = 0;
    }
    */

    // solve the LCP. we must make copy of A,b,lo,hi (A2,b2,lo2,hi2) for
    // SolveLCP() to permute. also, we'll clear the upper triangle of A2 to
    // ensure that it doesn't get referenced (if it does, the answer will be
    // wrong).

    memcpy (A2,A,n*nskip*sizeof(dReal));
    dClearUpperTriangle (A2,n);
    memcpy (b2,b,n*sizeof(dReal));
    memcpy (lo2,lo,n*sizeof(dReal));
    memcpy (hi2,hi,n*sizeof(dReal));
    dSetZero (x,n);
    dSetZero (w,n);

    dStopwatch sw;
    dStopwatchReset (&sw);
    dStopwatchStart (&sw);

    dSolveLCP (n,A2,x,b2,w,nub,lo2,hi2,0);
#ifdef dUSE_MALLOC_FOR_ALLOCA
    if (dMemoryFlag == d_MEMORY_OUT_OF_MEMORY) {
      UNALLOCA (tmp2);
      UNALLOCA (tmp1);
      UNALLOCA (hi2);
      UNALLOCA (lo2);
      UNALLOCA (b2);
      UNALLOCA (A2);
      UNALLOCA (hi);
      UNALLOCA (lo);
      UNALLOCA (w);
      UNALLOCA (b);
      UNALLOCA (x);
      UNALLOCA (A);
      return;
    }
#endif

    dStopwatchStop (&sw);
    double time = dStopwatchTime(&sw);
    total_time += time;
    double average = total_time / double(count+1) * 1000.0;

    // check the solution

    dMultiply0 (tmp1,A,x,n,n,1);
    for (i=0; i<n; i++) tmp2[i] = b[i] + w[i];
    dReal diff = dMaxDifference (tmp1,tmp2,n,1);
    // printf ("\tA*x = b+w, maximum difference = %.6e - %s (1)\n",diff,
    //	    diff > tol ? "FAILED" : "passed");
    if (diff > tol) dDebug (0,"A*x = b+w, maximum difference = %.6e",diff);
    int n1=0,n2=0,n3=0;
    for (i=0; i<n; i++) {
      if (x[i]==lo[i] && w[i] >= 0) {
	n1++;	// ok
      }
      else if (x[i]==hi[i] && w[i] <= 0) {
	n2++;	// ok
      }
      else if (x[i] >= lo[i] && x[i] <= hi[i] && w[i] == 0) {
	n3++;	// ok
      }
      else {
	dDebug (0,"FAILED: i=%d x=%.4e w=%.4e lo=%.4e hi=%.4e",i,
		x[i],w[i],lo[i],hi[i]);
      }
    }

    // pacifier
    printf ("passed: NL=%3d NH=%3d C=%3d   ",n1,n2,n3);
    printf ("time=%10.3f ms  avg=%10.4f\n",time * 1000.0,average);
  }

  UNALLOCA (A);
  UNALLOCA (x);
  UNALLOCA (b);
  UNALLOCA (w);
  UNALLOCA (lo);
  UNALLOCA (hi);
  UNALLOCA (A2);
  UNALLOCA (b2);
  UNALLOCA (lo2);
  UNALLOCA (hi2);
  UNALLOCA (tmp1);
  UNALLOCA (tmp2);
}

/*
// This is inefficient, since each time we duplicate A into a separate
// array so that we can perform a LU decomposition.  More efficient
// would be to allow updates of the LU decomposition, but I will worry
// about that later.
void solve( const double* A, int n, int lda, const double* b, double* x )
{
    // LAPACK won't like this:
    if( n == 0 )
        return;

    std::vector<double> A_temp( n*n );
    // Copy it one column at a time
    for( size_t i = 0; i < n; ++i )
        std::copy( A + i*lda, A + i*lda + n, A_temp.begin() + i*n );

    std::copy( b, b+n, x );

    int nrhs = 1;
    int lda = n;
    int ldb = 1;
    int info;
    std::vector<int> ipiv( n );
    dgesv( &n, &nrhs, &A_temp[0], &lda, &ipiv[0], x, &ldb, &info );

    // negative info indicates broken parameter
    assert( info >= 0 );

    assert( info == 0 );
    if( info > 0 )
    {
        // Do something useful here
        std::cout << "Matrix is singular!\n";
    }
}

void dSolveLCP_cdtwigg (int n, dReal *A, dReal *x, dReal *b, dReal *w,
		int nub, dReal *lo, dReal *hi, int *findex)
{
    // Permutation vector will keep track of the permutation we make
    // to slot DOFs in C into the first k rows of the matrix
    std::vector<size_t> p( n );
    for( size_t i = 0; i < n; ++i )
        p[i] = i;

    // We will do everything in double precision to make LAPACK happy
    std::vector<double> p_A( A, A + n*n );
    std::vector<double> p_x( x, x + n );
    std::vector<double> p_b( b, b + n );
    std::vector<double> p_w( w, w + n );

    std::vector<double> p_lo( lo, lo + n );
    std::vector<double> p_hi( hi, hi + n );

    for( size_t i = 0; i < n; ++i )
        assert( p_lo[i] <= 0.0 && p_hi[i] >= 0.0 );

    // We will keep the clamped variables in the first nC spots in
    // system.  On entry, this is exactly the set of variables for 
    // which lo == -infty and hi == infty.
    size_t nC = nub;

    // We will initialize p_w by doing a full matrix solve for the
    // set of initially Clamped variables
    std::fill( p_x.begin(), p_x.begin() + n, 0.0 );
    solve( &p_A[0], nC, n, b, &p_x[0] );

    // Now we get to iterate through all the variables in the system and move them
    // in and out of the C and NC sets as described in Baraff's paper.

    // The system is permuted in such a way that 0..nC-1 are clamped, nC..i-1 are unclamped,
    // and i..n-1 are "don't care" as defined above
    for( size_t i = nub; i < n; ++i )
    {
        // if w[i] >= 0, then nothing needs to be done; just add it to NC 
        // (which is done implicitly) and continue
        if( w[i] >= 0.0 )
            continue;

        // Compute fdirection
        std::vector<double> delta_f( n, 0.0 );
        if( p_a[i] > 0.0 )
            delta_f.back() = 1.0;
        else
            delta_f.back() = -1.0;

        // Transfer -A_{Cd} into v
        std::vector<double> v( i );
        for( size_t j = 0; j < i; ++j )
            v[j] = -A[ i*n + j ];

        // solve A_{11} x = -v_1
        solve( &p_A, nC, n, &v[0], &delta_f[0] );

        // compute delta_a = p_A * delta_f
        std::vector<double> delta_a( n );
        cblas_dgemv( 
            CblasColMajor, 
            CblasNoTrans,
            n,              // m
            n,              // n
            1.0,            // alpha
            &delta_a[0],    // a
            n,              // lda
            &delta_f[0],    // x
            1,              // incx
            0,              // beta (blas promises that if this is zero, then y need not be initialized)
            &delta_a[0],    // y
            1               // incy
            );

        // Now we need to figure out maxstep
        
    }
}
*/

