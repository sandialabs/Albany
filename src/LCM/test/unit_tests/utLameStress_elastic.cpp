/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Teuchos_UnitTestHarness.hpp"

namespace {

TEUCHOS_UNIT_TEST( Int, Basic )
{
  int i1 = 5;
  TEST_EQUALITY_CONST( i1, 5 );
}

TEUCHOS_UNIT_TEST( Int, Assignment )
{
  int i1 = 4;
  int i2 = i1;
  TEST_EQUALITY( i2, i1 );
}

} // namespace
