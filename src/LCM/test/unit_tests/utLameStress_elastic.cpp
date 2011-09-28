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
#include "LCM/evaluators/LameStress.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx.hpp"
#include "Teuchos_ParameterList.hpp"

using namespace std;

namespace {

TEUCHOS_UNIT_TEST( LameStress_elastic, Instantiation )
{
  Teuchos::ParameterList parameterList("Stress");
  parameterList.set<string>("DefGrad Name", "Deformation Gradient");
  parameterList.set<string>("Stress Name", "Stress");

  const int worksetSize = 1;
  const int numQPts = 1;
  const int numDim = 3;
  Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint> > qp_scalar =
    Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint>(worksetSize, numQPts));
  Teuchos::RCP<PHX::MDALayout<Cell, QuadPoint, Dim, Dim> > qp_tensor =
    Teuchos::rcp(new PHX::MDALayout<Cell, QuadPoint, Dim, Dim>(worksetSize, numQPts, numDim, numDim));
  parameterList.set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", qp_scalar);
  parameterList.set< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout", qp_tensor);

  parameterList.set<string>("Lame Material Model", "Elastic");
  Teuchos::ParameterList& materialModelParametersList = parameterList.sublist("Lame Material Parameters");
  materialModelParametersList.set<double>("Youngs Modulus", 1.0);
  materialModelParametersList.set<double>("Poissons Ratio", 0.25);

  LCM::LameStress<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits> lameStress(parameterList);

  int i1 = 5;
  TEST_EQUALITY_CONST( i1, 5 );
}

} // namespace
