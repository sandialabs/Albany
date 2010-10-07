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


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

template<typename EvalT, typename Traits>
QCAD::PoissonSource<EvalT, Traits>::
PoissonSource(Teuchos::ParameterList& p) :
  poissonSource(p.get<std::string>("Source Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  potential(p.get<std::string>("Variable Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout")),
  coordVec(p.get<std::string>("Coordinate Vector Name"),
	    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout"))
{
  Teuchos::ParameterList* psList = p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist=this->getValidPoissonSourceParameters();
  psList->validateParameters(*reflist,0);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  factor = psList->get("Factor", 1.0);

  // Add factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
   new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
        "Poisson Source Factor", this, paramLib);

  this->addDependentField(potential);
  this->addDependentField(coordVec);

  this->addEvaluatedField(poissonSource);
  this->setName("Poisson Source");
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(poissonSource,fm);
  this->utils.setFieldData(potential,fm);
  this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::PoissonSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
	poissonSource(cell,qp) = factor
          * chargeDistribution(numDims, &coordVec(cell,qp,0), potential(cell,qp));
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT& 
QCAD::PoissonSource<EvalT,Traits>::getValue(const std::string &n)
{
  return factor;
}

// **********************************************************************

template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::PoissonSource<EvalT,Traits>::getValidPoissonSourceParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("Valid Poisson Problem Params"));;

  validPL->set<double>("Factor", 1.0, "Constant multiplier in source term");

  return validPL;
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename QCAD::PoissonSource<EvalT,Traits>::ScalarT
QCAD::PoissonSource<EvalT,Traits>::chargeDistribution(
    const int numDim, const MeshScalarT* coord, const ScalarT& phi) const
{
   ScalarT charge;
   switch (numDim) {
     case 2:
          if (coord[1]<0.8) charge = (coord[1]*coord[1]);
          else   charge = 3.0;
          charge *= (1.0 + exp(-phi));
        break;
     case 1:
     case 3:
     default:
      TEST_FOR_EXCEPT(true);
   }
   return charge;
}
// **********************************************************************
