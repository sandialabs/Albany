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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace FELIX {


//**********************************************************************
template<typename EvalT, typename Traits>
StokesTauM<EvalT, Traits>::
StokesTauM(const Teuchos::ParameterList& p) :
  V           (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Gc            (p.get<std::string>                   ("Contravarient Metric Tensor Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  muFELIX    (p.get<std::string>                   ("FELIX Viscosity QP Variable Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  TauM            (p.get<std::string>                 ("Tau M Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
  
{
   Teuchos::ParameterList* tauM_list =
   p.get<Teuchos::ParameterList*>("Parameter List");

  meshSize = tauM_list->get("Mesh Size", 1.0);
  delta = tauM_list->get("Delta", 1.0);

  this->addDependentField(V);
  this->addDependentField(Gc);
  this->addDependentField(muFELIX);
 
  this->addEvaluatedField(TauM);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  normGc.resize(dims[0], numQPs);

  this->setName("StokesTauM"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesTauM<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(Gc,fm);
  this->utils.setFieldData(muFELIX,fm);
  
  this->utils.setFieldData(TauM,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesTauM<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  if (meshSize != 1.0) { //tau = h^numDims*delta - stabilization from Bochev et. al. "taxonomy" paper
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {       
          TauM(cell, qp) = delta; 
          for (std::size_t i=0; i<numDims; ++i)  
             TauM(cell,qp) *= meshSize; 
      }
    }
   }
   else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {       
        TauM(cell,qp) = 0.0;
        normGc(cell,qp) = 0.0;
        for (std::size_t i=0; i < numDims; ++i) {
          for (std::size_t j=0; j < numDims; ++j) {
            TauM(cell,qp) += V(cell,qp,i)*Gc(cell,qp,i,j)*V(cell,qp,j);
            normGc(cell,qp) += Gc(cell,qp,i,j)*Gc(cell,qp,i,j);          
          }
        }
        TauM(cell,qp) += 12.*muFELIX(cell,qp)*muFELIX(cell,qp)*std::sqrt(normGc(cell,qp));
        TauM(cell,qp) = 1./std::sqrt(TauM(cell,qp));
      }
    }
  }
}

//**********************************************************************
}

