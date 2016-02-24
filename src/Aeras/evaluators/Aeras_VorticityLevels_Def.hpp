//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
VorticityLevels<EvalT, Traits>::
VorticityLevels(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node   (p.get<std::string>   ("Velx"),           dl->node_vector_level),
  GradBF     (p.get<std::string>   ("Gradient BF Name"),        dl->node_qp_gradient),
  vort_val_qp (p.get<std::string>   ("Vorticity Variable Name"),dl->qp_scalar_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(vort_val_qp);

  this->setName("Aeras::VorticityLevels"+PHX::typeAsString<EvalT>());
  //std::cout<< "Aeras::VorticityLevels: " << numNodes << " " << numDims << " " << numQPs << " " << numLevels << std::endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void VorticityLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(vort_val_qp,fm);
}

//**********************************************************************
//Kokkos kernals
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void VorticityLevels<EvalT, Traits>::
operator() (const Vorticity_Tag& tag, const int & cell) const {

   
  for (int qp=0; qp < numQPs; ++qp) {
    for (int node= 0 ; node < numNodes; ++node) { 
      for (int level=0; level < numLevels; ++level) {
         vort_val_qp(cell,qp,level) = 0.0;
      }
    }
  }

  for (int qp=0; qp < numQPs; ++qp) {
    for (int node= 0 ; node < numNodes; ++node) { 
      for (int level=0; level < numLevels; ++level) {
         vort_val_qp(cell,qp,level) += (val_node(cell,node,level,1) * GradBF(cell,node,qp,0) 
                                     -  val_node(cell,node,level,0) * GradBF(cell,node,qp,1));
      }
    }
  }

}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void VorticityLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{


#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  PHAL::set(vort_val_qp, 0.0);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node= 0 ; node < numNodes; ++node) { 
        for (int level=0; level < numLevels; ++level) {
            vort_val_qp(cell,qp,level) += (val_node(cell,node,level,1) * GradBF(cell,node,qp,0) 
                                        -  val_node(cell,node,level,0) * GradBF(cell,node,qp,1));
        }
      }
    }
  }

#else

  Kokkos::parallel_for(Vorticity_Policy(0,workset.numCells),*this);

#endif

}

}

