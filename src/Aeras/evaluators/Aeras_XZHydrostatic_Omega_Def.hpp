//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

#include "Aeras_Eta.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_Omega<EvalT, Traits>::
XZHydrostatic_Omega(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Velocity  (p.get<std::string> ("Velocity"),             dl->node_vector_level),
  density   (p.get<std::string> ("Density"),              dl->node_scalar_level),
  Cpstar    (p.get<std::string> ("QP Cpstar"),            dl->node_scalar_level),
  gradp     (p.get<std::string> ("Gradient QP Pressure"), dl->qp_gradient_level),
  divpivelx (p.get<std::string> ("Divergence QP PiVelx"), dl->qp_scalar_level),
  omega     (p.get<std::string> ("Omega")              ,  dl->node_scalar_level),
  numQPs     (dl->node_qp_scalar    ->dimension(2)),
  numDims    (dl->node_qp_gradient  ->dimension(3)),
  numLevels  (dl->node_scalar_level ->dimension(2)),
  Cp         (p.isParameter("XZHydrostatic Problem") ? 
                p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Cp", 1005.7):
                p.get<Teuchos::ParameterList*>("Hydrostatic Problem")->get<double>("Cp", 1005.7)),
  E (Eta<EvalT>::self())
{

  this->addDependentField(Velocity);
  this->addDependentField(gradp);
  this->addDependentField(density);
  this->addDependentField(Cpstar);
  this->addDependentField(divpivelx);

  this->addEvaluatedField(omega);

  this->setName("Aeras::XZHydrostatic_Omega" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Omega<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Velocity  ,   fm);
  this->utils.setFieldData(gradp     ,   fm);
  this->utils.setFieldData(density   ,   fm);
  this->utils.setFieldData(Cpstar    ,   fm);
  this->utils.setFieldData(divpivelx ,   fm);
  this->utils.setFieldData(omega     ,   fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_Omega<EvalT, Traits>::
operator() (const XZHydrostatic_Omega_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp) {
    for (int level=0; level < numLevels; ++level) {
      ScalarT                               sum  = -0.5*divpivelx(cell,qp,level) * E.delta(level);
      for (int j=0; j<level; ++j)           sum -=     divpivelx(cell,qp,j)     * E.delta(j);
      for (int dim=0; dim < numDims; ++dim) sum += Velocity(cell,qp,level,dim)*gradp(cell,qp,level,dim);
      omega(cell,qp,level) = sum/(Cpstar(cell,qp,level)*density(cell,qp,level));
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Omega<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        ScalarT                               sum  = -0.5*divpivelx(cell,qp,level) * E.delta(level);
        for (int j=0; j<level; ++j)           sum -=     divpivelx(cell,qp,j)     * E.delta(j);
        for (int dim=0; dim < numDims; ++dim) sum += Velocity(cell,qp,level,dim)*gradp(cell,qp,level,dim);
        omega(cell,qp,level) = sum/(Cpstar(cell,qp,level)*density(cell,qp,level));
      }
    }
  }

#else
  Kokkos::parallel_for(XZHydrostatic_Omega_Policy(0,workset.numCells),*this);
  cudaCheckError();

#endif
}
}
