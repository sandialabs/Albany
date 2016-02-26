//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
PNP::ConcentrationResid<EvalT, Traits>::
ConcentrationResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF     (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  Concentration     ("Concentration", dl->qp_vector),
  Concentration_dot ("Concentration_dot", dl->qp_vector),
  ConcentrationGrad ("Concentration Gradient", dl->qp_vecgradient),
  PotentialGrad     ("Potential Gradient", dl->qp_gradient),
  ConcentrationResidual ("Concentration Residual",  dl->node_vector )
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Concentration);
  if (enableTransient)  this->addDependentField(Concentration_dot);
  this->addDependentField(ConcentrationGrad);
  this->addDependentField(PotentialGrad);

  this->addEvaluatedField(ConcentrationResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
  ConcentrationGrad.fieldTag().dataLayout().dimensions(dims);
  numSpecies = dims[2];

  // Placeholder for properties
  beta.resize(numSpecies);
  beta[0] =  1.0;
  beta[1] = -1.0;
  D.resize(numSpecies);
  D[0] =  1.0;
  D[1] =  2.0;

  this->setName("ConcentrationResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PNP::ConcentrationResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Concentration,fm);
  if (enableTransient) this->utils.setFieldData(Concentration_dot,fm);
  this->utils.setFieldData(ConcentrationGrad,fm);
  this->utils.setFieldData(PotentialGrad,fm);

  this->utils.setFieldData(ConcentrationResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PNP::ConcentrationResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools FST;

  // Scale gradient into a flux, reusing same memory
//  FST::scalarMultiplyDataData<ScalarT> (PhiFlux, Permittivity, PhiGrad);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
          for (std::size_t j=0; j < numSpecies; ++j) { 
            ConcentrationResidual(cell,node,j) = 0.0;
    } } }

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
        for (std::size_t qp=0; qp < numQPs; ++qp) {           
          for (std::size_t j=0; j < numSpecies; ++j) { 
            for (std::size_t dim=0; dim < numDims; ++dim) { 
              ConcentrationResidual(cell,node,j) += 
                D[j]*(ConcentrationGrad(cell,qp,j,dim)
                      + beta[j]*Concentration(cell,qp,j)*PotentialGrad(cell,qp,dim))
                *wGradBF(cell,node,qp,dim);
            }  
          }  
        }
      }
    }

}
//**********************************************************************

