//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "PHAL_Utilities.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
PNP::PotentialResid<EvalT, Traits>::
PotentialResid(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF         (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF     (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  Permittivity (p.get<std::string>  ("Permittivity Name"), dl->qp_scalar),
  Concentration     ("Concentration", dl->qp_vector),
  PotentialGrad     ("Potential Gradient", dl->qp_gradient),
  PotentialResidual ("Potential Residual",  dl->node_scalar )
{
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Permittivity);
  this->addDependentField(Concentration);
  this->addDependentField(PotentialGrad);

  this->addEvaluatedField(PotentialResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  wBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  Concentration.fieldTag().dataLayout().dimensions(dims);
  numSpecies = dims[2];

  // Placeholder for charges
  q.resize(numSpecies);
  q[0] =  5.0;
  q[1] = -5.0;

  this->setName("PotentialResid" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PNP::PotentialResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Permittivity,fm);
  this->utils.setFieldData(Concentration,fm);
  this->utils.setFieldData(PotentialGrad,fm);

  this->utils.setFieldData(PotentialResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PNP::PotentialResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  // Scale gradient into a flux
  // can't reuse memory, dependent fields must be const
  auto flux = PHAL::create_copy("tmp_flux", PotentialGrad.get_view());
  FST::scalarMultiplyDataData (flux, Permittivity.get_view(), PotentialGrad.get_view());
  FST::integrate(PotentialResidual.get_view(), flux, wGradBF.get_view(), false); // "false" overwrites

    
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
          for (std::size_t qp=0; qp < numQPs; ++qp) {           
            for (std::size_t j=0; j < numSpecies; ++j) { 
              PotentialResidual(cell,node) -= 
                q[j]*Concentration(cell,qp,j)*wBF(cell,node,qp);
//cout << "XXX " << cell << " " << node << " " << qp << " " << j << " " << q[j] << "  " << Concentration(cell,qp,j) << endl;
            }  
          }
        }
    }

}
//**********************************************************************

