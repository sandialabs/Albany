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

#include <typeinfo>

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  L2ProjectionResidual<EvalT, Traits>::
  L2ProjectionResidual(const Teuchos::ParameterList& p) :
    wBF         (p.get<std::string>                ("Weighted BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
    wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
    projectedField (p.get<std::string>               ("Projected Field Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
    Pfield      (p.get<std::string>               ("Projection Field Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
    TResidual   (p.get<std::string>                ("Residual Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
  {
    if (p.isType<bool>("Disable Transient"))
      enableTransient = !p.get<bool>("Disable Transient");
    else enableTransient = true;

    this->addDependentField(wBF);
    this->addDependentField(wGradBF);
    this->addDependentField(projectedField);
    this->addDependentField(Pfield);
 //   if (haveSource) this->addDependentField(Source);
 //   if (haveMechSource) this->addDependentField(MechSource);

    this->addEvaluatedField(TResidual);

    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vector_dl->dimensions(dims);

    worksetSize = dims[0];
    numNodes = dims[1];
    numQPs  = dims[2];
    numDims = dims[3];

    this->setName("L2ProjectionResidual"+PHX::TypeString<EvalT>::value);

  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void L2ProjectionResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(wGradBF,fm);
    this->utils.setFieldData(projectedField,fm);
    this->utils.setFieldData(Pfield,fm);
    this->utils.setFieldData(TResidual,fm);
  }

//**********************************************************************
template<typename EvalT, typename Traits>
void L2ProjectionResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
    for (std::size_t node=0; node < numNodes; ++node)
    {
      /*TResidual(cell,node)=0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
        TResidual(cell,node) += ( projectedField(cell,qp)-
        Pfield(cell, qp))*wBF(cell,node,qp);
      }*/
      for (std::size_t k=0; k<numDims*numDims; ++k){
        TResidual(cell,node,k)=0.0;

        for (std::size_t qp=0; qp < numQPs; ++qp){
          // need to transform tensor valued Pfield to a vector for projectedField and TResidual
          TResidual(cell,node,k) += (projectedField(cell,qp,k) -
          Pfield(cell,qp,k/numDims,k%numDims))*wBF(cell,node,qp);

          //cout << "Projected Field: " << Sacado::ScalarValue<ScalarT>::eval(projectedField(cell,node,k)) << std::endl;
          //cout << "PField: " << Sacado::ScalarValue<ScalarT>::eval(Pfield(cell,node,k/numDims,k%numDims)) << std::endl;
        }
      }
    }
  }
}
//**********************************************************************
}


