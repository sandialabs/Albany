//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <typeinfo>

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  L2ProjectionResidual<EvalT, Traits>::
  L2ProjectionResidual(const Teuchos::ParameterList& p) :
    wBF         (p.get<std::string>                ("Weighted BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout") ),
    wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
    projectedField (p.get<std::string>               ("Projected Field Name"),
                    p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout") ),
    Pfield      (p.get<std::string>               ("Projection Field Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
    TResidual   (p.get<std::string>                ("Residual Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout") ),
		 haveMechSource(false),
		 haveSource(false)
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
      p.get< Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vector_dl->dimensions(dims);

    worksetSize = dims[0];
    numNodes = dims[1];
    numQPs  = dims[2];
    numDims = dims[3];

    this->setName("L2ProjectionResidual"+PHX::typeAsString<EvalT>());

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
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int node=0; node < numNodes; ++node)
    {
      /*TResidual(cell,node)=0.0;
      for (int qp=0; qp < numQPs; ++qp)
      {
        TResidual(cell,node) += ( projectedField(cell,qp)-
        Pfield(cell, qp))*wBF(cell,node,qp);
      }*/
      for (int k=0; k<numDims*numDims; ++k){
        TResidual(cell,node,k)=0.0;

        for (int qp=0; qp < numQPs; ++qp){
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


