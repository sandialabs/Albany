//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

template<typename EvalT, typename Traits>
ScatterResidual<EvalT, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  numNodes(0), worksetSize(0)
{  
  std::string fieldName = p.get<std::string>("Scatter Field Name");
  numLevels = p.get< int >("Number of Vertical Levels");

  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>
    (fieldName, dl->dummy));

  const Teuchos::ArrayRCP<std::string>& names =
    p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");

    numFields = names.size();
    const std::size_t num_val = numFields;
    val.resize(numFields);
    for (std::size_t eq = 0; eq < numFields; ++eq) {
      PHX::MDField<ScalarT,Cell,Node,Dim> mdf(names[eq],dl->node_scalar_level);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void ScatterResidual<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{

    for (std::size_t eq = 0; eq < numFields; ++eq)
      this->utils.setFieldData(val[eq],fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  val[0].dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numNodes = dims[1];

}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  Teuchos::RCP<Epetra_Vector> f = workset.f;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) 
        for (std::size_t level = 0; level < numLevels; level++) { 
          int n=eq+numFields*level;
          //(*f)[nodeID[node][n]] += (this->val[eq])(cell,node,level);
          double x = val[eq](cell,node,level).val();
          Epetra_Vector &g = (*f);
          int i = nodeID[node][n];
          g[i] +=  x;
        }
     }
  }
}

}
