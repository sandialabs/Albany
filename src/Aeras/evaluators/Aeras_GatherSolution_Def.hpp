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
GatherSolution<EvalT, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  numNodes(0), worksetSize(0)
{  
  Teuchos::ArrayRCP<std::string> solution_names;
  if (p.getEntryPtr("Solution Names")) {
    solution_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");
  }
  numLevels = p.get< int >("Number of Vertical Levels");

  val.resize(solution_names.size());
  for (std::size_t eq = 0; eq < solution_names.size(); ++eq) {
    PHX::MDField<ScalarT,Cell,Node,Dim> f(solution_names[eq],dl->node_scalar_level);
    val[eq] = f;
    this->addEvaluatedField(val[eq]);
  }
  // repeat for xdot if transient is enabled
  const Teuchos::ArrayRCP<std::string>& names_dot =
    p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

  val_dot.resize(names_dot.size());
  for (std::size_t eq = 0; eq < names_dot.size(); ++eq) {
    PHX::MDField<ScalarT,Cell,Node,Dim> f(names_dot[eq],dl->node_scalar_level);
    val_dot[eq] = f;
    this->addEvaluatedField(val_dot[eq]);
  }
  numFields = val.size();
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void GatherSolution<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{

    for (std::size_t eq = 0; eq < numFields; ++eq)
      this->utils.setFieldData(val[eq],fm);
    for (std::size_t eq = 0; eq < val_dot.size(); ++eq)
      this->utils.setFieldData(val_dot[eq],fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  val[0].dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numNodes = dims[1];

}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolution<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
 
    for (std::size_t node = 0; node < this->numNodes; ++node) {
    const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      for (std::size_t eq = 0; eq < numFields; eq++) 
        for (std::size_t level = 0; level < numLevels; level++) { 
          int n=eq+numFields*level;
          (this->val[eq])(cell,node,level) = (*x)[eqID[n]];
          (this->val_dot[eq])(cell,node,level) = (*xdot)[eqID[n]];
        }
     }
  }
}

}
