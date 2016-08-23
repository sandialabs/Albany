//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
ScatterScalarNodalParameterBase<EvalT,Traits>::
ScatterScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  param_name = p.get<std::string>("Parameter Name");
  std::string field_name = p.isParameter("Field Name") ? p.get<std::string>("Field Name") : param_name;
  val = PHX::MDField<ParamScalarT,Cell,Node>(field_name,dl->node_scalar);
  numNodes = 0;

  this->addEvaluatedField(val);

  this->setName("Scatter Nodal Parameter" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.dimension(1);
}

// **********************************************************************

template<typename EvalT, typename Traits>
ScatterScalarNodalParameter<EvalT, Traits>::
ScatterScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  ScatterScalarNodalParameterBase<EvalT, Traits>(p,dl)
{
}

template<typename EvalT, typename Traits>
ScatterScalarNodalParameter<EvalT, Traits>::
ScatterScalarNodalParameter(const Teuchos::ParameterList& p) :
  ScatterScalarNodalParameterBase<EvalT, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "PHAL::ScatterScalarNodalParameter is supposed to be used only for Residual evaluation Type.");
}

// **********************************************************************
template<typename Traits>
void ScatterScalarNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> pvecT;
  try {
    pvecT = workset.distParamLib->get(this->param_name)->overlapped_vector();
  } catch (const std::logic_error& e) {
    const std::string evalt = PHX::typeAsString<PHAL::AlbanyTraits::Residual>();
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, std::logic_error,
      "PHAL::ScatterScalarNodalParameter<"
      << evalt.substr(1, evalt.size() - 2)
      << ", Traits>::evaluateFields: parameter " << this->param_name
      << " is not in workset.distParamLib. If this is a Tpetra-only build"
      << " we currently expect this result; sorry.");
  }
  Teuchos::ArrayRCP<ST> pvecT_constView = pvecT->get1dViewNonConst();

  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lid = wsElDofs((int)cell,(int)node,0);
      if(lid >= 0)
        pvecT_constView[lid] = (this->val)(cell,node);
    }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarExtruded2DNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "PHAL::ScatterScalarExtruded2DNodalParameter is supposed to be used only for Residual evaluation Type.");
}

// **********************************************************************
template<typename Traits>
void ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> pvecT;
  try {
    pvecT = workset.distParamLib->get(this->param_name)->overlapped_vector();
  } catch (const std::logic_error& e) {
    const std::string evalt = PHX::typeAsString<PHAL::AlbanyTraits::Residual>();
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, std::logic_error,
      "PHAL::ScatterScalarExtruded2DNodalParameter<"
      << evalt.substr(1, evalt.size() - 2)
      << ", Traits>::evaluateFields: parameter " << this->param_name
      << " is not in workset.distParamLib. If this is a Tpetra-only build"
      << " we currently expect this result; sorry.");
  }
  Teuchos::ArrayRCP<ST> pvecT_constView = pvecT->get1dViewNonConst();

  //const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  int numLayers = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
    //  LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(wsElDofs((int)cell,(int)node,0));
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
      LO base_id, ilayer;
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      if(ilayer==fieldLevel) {
        GO ginode = workset.disc->getOverlapNodeMapT()->getGlobalElement(lnodeId);
        pvecT_constView[pvecT->getMap()->getLocalElement(ginode)  ] = (this->val)(cell,node);
      }
    }
  }
}

// **********************************************************************

}
