//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_ScatterScalarNodalParameter.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
ScatterScalarNodalParameterBase<EvalT,Traits>::
ScatterScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  param_name = p.get<std::string>("Parameter Name");
  std::string field_name = p.isParameter("Field Name") ? p.get<std::string>("Field Name") : param_name;
  val = decltype(val)(field_name,dl->node_scalar);
  numNodes = 0;

  this->addDependentField(val);

  this->setName("Scatter Nodal Parameter" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.extent(1);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData /* workset */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "PHAL::ScatterScalarNodalParameter is supposed to be used only for Residual evaluation Type.");
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterScalarExtruded2DNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData /* workset */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "PHAL::ScatterScalarNodalParameter is supposed to be used only for Residual evaluation Type.");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
ScatterScalarNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
ScatterScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  ScatterScalarNodalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  // Create field tag
  nodal_field_tag =
    Teuchos::rcp(new PHX::Tag<ParamScalarT>(className, dl->dummy));

  this->addEvaluatedField(*nodal_field_tag);
}


// **********************************************************************
template<typename Traits>
void ScatterScalarNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->vector();
  Teuchos::ArrayRCP<ST> pvec_view = Albany::getNonconstLocalData(pvec);

  const Albany::IDArray& wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];
  auto param_overlap_vs = workset.distParamLib->get(this->param_name)->overlap_vector_space();
  auto param_vs = pvec->range();

  auto param_indexer = Albany::createGlobalLocalIndexer(param_vs);
  auto param_ov_indexer = Albany::createGlobalLocalIndexer(param_overlap_vs);
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lid_overlap = wsElDofs((int)cell,(int)node,0);
      const GO gid_overlap = param_ov_indexer->getGlobalElement(lid_overlap);
      const LO lid = param_indexer->getLocalElement(gid_overlap);
      if(lid >= 0) {
       pvec_view[lid] = (this->val)(cell,node);
      }
    }
  }
}

template<typename Traits>
ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
ScatterScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  ScatterScalarNodalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  fieldLevel = p.get<int>("Field Level");

  // Create field tag
  nodal_field_tag =
    Teuchos::rcp(new PHX::Tag<ParamScalarT>(className, dl->dummy));

  this->addEvaluatedField(*nodal_field_tag);
}

template<typename Traits>
void ScatterScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // TODO: find a way to abstract away from the map concept. Perhaps using Panzer::ConnManager?
  Teuchos::RCP<Thyra_Vector> pvec = workset.distParamLib->get(this->param_name)->vector();
  Teuchos::ArrayRCP<ST> pvec_view = Albany::getNonconstLocalData(pvec);

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  auto param_vs = pvec->range();
  auto overlapNodeVS = workset.disc->getOverlapNodeVectorSpace();

  auto param_indexer = Albany::createGlobalLocalIndexer(param_vs);
  auto ov_node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lnodeId = ov_node_indexer->getLocalElement(elNodeID[node]);
      LO base_id, ilayer;
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      if(ilayer==fieldLevel) {
        const LO lid = param_indexer->getLocalElement(elNodeID[node]);
        if(lid>=0) {
          pvec_view[ lid ] = (this->val)(cell,node);
        }
      }
    }
  }
}

} // namespace PHAL
