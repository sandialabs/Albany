//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename BetaScalarT>
StokesFOBasalResid<EvalT, Traits, BetaScalarT>::StokesFOBasalResid (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  basalResid (p.get<std::string> ("Basal Residual Variable Name"),dl->node_vector)
{
  basalSideName = p.get<std::string>("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

  u         = PHX::MDField<ScalarT,Cell,Side,QuadPoint,VecDim>(p.get<std::string> ("Velocity Side QP Variable Name"), dl_basal->qp_vector);
  beta      = PHX::MDField<BetaScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Basal Friction Coefficient Side QP Variable Name"),dl_basal->qp_scalar);
  BF        = PHX::MDField<RealType,Cell,Side,Node,QuadPoint>(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
  w_measure = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint> (p.get<std::string> ("Weighted Measure Name"), dl_basal->qp_scalar);

  this->addDependentField(u);
  this->addDependentField(beta);
  this->addDependentField(BF);
  this->addDependentField(w_measure);

  this->addEvaluatedField(basalResid);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->node_qp_gradient->dimensions(dims);
  int numSides = dims[1];
  numSideNodes = dims[2];
  numSideQPs   = dims[3];
  sideDim      = dims[4];
  numCellNodes = basalResid.fieldTag().dataLayout().dimension(1);

  dl->node_vector->dimensions(dims);
  vecDimFO     = std::min((int)dims[2],2);
  vecDim       = dims[2];

  regularized = p.get<Teuchos::ParameterList*>("Parameter List")->get("Regularize With Continuation",false);
  if (regularized)
  {
    homotopyParam = PHX::MDField<ScalarT,Dim>("Glen's Law Homotopy Parameter", dl->shared_param);
    this->addDependentField(homotopyParam);
  }

  // Index of the nodes on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
    {
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
    }
  }

  printedFF = -1.0;
  this->setName("StokesFOBasalResid"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename BetaScalarT>
void StokesFOBasalResid<EvalT, Traits, BetaScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(w_measure,fm);
  if (regularized)
    this->utils.setFieldData(homotopyParam,fm);

  this->utils.setFieldData(basalResid,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename BetaScalarT>
void StokesFOBasalResid<EvalT, Traits, BetaScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT ff = (regularized) ? pow(10.0, -10.0*homotopyParam(0)) : ScalarT(0);
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

    if (std::fabs(printedFF-ff)>0.0001*ff)
    {
        *output << "[Basal Residual] ff = " << ff << "\n";
        printedFF = ff;
    }
#endif

  // Zero out, to avoid leaving stuff from previous workset!
  for (int cell=0; cell<workset.numCells; ++cell)
    for (int node=0; node<numCellNodes; ++node)
      for (int dim=0; dim<vecDim; ++dim)
        basalResid(cell,node,dim) = 0;

  if (workset.sideSets->find(basalSideName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(basalSideName);

  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int node=0; node<numSideNodes; ++node)
    {
      for (int dim=0; dim<vecDimFO; ++dim)
      {
        basalResid(cell,sideNodes[side][node],dim) = 0.;
        for (int qp=0; qp<numSideQPs; ++qp)
        {
          basalResid(cell,sideNodes[side][node],dim) += (ff + beta(cell,side,qp)*u(cell,side,qp,dim))*BF(cell,side,node,qp)*w_measure(cell,side,qp);
        }
      }
    }
  }
}

} // Namespace FELIX
