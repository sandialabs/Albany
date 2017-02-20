//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
FELIX::L2ProjectedBoundaryLaplacianResidual<EvalT, Traits>::
L2ProjectedBoundaryLaplacianResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  laplacian_coeff = p.get<double>("Laplacian Coefficient", 1.0);
  mass_coeff = p.get<double>("Mass Coefficient", 1.0);


  // Setting up the fields required by the regularizations
  sideName = p.get<std::string> ("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  const std::string& solution_name       = p.get<std::string>("Solution Variable Name");
  const std::string& field_name          = p.get<std::string>("Field Name");
  const std::string& gradField_name      = p.get<std::string>("Field Gradient Name");
  const std::string& gradBFname          = p.get<std::string>("Gradient BF Side Name");
  const std::string& w_side_measure_name = p.get<std::string>("Weighted Measure Side Name");
  const std::string& residual_name = p.get<std::string>("L2 Projected Boundary Laplacian Residual Name");

  solution           = PHX::MDField<ScalarT,Cell,Node>(solution_name, dl->node_scalar);
  field              = PHX::MDField<ParamScalarT,Cell,Side,Node>(field_name, dl_side->node_scalar);
  gradField          = PHX::MDField<ScalarT,Cell,Side,QuadPoint,Dim>(gradField_name, dl_side->qp_gradient);
  gradBF             = PHX::MDField<MeshScalarT,Cell,Side,Node, QuadPoint,Dim>(gradBFname,dl_side->node_qp_gradient),
  w_side_measure     = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint>(w_side_measure_name, dl_side->qp_scalar);
  bdLaplacian_L2Projection_res = PHX::MDField<ScalarT,Cell,Node>(residual_name, dl->node_scalar);

  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numCells  = dl->node_scalar->dimension(0);
  numNodes  = dl->node_scalar->dimension(1);
 
  numSideNodes  = dl_side->node_scalar->dimension(2);
  numBasalQPs = dl_side->qp_scalar->dimension(2);
  int numSides = dl_side->node_scalar->dimension(1);
  sideDim  = cellType->getDimension()-1;

  this->addDependentField(solution);
  this->addDependentField(field);
  this->addDependentField(gradField);
  this->addDependentField(gradBF);
  this->addDependentField(w_side_measure);


  this->addEvaluatedField(bdLaplacian_L2Projection_res);

  this->setName("Boundary Laplacian L2 Projection Residual" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;


  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    //Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::L2ProjectedBoundaryLaplacianResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(solution, fm);
  this->utils.setFieldData(field, fm);
  this->utils.setFieldData(gradField, fm);
  this->utils.setFieldData(gradBF, fm);
  this->utils.setFieldData(w_side_measure, fm);

  this->utils.setFieldData(bdLaplacian_L2Projection_res, fm);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void FELIX::L2ProjectedBoundaryLaplacianResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  
  for (int cell=0; cell<numCells; ++cell)
    for (int inode=0; inode<numNodes; ++inode)
      bdLaplacian_L2Projection_res(cell,inode) = solution(cell,inode);
    

  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      MeshScalarT trapezoid_weights = 0;
      for (int qp=0; qp<numBasalQPs; ++qp)
        trapezoid_weights += w_side_measure(cell,side, qp);
      trapezoid_weights /= numSideNodes;

      for (int inode=0; inode<numSideNodes; ++inode) {
        ScalarT t = 0;
        for (int qp=0; qp<numBasalQPs; ++qp)
          for (int idim=0; idim<sideDim; ++idim)
            t -= laplacian_coeff*gradField(cell,side,qp,idim)*gradBF(cell,side,inode, qp,idim)*w_side_measure(cell,side, qp);

        //using trapezoidal rule to get diagonal mass matrix
        t += (solution(cell,sideNodes[side][inode])-mass_coeff*field(cell,side,inode))* trapezoid_weights;

        bdLaplacian_L2Projection_res(cell,sideNodes[side][inode]) = t;
    }
  }

  }
}
