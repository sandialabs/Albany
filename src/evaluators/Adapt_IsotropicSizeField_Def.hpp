//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"

template<typename T>
T Sqr(T num)
{
    return num * num;
}

template<typename EvalT, typename Traits>
Adapt::IsotropicSizeField<EvalT, Traits>::
IsotropicSizeField(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string>("Coordinate Vector Name"), dl->qp_vector),
  coordVec_vertices(p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector),
  qp_weights("Weights", dl->qp_scalar)
{

  //! get and validate IsotropicSizeField parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidSizeFieldParameters();
  plist->validateParameters(*reflist,0);

  // Isotropic --> element size scalar corresponding to the nominal element radius
  // Anisotropic --> element size vector (x, y, z) with the width, length, and height of the element
  // Weighted versions (upcoming) --> scale the above sizes with a scalar or vector field

  //! Scaling vectors / scalars to use for weighting
  if(plist->isParameter("Size Field Scaling Vector Field")) {
    scalingName = plist->get<std::string>("Size Field Scaling Vector Field");
    scalingType = VECTOR;
  }
  else if(plist->isParameter("Size Field Scaling Field")) {
    scalingName = plist->get<std::string>("Size Field Scaling Field");
    scalingType = SCALAR;
  }
  else {
    scalingType = NOTSCALED;
  }

  className = "Isotropic Size Field";
  outputToExodus = plist->get<bool>("Output to File", true);
  outputCellAverage = plist->get<bool>("Generate Cell Average", true);
  outputQPData = plist->get<bool>("Generate QP Values", false);
  outputNodeData = plist->get<bool>("Generate Nodal Values", false);

  //! number of quad points per cell and dimension
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  Teuchos::RCP<PHX::DataLayout> cell_dl = dl->cell_scalar;
  Teuchos::RCP<PHX::DataLayout> vert_vector_dl = dl->vertices_vector;
  numQPs = vector_dl->dimension(1);
  numDims = vector_dl->dimension(2);
  numVertices = vert_vector_dl->dimension(2);
 
  //! add dependent fields
/* Not now
  Teuchos::RCP<PHX::DataLayout>& field_dl = isVectorField ? vector_dl : scalar_dl;
  PHX::MDField<ScalarT> f(scalingName, field_dl);  field = f;
  this->addDependentField(field);
*/
  this->addDependentField(qp_weights);
  this->addDependentField(coordVec);
  this->addDependentField(coordVec_vertices);

  //! Register with state manager
  Albany::StateManager* pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");
  if( outputCellAverage ) {
    pStateMgr->registerStateVariable(className + "_Cell", dl->cell_scalar, dl->dummy, "all", "scalar", 0.0, false, outputToExodus);
  }
  if( outputQPData ) {
    pStateMgr->registerStateVariable(className + "_QP", dl->qp_scalar, dl->dummy, "all", "scalar", 0.0, false, outputToExodus);
  }
  if( outputNodeData ) {
    // The weighted projected value
    pStateMgr->registerStateVariable(className + "_Node", dl->node_scalar, dl->dummy, "all", "scalar", 0.0, false, outputToExodus);
    // The value of the weights used in the projection
    pStateMgr->registerStateVariable(className + "_NodeWgt", dl->node_scalar, dl->dummy, "all", "scalar", 0.0, false, outputToExodus);
  }

  // Create field tag
  size_field_tag = 
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));

  this->addEvaluatedField(*size_field_tag);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Adapt::IsotropicSizeField<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
//  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(coordVec_vertices,fm);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void Adapt::IsotropicSizeField<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  using Albany::ADValue;
  ScalarT value;

  if( outputCellAverage ) { // nominal radius

    // Get shards Array (from STK) for this workset
    Albany::MDArray data = (*workset.stateArrayPtr)[className + "_Cell"];
    std::vector<int> dims;
    data.dimensions(dims);
    int size = dims.size();


    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          getCellRadius(cell, value);
          data(cell, (std::size_t)0) = ADValue(value);
    }
  }

}

template<typename EvalT, typename Traits>
void Adapt::IsotropicSizeField<EvalT, Traits>::
getCellRadius(const std::size_t cell, typename EvalT::ScalarT& cellRadius) const
{
  std::vector<ScalarT> maxCoord(3,-1e10);
  std::vector<ScalarT> minCoord(3,+1e10);

  for (std::size_t v=0; v < numVertices; ++v) {
    for (std::size_t k=0; k < numDims; ++k) {
      if(maxCoord[k] < coordVec_vertices(cell,v,k)) maxCoord[k] = coordVec_vertices(cell,v,k);
      if(minCoord[k] > coordVec_vertices(cell,v,k)) minCoord[k] = coordVec_vertices(cell,v,k);
    }
  }

  cellRadius = 0.0;
  for (std::size_t k=0; k < numDims; ++k) 
    cellRadius += (maxCoord[k] - minCoord[k]) *  (maxCoord[k] - minCoord[k]);

  cellRadius = std::sqrt(cellRadius) / 2.0;

}


// **********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Adapt::IsotropicSizeField<EvalT,Traits>::getValidSizeFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid IsotropicSizeField Params"));;

  validPL->set<std::string>("Name", "", "Name of size field function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Name", "", "Field to save");
  validPL->set<std::string>("Vector Field Name", "", "Vector field to save");
  // 
  validPL->set<std::string>("Size Field Scaling Field", "<Field Name>", "Field to use to scale the element sizes (default - 1.0)");
  validPL->set<std::string>("Size Field Scaling Vector Field", "<Field Name>", "Field to use to scale the element sizes (default - 1.0)");
  validPL->set<bool>("Output to File", true, "Whether size field info should be output to a file");
  validPL->set<bool>("Generate Cell Average", true, "Whether cell average field should be generated");
  validPL->set<bool>("Generate QP Values", true, "Whether values at the quadpoints should be generated");
  validPL->set<bool>("Generate Nodal Values", true, "Whether values at the nodes should be generated");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

