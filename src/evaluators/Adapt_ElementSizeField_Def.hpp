//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Adapt_NodalDataBlock.hpp"

template<typename T>
T Sqr(T num)
{
    return num * num;
}

template<typename EvalT, typename Traits>
Adapt::ElementSizeFieldBase<EvalT, Traits>::
ElementSizeFieldBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec(p.get<std::string>("Coordinate Vector Name"), dl->qp_vector),
  coordVec_vertices(p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector),
  qp_weights("Weights", dl->qp_scalar)
{

  //! get and validate ElementSizeField parameter list
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

  className = "Element Size Field";
  outputToExodus = plist->get<bool>("Output to File", true);
  outputCellAverage = plist->get<bool>("Generate Cell Average", true);
  outputQPData = plist->get<bool>("Generate QP Values", false);
  outputNodeData = plist->get<bool>("Generate Nodal Values", false);
  isAnisotropic = plist->get<bool>("Anisotropic Size Field", false);

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
  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");

  if( outputCellAverage ) {
    if(isAnisotropic) //An-isotropic
      this->pStateMgr->registerStateVariable(className + "_Cell", dl->cell_vector, dl->dummy, "all", "scalar", 
         0.0, false, outputToExodus);
    else
      this->pStateMgr->registerStateVariable(className + "_Cell", dl->cell_scalar, dl->dummy, "all", "scalar", 
         0.0, false, outputToExodus);
  }

  if( outputQPData ) {
    if(isAnisotropic) //An-isotropic
      this->pStateMgr->registerStateVariable(className + "_QP", dl->qp_vector, dl->dummy, "all", 
        "scalar", 0.0, false, outputToExodus);
    else
      this->pStateMgr->registerStateVariable(className + "_QP", dl->qp_scalar, dl->dummy, "all", 
        "scalar", 0.0, false, outputToExodus);
  }

  if( outputNodeData ) {
    // The weighted projected value

    // Note that all dl->node_node_* layouts are handled by the Adapt_NodalDataBlock class, inside
    // of the state manager, as they require interprocessor synchronization

    if(isAnisotropic){ //An-isotropic
      this->pStateMgr->registerStateVariable(className + "_Node", dl->node_node_vector, dl->dummy, "all", 
         "scalar", 0.0, false, outputToExodus);


    }
    else {
      this->pStateMgr->registerStateVariable(className + "_Node", dl->node_node_scalar, dl->dummy, "all", 
         "scalar", 0.0, false, outputToExodus);

    }

    // The value of the weights used in the projection
    // Initialize to zero - should give us nan's during the division step if something is wrong
    this->pStateMgr->registerStateVariable(className + "_NodeWgt", dl->node_node_scalar, dl->dummy, "all", 
         "scalar", 0.0, false, outputToExodus);

  }

  // Create field tag
  size_field_tag = 
    Teuchos::rcp(new PHX::Tag<ScalarT>(className, dl->dummy));

  this->addEvaluatedField(*size_field_tag);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Adapt::ElementSizeFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
//  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(coordVec_vertices,fm);

}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
// **********************************************************************
template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::Residual, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  // Note that we only need to initialize the vectors when dealing with node data, as we assume
  // the vectors are initialized to zero for Epetra_Export "ADD" operation
  // Zero data for accumulation here
  if( this->outputNodeData ) { 
    workset.node_data->initializeVectors(0.0);
  }
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double value;

  if( this->outputCellAverage ) { // nominal radius

    // Get shards Array (from STK) for this workset
    Albany::MDArray data = (*workset.stateArrayPtr)[this->className + "_Cell"];
    std::vector<int> dims;
    data.dimensions(dims);
    int size = dims.size();


    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          this->getCellRadius(cell, value);
//          data(cell, (std::size_t)0) = ADValue(value);
          data(cell, (std::size_t)0) = value;
    }
  }

  if( this->outputNodeData ) { // nominal radius, store as nodal data that will be scattered and summed

    // Get the node data block container
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->pStateMgr->getNodalDataBlock();
    Teuchos::RCP<Epetra_Vector> data = node_data->getLocalNodeVec();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Epetra_BlockMap> local_node_map = node_data->getLocalMap();

    int l_nV = this->numVertices;
    int l_nD = this->numDims;



    for (int cell = 0; cell < workset.numCells; ++cell) { // loop over all elements in workset

      std::vector<double> maxCoord(3,-1e10);
      std::vector<double> minCoord(3,+1e10);

      // Get element width in x, y, z
      for (int v=0; v < l_nV; ++v) { // loop over all the "corners" of each element
        for (int k=0; k < l_nD; ++k) { // loop over each dimension of the problem
          if(maxCoord[k] < this->coordVec_vertices(cell,v,k)) maxCoord[k] = this->coordVec_vertices(cell,v,k);
          if(minCoord[k] > this->coordVec_vertices(cell,v,k)) minCoord[k] = this->coordVec_vertices(cell,v,k);
        }
      }

      if(this->isAnisotropic) //An-isotropic
        // Note: code assumes blocksize of blockmap is numDims + 1 - the last entry accumulates the weight
        for (int node = 0; node < l_nV; ++node) { // loop over all the "corners" of each element

          int global_node = wsElNodeID[cell][node]; // get the global id of this node

          int local_node = local_node_map->LID(global_node); // skip the node if it is not owned by me
          if(local_node < 0) continue;

          // accumulate 1/2 of the element width in each dimension - into each element corner
          for (int k=0; k < l_nD; ++k) 
//            data[global_node][k] += ADValue(maxCoord[k] - minCoord[k]) / 2.0;
            (*data)[local_node * l_nD + k] += (maxCoord[k] - minCoord[k]) / 2.0;

          // save the weight (denominator)
          (*data)[local_node * l_nD + l_nD - 1] += 1.0;
      } // end anisotropic size field

      else // isotropic size field
        // Note: code assumes blocksize of blockmap is 1 + 1 = 2 - the last entry accumulates the weight
        for (int node = 0; node < l_nV; ++node) { // loop over all the "corners" of each element

          int global_node = wsElNodeID[cell][node]; // get the global id of this node

          int local_node = local_node_map->LID(global_node); // skip the node if it is not owned by me
          if(local_node < 0) continue;

          // accumulate 1/2 of the element width in each dimension - into each element corner
          for (int k=0; k < l_nD; ++k) 
//            data[global_node][k] += ADValue(maxCoord[k] - minCoord[k]) / 2.0;
            (*data)[local_node * 2] += (maxCoord[k] - minCoord[k]) / 2.0;

          // save the weight (denominator)
          (*data)[local_node * 2 + 1] += 1.0;

          // the above calculates the average of the element width, depth, and height when
          // divided by the accumulated weights

      } // end isotropic size field
    } // end cell loop
  } // end node data if

}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  if( this->outputNodeData ) {

    // Note: we are in postEvaluate so all PEs call this

    // Get the node data block container
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->pStateMgr->getNodalDataBlock();
    Teuchos::RCP<Epetra_Vector> data = node_data->getLocalNodeVec();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Epetra_BlockMap> local_node_map = node_data->getLocalMap();

    // Build the exporter
    node_data->initializeExport();

    // do the export
    node_data->exportAddNodalDataBlock();

    int numNodes = this->numVertices;
    int blocksize = node_data->getBlocksize();

    // if isotropic, blocksize == 2 , if anisotropic blocksize == nDOF at node + 1
    // ndim if vector, ndim * ndim if tensor

    // all PEs divide the accumulated value(s) by the weights

    for (int v=0; v < numNodes; ++v) 
      for(int k=0; k < blocksize - 1; ++k)
            (*data)[v * blocksize + k] /= (*data)[v * blocksize + blocksize - 1];

    // Export the data from the local to overlapped decomposition
    // Divide the overlap field through by the weights
    // Store the overlapped vector data back in stk in the field "field_name"

    node_data->saveNodalDataState();

  }

}

// **********************************************************************
// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::Jacobian, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::Jacobian, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Jacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::Tangent, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::Tangent, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Tangent, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::Tangent, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG_MP

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::SGResidual, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::SGResidual, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGResidual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGResidual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::SGJacobian, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGJacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::SGTangent, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::SGTangent, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGTangent, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::SGTangent, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Mulit-point Residual
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::MPResidual, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::MPResidual, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPResidual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPResidual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::MPJacobian, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPJacobian, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}
// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
Adapt::
ElementSizeField<PHAL::AlbanyTraits::MPTangent, Traits>::
ElementSizeField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ElementSizeFieldBase<PHAL::AlbanyTraits::MPTangent, Traits>(p, dl)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPTangent, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

template<typename Traits>
void Adapt::ElementSizeField<PHAL::AlbanyTraits::MPTangent, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
}

#endif

// **********************************************************************

template<typename EvalT, typename Traits>
void Adapt::ElementSizeFieldBase<EvalT, Traits>::
getCellRadius(const std::size_t cell, typename EvalT::MeshScalarT& cellRadius) const
{
  std::vector<MeshScalarT> maxCoord(3,-1e10);
  std::vector<MeshScalarT> minCoord(3,+1e10);

  for (int v=0; v < numVertices; ++v) {
    for (int k=0; k < numDims; ++k) {
      if(maxCoord[k] < coordVec_vertices(cell,v,k)) maxCoord[k] = coordVec_vertices(cell,v,k);
      if(minCoord[k] > coordVec_vertices(cell,v,k)) minCoord[k] = coordVec_vertices(cell,v,k);
    }
  }

  cellRadius = 0.0;
  for (int k=0; k < numDims; ++k) 
    cellRadius += (maxCoord[k] - minCoord[k]) *  (maxCoord[k] - minCoord[k]);

  cellRadius = std::sqrt(cellRadius) / 2.0;

}


// **********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Adapt::ElementSizeFieldBase<EvalT,Traits>::getValidSizeFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ElementSizeField Params"));;

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
  validPL->set<bool>("Anisotropic Size Field", true, "Is this size field calculation anisotropic?");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");

  return validPL;
}

// **********************************************************************

