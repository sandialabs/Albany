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

  className = plist->get<std::string>("Size Field Name", "Element_Size_Field");
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
//    if(isAnisotropic) //An-isotropic
//    Always anisotropic?
      this->pStateMgr->registerStateVariable(className + "_QP", dl->qp_vector, dl->dummy, "all", 
        "scalar", 0.0, false, outputToExodus);
//    else
//      this->pStateMgr->registerStateVariable(className + "_QP", dl->qp_scalar, dl->dummy, "all", 
//        "scalar", 0.0, false, outputToExodus);
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
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->pStateMgr->getStateInfoStruct()->getNodalDataBlock();
    node_data->initializeVectors(0.0);
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

  if( this->outputQPData ) { // x_\xi \cdot x_\xi, x_\eta \cdot x_\eta, x_\zeta \cdot x_\zeta

    // Get shards Array (from STK) for this workset
    Albany::MDArray data = (*workset.stateArrayPtr)[this->className + "_QP"];
    std::vector<int> dims;
    data.dimensions(dims);
    int size = dims.size();


    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
          this->getCellRadius(cell, value);
//          data(cell, (std::size_t)0) = ADValue(value);
          data(cell, (std::size_t)0) = value;
    }
/*
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {      
        for (std::size_t i=0; i < numDims; ++i) { // loop over \xi, \eta, \zeta
          data(cell, qp, i) = 0.0;
          for (std::size_t j=0; j < numDims; ++j) {
            data(cell, qp, i) += coordVec(cell, qp, j) * wGradBF(cell, node, qp, j);
            for (std::size_t alpha=0; alpha < numDims; ++alpha) {  
              Gc(cell,qp,i,j) += jacobian_inv(cell,qp,alpha,i)*jacobian_inv(cell,qp,alpha,j); 
            }
          } 
        } 
      }
    }
*/

  }

  if( this->outputNodeData ) { // nominal radius, store as nodal data that will be scattered and summed

    // Get the node data block container
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->pStateMgr->getStateInfoStruct()->getNodalDataBlock();
    Teuchos::RCP<Epetra_Vector> data = node_data->getLocalNodeVec();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Epetra_BlockMap> local_node_map = node_data->getLocalMap();

    int l_nV = this->numVertices;
    int l_nD = this->numDims;
    int blocksize = node_data->getBlocksize();

    int  node_var_offset;
    int  node_var_ndofs;
    int  node_weight_offset;
    int  node_weight_ndofs;
    node_data->getNDofsAndOffset(this->className + "_Node", node_var_offset, node_var_ndofs);
    node_data->getNDofsAndOffset(this->className + "_NodeWgt", node_weight_offset, node_weight_ndofs);

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
          for (int k=0; k < node_var_ndofs; ++k) 
//            data[global_node][k] += ADValue(maxCoord[k] - minCoord[k]) / 2.0;
            (*data)[local_node * blocksize + node_var_offset + k] += (maxCoord[k] - minCoord[k]) / 2.0;

          // save the weight (denominator)
          (*data)[local_node * blocksize + node_weight_offset] += 1.0;

      } // end anisotropic size field

      else // isotropic size field
        // Note: code assumes blocksize of blockmap is 1 + 1 = 2 - the last entry accumulates the weight
        for (int node = 0; node < l_nV; ++node) { // loop over all the "corners" of each element

          int global_node = wsElNodeID[cell][node]; // get the global id of this node

          int local_node = local_node_map->LID(global_node); // skip the node if it is not owned by me
          if(local_node < 0) continue;

          // save element radius, just a scalar
          for (int k=0; k < l_nD; ++k) {
//            data[global_node][k] += ADValue(maxCoord[k] - minCoord[k]) / 2.0;
            (*data)[local_node * blocksize + node_var_offset] += (maxCoord[k] - minCoord[k]) / 2.0;
            // save the weight (denominator)
            (*data)[local_node * blocksize + node_weight_offset] += 1.0;

          }


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
    Teuchos::RCP<Adapt::NodalDataBlock> node_data = this->pStateMgr->getStateInfoStruct()->getNodalDataBlock();
    Teuchos::RCP<Epetra_Vector> data = node_data->getOverlapNodeVec();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID = workset.wsElNodeID;
    Teuchos::RCP<const Epetra_BlockMap> overlap_node_map = node_data->getOverlapMap();

    int  node_var_offset;
    int  node_var_ndofs;
    int  node_weight_offset;
    int  node_weight_ndofs;
    node_data->getNDofsAndOffset(this->className + "_Node", node_var_offset, node_var_ndofs);
    node_data->getNDofsAndOffset(this->className + "_NodeWgt", node_weight_offset, node_weight_ndofs);

    // Build the exporter
    node_data->initializeExport();

    // do the export
    node_data->exportAddNodalDataBlock();

    int numNodes = overlap_node_map->NumMyElements();
    int blocksize = node_data->getBlocksize();

    // if isotropic, blocksize == 2 , if anisotropic blocksize == nDOF at node + 1
    // ndim if vector, ndim * ndim if tensor

    // all PEs divide the accumulated value(s) by the weights

    for (int overlap_node=0; overlap_node < numNodes; ++overlap_node)

      for (int k=0; k < node_var_ndofs; ++k) 
            (*data)[overlap_node * blocksize + node_var_offset + k] /=
                (*data)[overlap_node * blocksize + node_weight_offset];


    // Export the data from the local to overlapped decomposition
    // Divide the overlap field through by the weights
    // Store the overlapped vector data back in stk in the field "field_name"

    node_data->saveNodalDataState();

  }

}

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

  validPL->set<std::string>("Name", "", "Name of size field Evaluator");
  validPL->set<std::string>("Size Field Name", "", "Size field prefix");

  validPL->set<bool>("Output to File", true, "Whether size field info should be output to a file");
  validPL->set<bool>("Generate Cell Average", true, "Whether cell average field should be generated");
  validPL->set<bool>("Generate QP Values", true, "Whether values at the quadpoints should be generated");
  validPL->set<bool>("Generate Nodal Values", true, "Whether values at the nodes should be generated");
  validPL->set<bool>("Anisotropic Size Field", true, "Is this size field calculation anisotropic?");

  return validPL;
}

// **********************************************************************

