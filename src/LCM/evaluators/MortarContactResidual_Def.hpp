//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
MortarContactResidualBase<EvalT, Traits>::
MortarContactResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :

  meshSpecs      (p.get<Teuchos::RCP<Albany::MeshSpecsStruct>>("Mesh Specs Struct")),
  // The array of names of all the master side sets in the problem
  masterSideNames (p.get<Teuchos::ArrayRCP<std::string>>("Master Side Set Names")), 
  slaveSideNames (p.get<Teuchos::ArrayRCP<std::string>>("Slave Side Set Names")), 

  // The array of sidesets to process
  sideSetIDs (p.get<Teuchos::ArrayRCP<std::string>>("Sideset IDs")), 

  // Node coords
  coordVec       (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector) 

{
  std::string fieldName;

  // Allow the user to name this instance of the mortar projection operation
  // The default is just "MortarProjection"
  if (p.isType<std::string>("Projection Field Name"))
    fieldName = p.get<std::string>("Projection Field Name");
  else fieldName = "MortarProjection";

  mortar_projection_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  // Get the array of residual quantities that we need to project into the integration space
  // These are the physics residuals that this class evaluates
  const Teuchos::ArrayRCP<std::string>& names =
    p.get< Teuchos::ArrayRCP<std::string>>("Residual Names");

  tensorRank = p.get<int>("Tensor Rank");

  // scalar
  if (tensorRank == 0 ) {
    numFieldsBase = names.size();
    const std::size_t num_val = numFieldsBase;
    val.resize(num_val);
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      PHX::MDField<ScalarT,Cell,Node> mdf(names[eq],dl->node_scalar);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }
  }
  // vector
  else
  if (tensorRank == 1 ) {
    valVec.resize(1);
    PHX::MDField<ScalarT,Cell,Node,Dim> mdf(names[0],dl->node_vector);
    valVec[0] = mdf;
    this->addDependentField(valVec[0]);
    numFieldsBase = dl->node_vector->dimension(2);
  }
  // tensor
  else
  if (tensorRank == 2 ) {
    valTensor.resize(1);
    PHX::MDField<ScalarT,Cell,Node,Dim,Dim> mdf(names[0],dl->node_tensor);
    valTensor[0] = mdf;
    this->addDependentField(valTensor[0]);
    numFieldsBase = (dl->node_tensor->dimension(2))*(dl->node_tensor->dimension(3));
  }

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

  // Tell PHAL which field we are evaluating. Note that this is actually a dummy, as we fill in the
  // residual vector directly. This tells PHAL to call this evaluator to satisfy this dummy field.
  this->addEvaluatedField(*mortar_projection_operation);

  this->setName(fieldName+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void MortarContactResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
      this->utils.setFieldData(val[eq],fm);
    numNodes = val[0].dimension(1);
  }
  else 
  if (tensorRank == 1) {
    this->utils.setFieldData(valVec[0],fm);
    numNodes = valVec[0].dimension(1);
  }
  else 
  if (tensorRank == 2) {
    this->utils.setFieldData(valTensor[0],fm);
    numNodes = valTensor[0].dimension(1);
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::Residual,Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase)

{
}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // We assume global search is done. Perform local search to pair up each master segment in the element
  // workset with the slave segments that it may potentially interact with

  // Then, form the mortar integration space


  // No work to do
  if(workset.sideSets == Teuchos::null || 
     this->masterSideNames.size() == 0 || 
     this->slaveSideNames.size() == 0 || 
     this->sideSetIDs.size() == 0)
    return;

  const Albany::SideSetList& ssList = *(workset.sideSets);

  for(std::size_t i = 0; i < this->sideSetIDs.size(); i++){

    Albany::SideSetList::const_iterator it = ssList.find(this->sideSetIDs[i]);

      if(it == ssList.end()) continue; // This sideset does not exist in this workset - try the next one

/*
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
       for (std::size_t node=0; node < numNodes; ++node)
         for (std::size_t dim=0; dim < 3; ++dim)
             neumann(cell, node, dim) = 0.0; // zero out the accumulation vector
*/

      const std::vector<Albany::SideStruct>& sideSet = it->second;

      // Loop over the sides that form the boundary condition

      for (std::size_t side=0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

        const bool on_boundary = false; // node does not lie on a boundary.
        const int  nid = 0; // need accessor for node global id.
        const int  print_level = 0;
        const int num_dofs_per_node = 2;

//        MOERTEL::Node node( nid, x, num_dofs_per_node, dof, on_boundary, print_level );
 
//        interface.AddNode(node,side);

        // Get the data that corresponds to the side. 

        const int elem_GID = sideSet[side].elem_GID; // GID of the element that contains the master segment
        const int elem_LID = sideSet[side].elem_LID; // LID (numbered from zero) id of the master segment on this processor
        const int elem_side = sideSet[side].side_local_id; // which edge of the element the side is (cf. exodus manual)?
        const int elem_block = sideSet[side].elem_ebIndex; // which  element block is the element in?

      }
    }



  // Then assemble the DOFs (flux, traction) at the slaves into the master side local elements

#if 0  // Here is the assemble code, more or less

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;

  //get nonconst (read and write) view of fT
  Teuchos::ArrayRCP<ST> f_nonconstView = fT->get1dViewNonConst();

  if (this->tensorRank == 0) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->val[eq])(cell,node);
    }
  } else 
  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->valVec[0])(cell,node,eq);
    }
  } else
  if (this->tensorRank == 2) {
    int numDims = this->valTensor[0].dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t i = 0; i < numDims; i++)
          for (std::size_t j = 0; j < numDims; j++)
            f_nonconstView[nodeID[node][this->offset + i*numDims + j]] += (this->valTensor[0])(cell,node,i,j);
  
    }
  }
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  ScalarT *valptr;

  bool loadResid = Teuchos::nonnull(fT);
  LO rowT;
  Teuchos::Array<LO> colT;

  int neq = workset.wsElNodeEqID[0][0].size();
  int nunk = neq*this->numNodes;
  colT.resize(nunk);

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];
    // Local Unks: Loop over nodes in element, Loop over equations per node

    for (unsigned int node_col=0, i=0; node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node_col + eq_col] =  nodeID[node_col][eq_col];
      }
    }

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
          else
          if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
          else
          if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        rowT = nodeID[node][this->offset + eq];
        if (loadResid) {
          fT->sumIntoLocalValue(rowT, valptr->val());
        }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          if (workset.is_adjoint) {
            // Sum Jacobian transposed
            for (unsigned int lunk=0; lunk<nunk; lunk++)
              JacT->sumIntoLocalValues(colT[lunk], Teuchos::arrayView(&rowT, 1), Teuchos::arrayView(&(valptr->fastAccessDx(lunk)), 1));
          }
          else {
            // Sum Jacobian entries all at once
            JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr->fastAccessDx(0)), nunk));
          }
        } // has fast access
      }
    }
  }
#endif
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::Tangent, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::Tangent,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::Tangent, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  ScalarT *valptr;


  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
          else
          if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
          else
          if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        int row = nodeID[node][this->offset + eq];

        if (Teuchos::nonnull(fT))
          fT->sumIntoLocalValue(row, valptr->val());

	if (Teuchos::nonnull(JVT))
	  for (int col=0; col<workset.num_cols_x; col++)
	    JVT->sumIntoLocalValue(row, col, valptr->dx(col));

	if (Teuchos::nonnull(fpT)) 
	  for (int col=0; col<workset.num_cols_p; col++)
	    fpT->sumIntoLocalValue(row, col, valptr->dx(col+workset.param_offset));
      }
    }
  }
#endif
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.VpT->getNumVectors();
  ScalarT *valptr;

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  if (trans) {
    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double>>& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size()/numFields;
      for (int i=0; i<num_deriv; i++) {
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            for (std::size_t eq = 0; eq < numFields; eq++) {
              if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
              else
              if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
              else
              if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

              val += valptr->dx(i)*local_Vp[node*numFields+eq][col];
            }
          }
          const LO row = wsElDofs((int)cell,i,0);
          if(row >=0)
            fpVT->sumIntoLocalValue(row, col, val);
        }
      }
    }

  }

  else {

    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID =
        workset.wsElNodeEqID[cell];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double>>& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
          else
          if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
          else
          if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

          const int row = nodeID[node][this->offset + eq];
          for (int col=0; col<num_cols; col++) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i)
              val += valptr->dx(i)*local_Vp[i][col];
            fpVT->sumIntoLocalValue(row, col, val);
          }
        }
      }
    }

  }
#endif
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::SGResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  ScalarT *valptr;

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
        else
        if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
        else
        if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID[node][this->offset + eq]] += valptr->coeff(block);
      }
    }
  }
#endif
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>> Jac =
    workset.sg_Jac;
  ScalarT *valptr;

  int row, lcol, col;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
        else
        if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
        else
        if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        row = nodeID[node][this->offset + eq];
        int neq = nodeID[node].size();

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));
        }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {
                c = valptr->fastAccessDx(lcol).coeff(block);
                if (workset.is_adjoint) {
                  (*Jac)[block].SumIntoMyValues(col, 1, &c, &row);
                }
                else {
                  (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
                }
              }
            } // column equations
          } // column nodes
        } // has fast access
      }
    }
  }
#endif
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::SGTangent,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::SGTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > JV = workset.sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > fp = workset.sg_fp;
  ScalarT *valptr;

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "One of sg_f, sg_JV, or sg_fp must be non-null! " <<
                       std::endl);

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
        else
        if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
        else
        if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        int row = nodeID[node][this->offset + eq];

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr->dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset).coeff(block));
      }
    }
  }
#endif
}
#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  ScalarT *valptr;

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
        else
        if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
        else
        if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);
        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID[node][this->offset + eq]] += valptr->coeff(block);
      }
    }
  }
#endif
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix>> Jac =
    workset.mp_Jac;
  ScalarT *valptr;

  int row, lcol, col;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
        else
        if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
        else
        if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        row = nodeID[node][this->offset + eq];
        int neq = nodeID[node].size();

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));
        }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {
                c = valptr->fastAccessDx(lcol).coeff(block);
                (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
              }
            } // column equations
          } // column nodes
        } // has fast access
      }
    }
  }
#endif
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
MortarContactResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : MortarContactResidualBase<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl),
  numFields(MortarContactResidualBase<PHAL::AlbanyTraits::MPTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
preEvaluate(typename Traits::PreEvalData d){

// Put global search in here

}

template<typename Traits>
void MortarContactResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#if 0
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > JV = workset.mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > fp = workset.mp_fp;
  ScalarT *valptr;

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "One of mp_f, mp_JV, or mp_fp must be non-null! " <<
                       std::endl);

  int numDim=0;
  if(this->tensorRank==2)
    numDim = this->valTensor[0].dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int>>& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->tensorRank == 0) valptr = &(this->val[eq])(cell,node);
        else
        if (this->tensorRank == 1) valptr = &((this->valVec[0])(cell,node,eq));
        else
        if (this->tensorRank == 2) valptr = &(this->valTensor[0])(cell,node, eq/numDim, eq%numDim);

        int row = nodeID[node][this->offset + eq];

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr->dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset).coeff(block));
      }
    }
  }
#endif
}
#endif

}

