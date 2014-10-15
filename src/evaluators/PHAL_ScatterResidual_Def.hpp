//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
//#include "Kokkos_Cuda.hpp"
#include "Kokkos_View.hpp"
#include "Sacado.hpp"
#include "Kokkos_View_Fad.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
ScatterResidualBase<EvalT, Traits>::
ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldName;
  if (p.isType<std::string>("Scatter Field Name"))
    fieldName = p.get<std::string>("Scatter Field Name");
  else fieldName = "Scatter";

  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>
    (fieldName, dl->dummy));

  const Teuchos::ArrayRCP<std::string>& names =
    p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");

  if (p.isType<bool>("Vector Field"))
          vectorField = p.get<bool>("Vector Field");
  else vectorField = false;
  // scalar
  if (!vectorField) {
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
  else {
    valVec.resize(1);
    PHX::MDField<ScalarT,Cell,Node,Dim> mdf(names[0],dl->node_vector);
    valVec[0] = mdf;
    this->addDependentField(valVec[0]);
    numFieldsBase = dl->node_vector->dimension(2);
  }

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

   //Irina Debug
  // valVec_temp = Kokkos::View <double***, Kokkos::LayoutLeft, PHX::Device> ("valVec_temp", 50,8,3);
   Index=Kokkos::View <int***, PHX::Device>("Index_kokkos", dl->node_vector->dimension(0), dl->node_vector->dimension(1), dl->node_vector->dimension(2));

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (!vectorField) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
      this->utils.setFieldData(val[eq],fm);
    numNodes = val[0].dimension(1);
  }
  else {
    this->utils.setFieldData(valVec[0],fm);
    numNodes = valVec[0].dimension(1);
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase)

{
}
//************************************************************************
//Residual Kokkos functor
template <class MdFieldType, class TpetraType, class IndexType>
class ScatterToVector_resid {

 TpetraType f_;
 MdFieldType valVec_;
 IndexType wsElNodeEqID_;
 const int numNodes_;
 const int numFields_; 

 public:

 typedef PHX::Device device_type;

 ScatterToVector_resid ( TpetraType &f,
                   MdFieldType &valVec,
                   IndexType &wsElNodeEqID,
                   int numNodes,
                   int numFields)
 : f_(f)
 , valVec_(valVec)
 , wsElNodeEqID_(wsElNodeEqID) 
 , numNodes_(numNodes)
 , numFields_(numFields){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
  {
    for (int node = 0; node < numNodes_; ++node) {
      for (int dim = 0; dim < numFields_; dim++){
#ifdef NOATOMIC
          f_[wsElNodeEqID_(i,node,dim)]+=valVec_(i, node,dim);
#else
          Kokkos::atomic_fetch_add(&f_[wsElNodeEqID_(i,node,dim)],valVec_(i, node,dim));
#endif
      }
     }
   }
};

//************************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;

  //get nonconst (read and write) view of fT
  Teuchos::ArrayRCP<ST> f_nonconstView = fT->get1dViewNonConst();
#ifdef NO_KOKKOS_ALBANY
  if (this->vectorField) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->valVec[0])(cell,node,eq);
    } 
  } else {
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID[node][this->offset + eq]] += (this->val[eq])(cell,node);
    } 
  }
#else

// for  (std::size_t cell=0; cell < workset.numCells; ++cell )
//    for (std::size_t node = 0; node < this->numNodes; ++node)
//      for (std::size_t eq = 0; eq < numFields; eq++)
//           std::cout <<this->offset<< "     " << workset.wsElNodeEqID[cell][node][this->offset+eq] << "    " << workset.wsElNodeEqID_kokkos(cell,node,eq) <<std::endl;
//              this->Index(cell, node, eq) = workset.wsElNodeEqID[cell][node][this->offset + eq];

  //std::cout << workset.wsElNodeEqID[10][2][2] << "    " << workset.wsElNodeEqID_kokkos(10,2,2) <<std::endl;

 Kokkos::parallel_for(workset.numCells, ScatterToVector_resid< PHX::MDField<ScalarT,Cell,Node,Dim> , Teuchos::ArrayRCP<ST>, Kokkos::View<int***, PHX::Device> >(f_nonconstView, this->valVec[0], workset.wsElNodeEqID_kokkos, this->numNodes, numFields) );
#endif  
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}
// **********************************************************************
template < class FadType1, class TpetraType1, class TpetraType2, class MdFieldType, class IndexType >
class ScatterToVector_jacob {

 TpetraType1 Jac_;
 TpetraType2 f_;
 MdFieldType valVec_;
 IndexType wsElNodeEqID_;
 const int numNodes_;
 const int numFields_;

 public:

 typedef PHX::Device device_type;

 ScatterToVector_jacob ( TpetraType1 &Jac,
                   TpetraType2 &f,
                   MdFieldType &valVec,
                   IndexType &wsElNodeEqID,
                   int numNodes,
                   int numFields)
 : Jac_(Jac)
 , f_(f)
 , valVec_(valVec)
 , wsElNodeEqID_(wsElNodeEqID)
 , numNodes_(numNodes)
 , numFields_(numFields){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
  {
    int neq=2;
    int nunk=neq*numNodes_;
    int colT[nunk];
    const int ncol = 0; //FIXME
    //SparseRowView<CrsMatrix> row_view

    for (int node = 0; node < numNodes_; ++node) {
      for (int dim = 0; dim < neq; dim++){
        colT[neq * node + dim] = wsElNodeEqID_(i,node,dim);
      }
    }

    for (int node = 0; node < numNodes_; ++node) {
      for (int dim = 0; dim < numFields_; dim++){
         int row=wsElNodeEqID_(i,node,dim);
          Kokkos::SparseRowView<TpetraType1> row_view = Jac_.row(row);
          f_->sumIntoLocalValue(row, (valVec_(i,node,dim)).val());
          //FadType1 fad = valVec_(i,node,dim);
          Jac_.sumIntoValues (row, colT, nunk,  const_cast<double*>((valVec_(i,node, dim)).dx()), true);
         // Jac_->sumIntoLocalValues(row, colT, Teuchos::arrayView(&((valVec_(i,node,dim)).fastAccessDx(0)), nunk));
      }
     }
   }
};

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;
  typedef typename Tpetra_CrsMatrix::k_local_matrix_type  LocalMatrixType2 ;
  if (!JacT->isFillComplete())
      JacT->fillComplete();
  LocalMatrixType2 jacobian=JacT->getLocalMatrix();
  //const unsigned nrow = jacobian.numRows();
 // std::cout<< JacT;
  bool loadResid = Teuchos::nonnull(fT);
  LO rowT;
  Teuchos::Array<LO> colT;
#ifdef NO_KOKKOS_ALBANY
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    int neq = nodeID[0].size();
    int nunk = neq*this->numNodes;
    colT.resize(nunk);
//std::cout << neq << "   " << nunk <<"   " <<  this->numNodes << "    " <<std::endl;
    // Local Unks: Loop over nodes in element, Loop over equations per node

    for (unsigned int node_col=0, i=0; node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node_col + eq_col] =  nodeID[node_col][eq_col];
      }
    }

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
         
       rowT = nodeID[node][this->offset + eq];
        if (loadResid) {
          fT->sumIntoLocalValue(rowT, ((this->valVec[0])(cell,node,eq)).val());
        }


       if (this->vectorField){
         if ((((this->valVec[0])(cell,node,eq))).hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
               for (unsigned int lunk=0; lunk<nunk; lunk++)
                 JacT->sumIntoLocalValues(colT[lunk], Teuchos::arrayView(&rowT, 1), Teuchos::arrayView(&(((this->valVec[0])(cell,node,eq)).fastAccessDx(lunk)), 1));
                }
            else {
             // Sum Jacobian entries all at once
              JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(((this->valVec[0])(cell,node,eq)).fastAccessDx(0)), nunk));
           }
         }//hase fast accesss
           
       }
       else {
          if ((((this->valVec[eq])(cell,node,eq))).hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int lunk=0; lunk<nunk; lunk++)
               JacT->sumIntoLocalValues(colT[lunk], Teuchos::arrayView(&rowT, 1), Teuchos::arrayView(&(((this->valVec[eq])(cell,node,eq)).fastAccessDx(lunk)), 1));
              }
            else {
             // Sum Jacobian entries all at once
             JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(((this->valVec[eq])(cell,node,eq)).fastAccessDx(0)), nunk));
             }
          }//hase fast accesss
      }
     }
    }
   }

 #else
//for  (std::size_t cell=0; cell < workset.numCells; ++cell )
//    for (std::size_t node = 0; node < this->numNodes; ++node)
//      for (std::size_t eq = 0; eq < numFields; eq++)
//              this->Index(cell, node, eq) = workset.wsElNodeEqID[cell][node][this->offset + eq];

//const unsigned nrow = jacobian.numRows();
//        std::cout << "JacobianGraph[ "
//                  << jacobian.numRows() << " x " << jacobian.numCols()
//                  << " ] {" << std::endl ;

/*        for ( unsigned irow = 0 ; irow < nrow ; ++irow ) {
          std::cout << "  row[" << irow << "]{" ;
          const unsigned entry_end = jacobian.graph.row_map(irow+1);
          for ( unsigned entry = jacobian.graph.row_map(irow) ; entry < entry_end ; ++entry ) {
            std::cout << " " << jacobian.graph.entries(entry);
          }
          std::cout << " }" << std::endl ;
        }
        std::cout << "}" << std::endl ;
*/
 Kokkos::parallel_for(workset.numCells, ScatterToVector_jacob< ScalarT, Tpetra_LocalMatrixType, Teuchos::RCP<Tpetra_Vector> , PHX::MDField<ScalarT,Cell,Node,Dim>, Kokkos::View<int***, PHX::Device> >(jacobian, fT, this->valVec[0], workset.wsElNodeEqID_kokkos, this->numNodes, numFields) );
#endif

JacT->resumeFill();

//Initial Tpetra code
/*
          if (this->vectorField) valptr = &((this->valVec[0])(cell,node,eq));
          else                   valptr = &(this->val[eq])(cell,node);

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
*/
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Tangent,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  ScalarT *valptr;


  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {

        //Innitial Tpetra code
       /* 
          if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
          else                   valptr = &(this->val[eq])(cell,node);

        int row = nodeID[node][this->offset + eq];

        if (Teuchos::nonnull(fT))
          fT->sumIntoLocalValue(row, valptr->val());

	if (Teuchos::nonnull(JVT))
	  for (int col=0; col<workset.num_cols_x; col++)
	    JVT->sumIntoLocalValue(row, col, valptr->dx(col));

	if (Teuchos::nonnull(fpT)) 
	  for (int col=0; col<workset.num_cols_p; col++)
	    fpT->sumIntoLocalValue(row, col, valptr->dx(col+workset.param_offset));
        */

        int row = nodeID[node][this->offset + eq];

        if (this->vectorField){
          if (Teuchos::nonnull(fT))
          fT->sumIntoLocalValue(row, ((this->valVec[0])(cell,node,eq)).val());

        if (Teuchos::nonnull(JVT))
          for (int col=0; col<workset.num_cols_x; col++)
            JVT->sumIntoLocalValue(row, col, ((this->valVec[0])(cell,node,eq)).dx(col));

        if (Teuchos::nonnull(fpT)) 
          for (int col=0; col<workset.num_cols_p; col++)
            fpT->sumIntoLocalValue(row, col, ((this->valVec[0])(cell,node,eq)).dx(col+workset.param_offset));

        }
        else
        {
        if (this->vectorField){
          if (Teuchos::nonnull(fT))
          fT->sumIntoLocalValue(row, ((this->val[eq])(cell,node)).val());
        
        if (Teuchos::nonnull(JVT))
          for (int col=0; col<workset.num_cols_x; col++)
            JVT->sumIntoLocalValue(row, col, ((this->val[eq])(cell,node)).dx(col));
         
        if (Teuchos::nonnull(fpT))
          for (int col=0; col<workset.num_cols_p; col++)
            fpT->sumIntoLocalValue(row, col, ((this->val[eq])(cell,node)).dx(col+workset.param_offset));

             
         }

      }
    }
  }
}
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.VpT->getNumVectors();
//  ScalarT *valptr;

  if (trans) {

    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<int>& dist_param_index =
        workset.dist_param_index[cell];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (int i=0; i<num_deriv; i++) {
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            for (std::size_t eq = 0; eq < numFields; eq++) {
            //  if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
            //  else                   valptr = &(this->val[eq])(cell,node);
            //  val += valptr->dx(i)*local_Vp[col][i];
               if (this->vectorField)
                    val += ((this->valVec[0])(cell,node,eq)).dx(i)*local_Vp[col][i];
               else
                    val += ((this->val[eq])(cell,node)).dx(i)*local_Vp[col][i];
            }
          }
          const int row = dist_param_index[i];
          fpVT->sumIntoLocalValue(row, col, val);
        }
      }
    }

  }

  else {

    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID =
        workset.wsElNodeEqID[cell];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
      //    if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
      //    else                   valptr = &(this->val[eq])(cell,node);
      //    const int row = nodeID[node][this->offset + eq];
      //    for (int col=0; col<num_cols; col++) {
      //      double val = 0.0;
      //      for (int i=0; i<num_deriv; ++i)
      //        val += valptr->dx(i)*local_Vp[col][i];
      //      fpV->SumIntoMyValue(row, col, val);
      //    }
           const int row = nodeID[node][this->offset + eq];
           for (int col=0; col<num_cols; col++) {
            double val = 0.0;
           if (this->vectorField){
             for (int i=0; i<num_deriv; ++i)
                 val += ((this->valVec[0])(cell,node,eq)).dx(i)*local_Vp[col][i];
           }
           else{
              for (int i=0; i<num_deriv; ++i)
                 val += ((this->val[eq])(cell,node)).dx(i)*local_Vp[col][i];
           }
           fpVT->sumIntoLocalValue(row, col, val);
        }
      }
    }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG_MP
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::SGResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
//  ScalarT *valptr;

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
         // if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
         // else                   valptr = &(this->val[eq])(cell,node);
        if (this->vectorField){
          for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][this->offset + eq]] += ((this->valVec[0])(cell,node,eq)).coeff(block);
           }
        else{
           for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][this->offset + eq]] += ((this->val[eq])(cell,node)).coeff(block);
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > Jac =
    workset.sg_Jac;
//  ScalarT *valptr;

  int row, lcol, col;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
      //    if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
      //    else                   valptr = &(this->val[eq])(cell,node);

        row = nodeID[node][this->offset + eq];
        int neq = nodeID[node].size();

        if (this->vectorField){

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->valVec[0])(cell,node,eq)).val().coeff(block));
        }

        // Check derivative array is nonzero
        if ((this->valVec[0])(cell,node,eq)).hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {
                c = ((this->valVec[0])(cell,node,eq)).fastAccessDx(lcol).coeff(block);
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

       else{

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->val[eq](cell,node)).val().coeff(block));
        }
        if (((this->val[eq])(cell,node)).hasFastAccess()) {
           for (unsigned int node_col=0; node_col<this->numNodes; node_col++){ 
             for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;
              col =  nodeID[node_col][eq_col];
              for (int block=0; block<nblock_jac; block++) {
                c = ((this->val[eq])(cell,node)).fastAccessDx(lcol).coeff(block);
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
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::SGTangent,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::SGTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > JV = workset.sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > fp = workset.sg_fp;
//  ScalarT *valptr;

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

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        //  if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        //  else                   valptr = &(this->val[eq])(cell,node);

        int row = nodeID[node][this->offset + eq];

       if (this->vectorField){
        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->valVec[0])(cell,node,eq)).val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, ((this->valVec[0])(cell,node,eq)).dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, ((this->valVec[0])(cell,node,eq)).dx(col+workset.param_offset).coeff(block));
       }
       else{
         if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->val[eq])(cell,node)).val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, ((this->val[eq])(cell,node)).dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, ((this->val[eq])(cell,node)).dx(col+workset.param_offset).coeff(block));


       } 
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
//  ScalarT *valptr;

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
         // if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        //  else                   valptr = &(this->val[eq])(cell,node);
        if (this->vectorField){
          for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][this->offset + eq]] += ((this->valVec[0])(cell,node,eq)).coeff(block);
        }
        else{
           for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][this->offset + eq]] += ((this->val[eq])(cell,node)).coeff(block);
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > Jac =
    workset.mp_Jac;
//  ScalarT *valptr;

  int row, lcol, col;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
         // if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
         // else                   valptr = &(this->val[eq])(cell,node);

        row = nodeID[node][this->offset + eq];
        int neq = nodeID[node].size();

       if (this->vectorField){

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->valVec[0])(cell,node,eq)).val().coeff(block));
        }

        // Check derivative array is nonzero
        if (((this->valVec[0])(cell,node,eq)).hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {
                c = ((this->valVec[0])(cell,node,eq)).fastAccessDx(lcol).coeff(block);
                (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
              }
            } // column equations
          } // column nodes
        } // has fast access
       }
       else
       {
          if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->val[eq])(cell,node)).val().coeff(block));
        }

        // Check derivative array is nonzero
        if (((this->val[eq])(cell,node)).hasFastAccess()) { 
         for (unsigned int node_col=0; node_col<this->numNodes; node_col++){  
           for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;
              col =  nodeID[node_col][eq_col];
              for (int block=0; block<nblock_jac; block++) {
                c = ((this->val[eq])(cell,node)).fastAccessDx(lcol).coeff(block);
                (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
              }
            } // column equations
          } // column nodes
        } // has fast access


       } 
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::MPTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > JV = workset.mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > fp = workset.mp_fp;
 // ScalarT *valptr;

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

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
         // if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
         // else                   valptr = &(this->val[eq])(cell,node);

        int row = nodeID[node][this->offset + eq];

        if (this->vectorField) {

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->valVec[0])(cell,node,eq)).val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, ((this->valVec[0])(cell,node,eq)).dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, ((this->valVec[0])(cell,node,eq)).dx(col+workset.param_offset).coeff(block));
        }
        else{
         if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, ((this->val[eq])(cell,node)).val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, ((this->val[eq])(cell,node)).dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, ((this->val[eq])(cell,node)).dx(col+workset.param_offset).coeff(block));

        }
      }
    }
  }
}
#endif //ALBANY_SG_MP

}

