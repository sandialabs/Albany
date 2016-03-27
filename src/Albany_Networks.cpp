//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_Networks.hpp"

//IK, 9/12/14: right now this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

void 
Albany::ReactorNetworkModel::
evalModel(
  const Teuchos::Array<EpetraExt::ModelEvaluator::InArgs>& model_inargs, 
  const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs>& model_outargs,
  const EpetraExt::ModelEvaluator::InArgs& network_inargs, 
  const EpetraExt::ModelEvaluator::OutArgs& network_outargs,
  const Teuchos::Array<int>& n_p,
  const Teuchos::Array<int>& n_g,
  const Teuchos::Array< Teuchos::RCP<Epetra_Vector> >& p,
  const Teuchos::Array< Teuchos::RCP<Epetra_Vector> >& g,
  const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dgdp,
  const Teuchos::Array<EpetraExt::ModelEvaluator::EDerivativeMultiVectorOrientation>& dgdp_layout,
  const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs::sg_vector_t>& p_sg,
  const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs::sg_vector_t>& g_sg,
  const Teuchos::Array<Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> >& dgdp_sg,
  const Teuchos::Array<EpetraExt::ModelEvaluator::EDerivativeMultiVectorOrientation>& dgdp_sg_layout) const
{
     
  // f
  Teuchos::RCP<Epetra_Vector> f = network_outargs.get_f();
  if (f != Teuchos::null) {
    // g[0]->Print(std::cout << "g[0] = " << std::endl);
    // g[1]->Print(std::cout << "g[1] = " << std::endl);
    f->PutScalar(0.0);
    for (int i=0; i<n; i++) {
      (*f)[i]     = (*g[0])[i+n] - (*g[1])[i];
      (*f)[i+n]   = (*g[0])[i]   - (*g[1])[i+n];
      (*f)[i+2*n] = (*p[0])[i+n] + (*p[1])[i];
      (*f)[i+3*n] = (*p[0])[i]   + (*p[1])[i+n];
    }
    // f->Print(std::cout << "f = " << std::endl);
  }

  // W
  Teuchos::RCP<Epetra_Operator> W = network_outargs.get_W();
  if (W != Teuchos::null) {
    // dgdp[0]->Print(std::cout << "dgdp[0] = " << std::endl);
    // dgdp[1]->Print(std::cout << "dgdp[1] = " << std::endl);
    Teuchos::RCP<Epetra_CrsMatrix> W_crs = 
      Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(W, true);
    W_crs->PutScalar(0.0);
    int row, col;
    double val;

    // Block row 1
    for (int i=0; i<n; i++) {
      row = i; 
	  
      // (1,1) block
      for (int j=0; j<n; j++) {
	col = j; 
	val = (*dgdp[0])[j][i+n];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      // (1,2) block
      for (int j=0; j<n; j++) {
	col = n+j; 
	val = (*dgdp[0])[j+n][i+n];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      // (1,3) block
      for (int j=0; j<n; j++) {
	col = 2*n+j; 
	val = -(*dgdp[1])[j][i];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      // (1,4) block
      for (int j=0; j<n; j++) {
	col = 3*n+j; 
	val = -(*dgdp[1])[j+n][i];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

    }

    // Block row 2
    for (int i=0; i<n; i++) {
      row = n+i; 
	  
      // (1,1) block
      for (int j=0; j<n; j++) {
	col = j; 
	val = (*dgdp[0])[j][i];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      // (1,2) block
      for (int j=0; j<n; j++) {
	col = n+j; 
	val = (*dgdp[0])[j+n][i];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      // (1,3) block
      for (int j=0; j<n; j++) {
	col = 2*n+j; 
	val = -(*dgdp[1])[j][i+n];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      // (1,4) block
      for (int j=0; j<n; j++) {
	col = 3*n+j; 
	val = -(*dgdp[1])[j+n][i+n];
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

    }

    // Block row 3
    for (int i=0; i<n; i++) {
      row = 2*n+i; 
	  
      // (3,1) block -- zero

      // (3,2) block
      col = n+i; 
      val = 1.0;
      W_crs->ReplaceGlobalValues(row, 1, &val, &col);

      // (3,3) block
      col = 2*n+i; 
      val = 1.0;
      W_crs->ReplaceGlobalValues(row, 1, &val, &col);

      // (3,4) block -- zero
    }

    // Block row 4
    for (int i=0; i<n; i++) {
      row = 3*n+i; 
	  
      // (4,1) block
      col = 0+i; 
      val = 1.0;
      W_crs->ReplaceGlobalValues(row, 1, &val, &col);

      // (4,2) block -- zero

      // (4,3) block -- zero

      // (4,4) block
      col = 3*n+i; 
      val = 1.0;
      W_crs->ReplaceGlobalValues(row, 1, &val, &col);
    }

    // W_crs->Print(std::cout << "W_crs =" << std::endl);
  }
      
  // f_sg
  if (network_outargs.supports(EpetraExt::ModelEvaluator::OUT_ARG_f_sg)) {
    EpetraExt::ModelEvaluator::OutArgs::sg_vector_t f_sg = 
      network_outargs.get_f_sg();
    if (f_sg != Teuchos::null) {
      // std::cout << "g_sg[0] = " << std::endl << *(g_sg[0]) << std::endl;
      // std::cout << "g_sg[1] = " << std::endl << *(g_sg[1]) << std::endl;
      f_sg->init(0.0);
      for (int block=0; block<f_sg->size(); block++) {
	for (int i=0; i<n; i++) {
	  (*f_sg)[block][i]     = (*g_sg[0])[block][i+n] - (*g_sg[1])[block][i];
	  (*f_sg)[block][i+n]   = (*g_sg[0])[block][i]   - (*g_sg[1])[block][i+n];
	  (*f_sg)[block][i+2*n] = (*p_sg[0])[block][i+n] + (*p_sg[1])[block][i];
	  (*f_sg)[block][i+3*n] = (*p_sg[0])[block][i]   + (*p_sg[1])[block][i+n];
	}
      }
    }
  }
      
  // W_sg
  if (network_outargs.supports(EpetraExt::ModelEvaluator::OUT_ARG_W_sg)) {
    EpetraExt::ModelEvaluator::OutArgs::sg_operator_t W_sg = 
      network_outargs.get_W_sg();
    if (W_sg != Teuchos::null) {
      // std::cout << "dgdp_sg[0] = " << std::endl << *(dgdp_sg[0]) << std::endl;
      // std::cout << "dgdp_sg[1] = " << std::endl << *(dgdp_sg[1]) << std::endl;
      int row, col;
      double val;
      for (int block=0; block<W_sg->size(); block++) {
	Teuchos::RCP<Epetra_CrsMatrix> W_crs = 
	  Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(
	    W_sg->getCoeffPtr(block), true);
	    
	W_crs->PutScalar(0.0);
	    
	// Block row 1
	for (int i=0; i<n; i++) {
	  row = i; 
	  
	  // (1,1) block
	  for (int j=0; j<n; j++) {
	    col = j; 
	    val = (*dgdp_sg[0])[block][j][i+n];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	      
	  // (1,2) block
	  for (int j=0; j<n; j++) {
	    col = n+j; 
	    val = (*dgdp_sg[0])[block][j+n][i+n];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	      
	  // (1,3) block
	  for (int j=0; j<n; j++) {
	    col = 2*n+j; 
	    val = -(*dgdp_sg[1])[block][j][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	  // (1,4) block
	  for (int j=0; j<n; j++) {
	    col = 3*n+j; 
	    val = -(*dgdp_sg[1])[block][j+n][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }

	}

	// Block row 2
	for (int i=0; i<n; i++) {
	  row = n+i; 
	      
	  // (1,1) block
	  for (int j=0; j<n; j++) {
	    col = j; 
	    val = (*dgdp_sg[0])[block][j][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	      
	  // (1,2) block
	  for (int j=0; j<n; j++) {
	    col = n+j; 
	    val = (*dgdp_sg[0])[block][j+n][i];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	      
	  // (1,3) block
	  for (int j=0; j<n; j++) {
	    col = 2*n+j; 
	    val = -(*dgdp_sg[1])[block][j][i+n];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	      
	  // (1,4) block
	  for (int j=0; j<n; j++) {
	    col = 3*n+j; 
	    val = -(*dgdp_sg[1])[block][j+n][i+n];
	    W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	  }
	      
	}

      }

      // Rows 3 and 4, only the mean block
      Teuchos::RCP<Epetra_CrsMatrix> W_crs = 
	Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(
	  W_sg->getCoeffPtr(0), true);

      // Block row 3
      for (int i=0; i<n; i++) {
	row = 2*n+i; 
	  
	// (3,1) block -- zero

	// (3,2) block
	col = n+i; 
	val = 1.0;
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	    
	// (3,3) block
	col = 2*n+i; 
	val = 1.0;
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	    
	// (3,4) block -- zero
      }
	  
      // Block row 4
      for (int i=0; i<n; i++) {
	row = 3*n+i; 
	    
	// (4,1) block
	col = 0+i; 
	val = 1.0;
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
	    
	// (4,2) block -- zero
	    
	// (4,3) block -- zero
	    
	// (4,4) block
	col = 3*n+i; 
	val = 1.0;
	W_crs->ReplaceGlobalValues(row, 1, &val, &col);
      }

      //std::cout << "W_sg = " << *W_sg << std::endl;
    }
  }
}


