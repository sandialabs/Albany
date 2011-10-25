//#ifdef HAVE_STOKHOS_ANASAZI
//
// Global function to encapsulate KL solution computation...
//
//      if (SG_Method != SG_NI) {
//

int Stokhos::KL_Expansion(Teuchos::RCP<Stokhos::SGModelEvaluator> sg_model, 
			  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly >sg_u,
			  int NumKL,
			  Teuchos::Array<double>& evals,
			  Teuchos::RCP<Epetra_MultiVector>& evecs)
{

	// Compute KL expansion of stochastic galerkin solution, sg_u
	Teuchos::RCP<EpetraExt::BlockVector> X;
	X = Teuchos::rcp(new EpetraExt::BlockVector(finalSolution->Map(),
						    *(sg_model->get_x_map())));
	sg_u->assignToBlockVector(*X);
	Teuchos::RCP<EpetraExt::BlockVector> X_ov = 
	  sg_model->import_solution(*X);
	Teuchos::RCP<const EpetraExt::BlockVector> cX_ov = X_ov;
	Stokhos::PCEAnasaziKL pceKL(cX_ov, *basis, NumKL);
	Teuchos::ParameterList anasazi_params = pceKL.getDefaultParams();
	//anasazi_params.set("Num Blocks", 10);
	//anasazi_params.set("Step Size", 50);
	anasazi_params.set("Verbosity",  
			   Anasazi::FinalSummary + 
			   //Anasazi::StatusTestDetails + 
			   //Anasazi::IterationDetails + 
			   Anasazi::Errors + 
			   Anasazi::Warnings);
	bool result = pceKL.computeKL(anasazi_params);
	if (!result)
	  utils.out() << "KL Eigensolver did not converge!" << std::endl;

// Retrieve evals/evectors into return argument slots...
	evals = pceKL.getEigenvalues();
	evecs = pceKL.getEigenvectors();

}

//	utils.out() << "KL eigenvalues = " << std::endl;
//	for (int i=0; i<evals.size(); i++)
//	  utils.out() << std::sqrt(evals[i]) << std::endl;
//
//         } // if (SG_Method != SG_NI)
// #endif
