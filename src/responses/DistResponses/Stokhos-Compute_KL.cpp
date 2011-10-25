//
// Global function to encapsulate KL solution computation...
//

bool Stokhos::KL_OnSolutionMultiVector( Teuchos::RCP<Stokhos::SGModelEvaluator> sg_model, 
				       Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> sg_u,
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
	  {
	    utils.out() << "KL Eigensolver did not converge!" << std::endl;
	    return result;
	  }
// Retrieve evals/evectors into return argument slots...
	evals = pceKL.getEigenvalues();
	evecs = pceKL.getEigenvectors();
	return result;
}

