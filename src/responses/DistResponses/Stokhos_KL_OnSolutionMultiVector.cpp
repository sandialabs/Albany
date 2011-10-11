//
// Global function to encapsulate KL solution computation...
//

bool Stokhos::KL_OnSolutionMultiVector( const Teuchos::RCP<ENAT::SGNOXSolver>& App_sg, 
					const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& sg_u,
					const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
					const int NumKL,
					Teuchos::Array<double>& evals,
					Teuchos::RCP<Epetra_MultiVector>& evecs)
{

  Teuchos::RCP<EpetraExt::BlockVector> X;
  X = Teuchos::rcp(new EpetraExt::BlockVector(finalSolution->Map(),
						    *(App_sg->get_x_map())));
  sg_u->assignToBlockVector(*X);

  Teuchos::RCP<EpetraExt::BlockVector> X_ov = 
    App_sg->import_solution(*X);
  Teuchos::RCP<const EpetraExt::BlockVector> cX_ov = X_ov;

  // pceKL is object with member functions that explicitly call anasazi
  Stokhos::PCEAnasaziKL pceKL(cX_ov, *basis, NumKL);

  // Set parameters for anasazi
  Teuchos::ParameterList anasazi_params = pceKL.getDefaultParams();
  //anasazi_params.set("Num Blocks", 10);
  //anasazi_params.set("Step Size", 50);
  anasazi_params.set("Verbosity",  
		     Anasazi::FinalSummary + 
		     //Anasazi::StatusTestDetails + 
		     //Anasazi::IterationDetails + 
		     Anasazi::Errors + 
		     Anasazi::Warnings);

  // Self explanatory
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

