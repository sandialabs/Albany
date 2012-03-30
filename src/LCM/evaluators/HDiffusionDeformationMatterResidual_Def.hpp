/********************************************************************\
 *            Albany, Copyright (2010) Sandia Corporation             *
 *                                                                    *
 * Notice: This computer software was prepared by Sandia Corporation, *
 * hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
 * the Department of Energy (DOE). All rights in the computer software*
 * are reserved by DOE on behalf of the United States Government and  *
 * the Contractor as provided in the Contract. You are authorized to  *
 * use this computer software for Governmental purposes but it is not *
 * to be released or distributed to the public. NEITHER THE GOVERNMENT*
 * NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
 * ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
 * including this sentence must appear on any copies of this software.*
 *    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  HDiffusionDeformationMatterResidual<EvalT, Traits>::
  HDiffusionDeformationMatterResidual(const Teuchos::ParameterList& p) :
    wBF         (p.get<std::string>                ("Weighted BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
	wGradBF     (p.get<std::string>                ("Weighted Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
	GradBF      (p.get<std::string>                   ("Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
    Dstar (p.get<std::string>                   ("Effective Diffusivity Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    DL   (p.get<std::string>                       ("Diffusion Coefficient Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    Clattice (p.get<std::string>                   ("QP Variable Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	eqps (p.get<std::string>                   ("eqps Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    eqpsFactor (p.get<std::string>                   ("Strain Rate Factor Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	Ctrapped (p.get<std::string>                   ("Trapped Concentration Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	Ntrapped (p.get<std::string>                   ("Trapped Solvent Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    CLGrad       (p.get<std::string>               ("Gradient QP Variable Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
	stressGrad       (p.get<std::string>               ("Gradient Hydrostatic Stress Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
//    Source      (p.get<std::string>                ("Source Name"),
//		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
//	MechSource      (p.get<std::string>            ("Mechanical Source Name"),
//		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    DefGrad      (p.get<std::string>               ("Deformation Gradient Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
	Pstress      (p.get<std::string>               ("Stress Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
	weights       (p.get<std::string>                   ("Weights Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	tauFactor  (p.get<std::string>                   ("Tau Contribution Name"),
		 		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	deltaTime (p.get<std::string>                  ("Delta Time Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Workset Scalar Data Layout")),
    TResidual   (p.get<std::string>                ("Residual Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
 //   haveSource  (p.get<bool>("Have Source"))
 //   ,haveMechSource  (p.get<bool>("Have Mechanical Source"))
  {
    if (p.isType<bool>("Disable Transient"))
      enableTransient = !p.get<bool>("Disable Transient");
    else enableTransient = true;

    this->addDependentField(wBF);
    this->addDependentField(wGradBF);
    this->addDependentField(GradBF);
    this->addDependentField(Dstar);
    this->addDependentField(DL);
    this->addDependentField(Clattice);
    this->addDependentField(eqps);
    this->addDependentField(eqpsFactor);
    this->addDependentField(Ctrapped);
    this->addDependentField(Ntrapped);
    this->addDependentField(CLGrad);
    this->addDependentField(stressGrad);
    this->addDependentField(DefGrad);
    this->addDependentField(Pstress);
    this->addDependentField(weights);
    this->addDependentField(tauFactor);
    this->addDependentField(deltaTime);

 //   if (haveSource) this->addDependentField(Source);
 //   if (haveMechSource) this->addDependentField(MechSource);



    this->addEvaluatedField(TResidual);


    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vector_dl->dimensions(dims);
    numQPs  = dims[1];
    numDims = dims[2];

    Teuchos::RCP<PHX::DataLayout> node_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout");
    std::vector<PHX::DataLayout::size_type> ndims;
    node_dl->dimensions(ndims);
    worksetSize = dims[0];
    numNodes = dims[1];

    // Get data from previous converged time step
    ClatticeName = p.get<std::string>("QP Variable Name")+"_old";
    eqpsName = p.get<std::string>("eqps Name")+"_old";


    // Allocate workspace for temporary variables
    Hflux.resize(dims[0], numQPs, numDims);
    Hfluxdt.resize(dims[0], numQPs, numDims);
    pterm.resize(dims[0], numQPs);


    C.resize(worksetSize, numQPs, numDims, numDims);
    Cinv.resize(worksetSize, numQPs, numDims, numDims);
    CinvTgrad.resize(worksetSize, numQPs, numDims);
    CinvTaugrad.resize(worksetSize, numQPs, numDims);

    pTTterm.resize(dims[0], numQPs, numDims);
    pBterm.resize(dims[0], numNodes, numQPs, numDims);
    pTranTerm.resize(worksetSize, numDims);

    this->setName("HDiffusionDeformationMatterResidual"+PHX::TypeString<EvalT>::value);

  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void HDiffusionDeformationMatterResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
	this->utils.setFieldData(wBF,fm);
	this->utils.setFieldData(wGradBF,fm);
    this->utils.setFieldData(GradBF,fm);
	this->utils.setFieldData(Dstar,fm);
	this->utils.setFieldData(DL,fm);
	this->utils.setFieldData(Clattice,fm);
	this->utils.setFieldData(eqps,fm);
	this->utils.setFieldData(eqpsFactor,fm);
	this->utils.setFieldData(Ctrapped,fm);
	this->utils.setFieldData(Ntrapped,fm);
	this->utils.setFieldData(CLGrad,fm);
	this->utils.setFieldData(stressGrad,fm);
	this->utils.setFieldData(DefGrad,fm);
	this->utils.setFieldData(Pstress,fm);
	this->utils.setFieldData(tauFactor,fm);
	this->utils.setFieldData(weights,fm);
	this->utils.setFieldData(deltaTime,fm);

//    if (haveSource) this->utils.setFieldData(Source);
 //   if (haveMechSource) this->utils.setFieldData(MechSource);

    this->utils.setFieldData(TResidual,fm);
  }

//**********************************************************************
template<typename EvalT, typename Traits>
void HDiffusionDeformationMatterResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;


  Albany::MDArray Clattice_old = (*workset.stateArrayPtr)[ClatticeName];
  Albany::MDArray eqps_old = (*workset.stateArrayPtr)[eqpsName];

  ScalarT dt = deltaTime(0);


  // Set Warning message
//  if (Clattice_old(1,1) <= 0 || Clattice(1,1) <= 0 ) {
//	  cout << "negative or zero lattice concentration detected. Error! \n";
//  }

// To overcome the ill-conditioning, the balance law f(u,c_L)=0 is
// replaced by dt/dl*f(u, c_L) = 0


  // compute the 'material' flux
  FST::tensorMultiplyDataData<ScalarT> (C, DefGrad, DefGrad, 'T');
  Intrepid::RealSpaceTools<ScalarT>::inverse(Cinv, C);
  FST::tensorMultiplyDataData<ScalarT> (CinvTgrad, Cinv, CLGrad);






  // FST::scalarMultiplyDataData<ScalarT> (Hflux, DL, CLGrad);

  // For debug only
  // FST::integrate<ScalarT>(TResidual, CLGrad, wGradBF, Intrepid::COMP_CPP, false); // this one works
   FST::integrate<ScalarT>(TResidual, CinvTgrad, wGradBF, Intrepid::COMP_CPP, false); // this also works
  //FST::integrate<ScalarT>(TResidual, Hflux, wGradBF, Intrepid::COMP_CPP, false);

  ScalarT artificialViscosity(1e-6); // Currently it's set to be constant, but I will change it -S.Sun

   for (std::size_t cell=0; cell < workset.numCells; ++cell) {

 		  for (std::size_t node=0; node < numNodes; ++node) {

 			 TResidual(cell,node) = TResidual(cell,node)*dt;
 		  }
   }


  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

		  for (std::size_t node=0; node < numNodes; ++node) {
	//		  TResidual(cell,node)=0.0;
			  for (std::size_t qp=0; qp < numQPs; ++qp) {

				  // Transient Term
				  TResidual(cell,node) += Dstar(cell, qp)*(
						     Clattice(cell,qp)- Clattice_old(cell, qp)
						    )*wBF(cell, node, qp)
						    /(DL(cell,qp)+artificialViscosity);

				  // Strain Rate Term
				  TResidual(cell,node) += Ctrapped(cell, qp)/Ntrapped(cell, qp)*
						                  eqpsFactor(cell,qp)*(
				  						     eqps(cell,qp)- eqps_old(cell, qp)
				  						    ) *wBF(cell, node, qp)*dt
				  						  /(DL(cell,qp)+artificialViscosity);

			  }
		  }
  }




  // hydrostatic stress term
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
	  for (std::size_t qp=0; qp < numQPs; ++qp)
	  {
		  {
			  for (std::size_t node=0; node < numNodes; ++node)
			  {
				  for (std::size_t i=0; i<numDims; i++)
				  {
					  for (std::size_t j=0; j<numDims; j++)
					  {
						  TResidual(cell,node) -= tauFactor(cell,qp)*
	                		          wGradBF(cell, node, qp, i)*
	                		          Cinv(cell,qp,i,j)*
	                		          stressGrad(cell, qp, j)*dt
	                		          /(DL(cell,qp)+artificialViscosity);
					  }

				  }
			  }
		  }
	  }
  }



  //---------------------------------------------------------------------------//
  // Stabilization Term (only 2D and 3D problem need stabilizer)



  for (std::size_t cell=0; cell < workset.numCells; ++cell){


   vol = 0.0;
   for (std::size_t qp=0; qp < numQPs; ++qp) {
   	   vol  += weights(cell,qp);
      }

   for (std::size_t i=0; i<numDims; i++){
	   pTranTerm(cell,i) = 0;
	   for (std::size_t qp=0; qp < numQPs; ++qp) {
		   pTranTerm(cell, i) += weights(cell,qp)*(
				   CinvTgrad(cell,qp,i) );
	   }
	   pTranTerm(cell,i) /=  vol;
   }
  }

  ScalarT stabParameter(0e1);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

	  for (std::size_t node=0; node < numNodes; ++node) {
		  for (std::size_t qp=0; qp < numQPs; ++qp) {
			  for (std::size_t i=0; i<numDims; i++)
			  				  {
//			  TResidual(cell,node) +//= pTranTerm(cell,i)*stabParameter*dt
//                    		             *wGradBF(cell, node, qp,i);
//				  TResidual(cell,node) += -CinvTgrad(cell,qp,i)/(pTranTerm(cell,1)+ pTranTerm(cell,2) + pTranTerm(cell,3))*1e-3*
//						                   GradBF(cell, node, qp,i)*TResidual(cell,node)
//				                      		             *wBF(cell, node, qp);
			  				  }
		  }
	  }
  }










}

//**********************************************************************
}


