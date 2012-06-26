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
    wBF         (p.get<std::string>           ("Weighted BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
	wGradBF     (p.get<std::string>           ("Weighted Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
	GradBF      (p.get<std::string>           ("Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
    Dstar (p.get<std::string>                 ("Effective Diffusivity Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    DL   (p.get<std::string>                  ("Diffusion Coefficient Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    Clattice (p.get<std::string>              ("QP Variable Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	eqps (p.get<std::string>                  ("eqps Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    eqpsFactor (p.get<std::string>            ("Strain Rate Factor Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	Ctrapped (p.get<std::string>              ("Trapped Concentration Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	Ntrapped (p.get<std::string>              ("Trapped Solvent Name"),
         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    CLGrad       (p.get<std::string>          ("Gradient QP Variable Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
	stressGrad       (p.get<std::string>      ("Gradient Hydrostatic Stress Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
//    Source      (p.get<std::string>                ("Source Name"),
//		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
//	MechSource      (p.get<std::string>            ("Mechanical Source Name"),
//		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
   stabParameter  (p.get<std::string>         ("Material Property Name"),
		p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    DefGrad      (p.get<std::string>          ("Deformation Gradient Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
	Pstress      (p.get<std::string>          ("Stress Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
	weights       (p.get<std::string>         ("Weights Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	tauFactor  (p.get<std::string>            ("Tau Contribution Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
    elementLength (p.get<std::string>         ("Element Length Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
	deltaTime (p.get<std::string>             ("Delta Time Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Workset Scalar Data Layout")),
    TResidual   (p.get<std::string>           ("Residual Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
 //   haveSource  (p.get<bool>("Have Source"))
 //   ,haveMechSource  (p.get<bool>("Have Mechanical Source"))
  {
    if (p.isType<bool>("Disable Transient"))
      enableTransient = !p.get<bool>("Disable Transient");
    else enableTransient = true;

    this->addDependentField(stabParameter);
    this->addDependentField(elementLength);
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
    CLGradName = p.get<std::string>("Gradient QP Variable Name")+"_old";
    eqpsName = p.get<std::string>("eqps Name")+"_old";


    // Allocate workspace for temporary variables
    Hflux.resize(dims[0], numQPs, numDims);
    Hfluxdt.resize(dims[0], numQPs, numDims);
    pterm.resize(dims[0], numQPs);
    tpterm.resize(dims[0], numNodes, numQPs);

    artificalDL.resize(dims[0], numQPs);
    stabilizedDL.resize(dims[0], numQPs);

    C.resize(worksetSize, numQPs, numDims, numDims);
    Cinv.resize(worksetSize, numQPs, numDims, numDims);
    CinvTgrad.resize(worksetSize, numQPs, numDims);
    CinvTgrad_old.resize(worksetSize, numQPs, numDims);
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
	this->utils.setFieldData(stabParameter,fm);
	this->utils.setFieldData(elementLength,fm);
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
  Albany::MDArray CLGrad_old = (*workset.stateArrayPtr)[CLGradName];



  ScalarT dt = deltaTime(0);
  ScalarT temp(1.0);

  ScalarT fac;
  if (dt==0) {
	  fac = 1.0e20;
  }
  else
  {
	  fac = 1.0/dt;
  }


  // compute artifical diffusivity

  // for 1D this is identical to lumped mass as shown in Prevost's paper.

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

	  for (std::size_t qp=0; qp < numQPs; ++qp) {

		      temp = elementLength(cell,qp)*elementLength(cell,qp)/6.0*Dstar(cell,qp)/DL(cell,qp)*fac;

//		      temp = elementLength(cell,qp)/6.0*Dstar(cell,qp)/DL(cell,qp)*fac - 1/elementLength(cell,qp);
//		      if (  temp > 1.0 )
		 //     {
			    artificalDL(cell,qp) = stabParameter(cell,qp)*
			//    	   (temp) // temp - DL is closer to the limit ...if lumped mass is preferred..
				      std::abs(temp) // should be 1 but use 0.5 for safety
				      *(0.5 + 0.5*std::tanh( (temp-1)/DL(cell,qp)  ))
				      // smoothened Heavside function
	  			      *DL(cell,qp) //*stabParameter(cell,qp)
				      ;
		 //     }
/*		      else
		      {
		    	  artificalDL(cell,qp) =
		    			  (temp) // 1.25 = safety factor
		    	  	     *DL(cell,qp) //*stabParameter(cell,qp)
		    	  	     ;
		      }
*/
//		      cout << temp << endl;
		  }

  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

	   for (std::size_t qp=0; qp < numQPs; ++qp) {
		   stabilizedDL(cell,qp) = artificalDL(cell,qp)/( DL(cell,qp) + artificalDL(cell,qp) );
      }
 }


  // compute the 'material' flux
  FST::tensorMultiplyDataData<ScalarT> (C, DefGrad, DefGrad, 'T');
  Intrepid::RealSpaceTools<ScalarT>::inverse(Cinv, C);
  FST::tensorMultiplyDataData<ScalarT> (CinvTgrad_old, Cinv, CLGrad_old);
  FST::tensorMultiplyDataData<ScalarT> (CinvTgrad, Cinv, CLGrad);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

	   for (std::size_t qp=0; qp < numQPs; ++qp) {
		   for (std::size_t j=0; j<numDims; j++){
			//  CinvTgrad_old(cell,qp,j) = 0.0;
			//   for (std::size_t node=0; node < numNodes; ++node) {
			     Hflux(cell,qp,j) = CinvTgrad(cell,qp,j)
			       	    -stabilizedDL(cell,qp)
			    		 *CinvTgrad_old(cell,qp,j)
		   		//			  }
			    		 ;
		   }

      }
 }



  // FST::scalarMultiplyDataData<ScalarT> (Hflux, DL, CLGrad);

  // For debug only
  // FST::integrate<ScalarT>(TResidual, CLGrad, wGradBF, Intrepid::COMP_CPP, false); // this one works
   FST::integrate<ScalarT>(TResidual, Hflux, wGradBF, Intrepid::COMP_CPP, false); // this also works
  //FST::integrate<ScalarT>(TResidual, Hflux, wGradBF, Intrepid::COMP_CPP, false);

  // multiplied the equation by dt.

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
				  TResidual(cell,node) +=
						  Dstar(cell, qp)/ ( DL(cell,qp)  + artificalDL(cell,qp)  )*(
				 				  		     Clattice(cell,qp)- Clattice_old(cell, qp)
				 				  		     )*
				 			      	    wBF(cell, node, qp);




				  // Transient Term
				  //TResidual(cell,node) += Dstar(cell, qp)*(
				  //		     Clattice(cell,qp)- Clattice_old(cell, qp)
			      //	    )*wBF(cell, node, qp)
			      //	    /DL(cell,qp);

				  // Strain Rate Term
				  TResidual(cell,node) += Ctrapped(cell, qp)/Ntrapped(cell, qp)*
						                  eqpsFactor(cell,qp)*(
				  						     eqps(cell,qp)- eqps_old(cell, qp)
				  						    ) *wBF(cell, node, qp)
				  						  /(DL(cell,qp) + artificalDL(cell,qp) ) ;

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
	                		          /( DL(cell,qp) + artificalDL(cell,qp) );
					  }

				  }
			  }
		  }
	  }
  }



  //---------------------------------------------------------------------------//
  // Stabilization Term


  ScalarT CLPbar(0);
  ScalarT vol(0);

  for (std::size_t cell=0; cell < workset.numCells; ++cell){



   CLPbar = 0.0;
   vol = 0.0;

   for (std::size_t qp=0; qp < numQPs; ++qp) {
	CLPbar += weights(cell,qp)*(
		  		     Clattice(cell,qp) - Clattice_old(cell, qp)
			                      );
	vol  += weights(cell,qp);
   }
   CLPbar /= vol;

   for (std::size_t qp=0; qp < numQPs; ++qp) {
   pterm(cell,qp) = CLPbar;
        }

   for (std::size_t node=0; node < numNodes; ++node) {
  	     trialPbar = 0.0;
   		 for (std::size_t qp=0; qp < numQPs; ++qp) {
   			  trialPbar += wBF(cell,node,qp);
   		 }
   		 trialPbar /= vol;
   		 for (std::size_t qp=0; qp < numQPs; ++qp) {
   		 		   tpterm(cell,node,qp) = trialPbar;
  		 }

  }

 }



  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

 	  for (std::size_t node=0; node < numNodes; ++node) {
 		  for (std::size_t qp=0; qp < numQPs; ++qp) {
  				  TResidual(cell,node) -=
  						                  // (2.0-12*dt*DL(cell,qp)
  				                          // /elementLength(cell,qp)/elementLength(cell,qp))
  						                  stabParameter(cell,qp)
  				                          *Dstar(cell, qp)/ ( DL(cell,qp)  + artificalDL(cell,qp)  )*
  						(
  						  - Clattice(cell,qp) + Clattice_old(cell, qp)
  						  +pterm(cell,qp)
  						  )
  			  		     *(wBF(cell, node, qp));

 		  }
 	  }
   }








}

//**********************************************************************
}


