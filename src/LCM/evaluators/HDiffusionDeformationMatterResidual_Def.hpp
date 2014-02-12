//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>

#include <Intrepid_FunctionSpaceTools.hpp>
//#include <Intrepid_RealSpaceTools.hpp>
#include <Intrepid_MiniTensor.h>

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
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
     stab_param_(p.get<RealType>("Stabilization Parameter"))

  {
    if (p.isType<bool>("Disable Transient"))
      enableTransient = !p.get<bool>("Disable Transient");
    else enableTransient = true;

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
    Hflux.resize(worksetSize, numQPs, numDims);
    pterm.resize(worksetSize, numQPs);
    tpterm.resize(worksetSize, numNodes, numQPs);
    artificalDL.resize(worksetSize, numQPs);
    stabilizedDL.resize(worksetSize, numQPs);
    C.resize(worksetSize, numQPs, numDims, numDims);
    Cinv.resize(worksetSize, numQPs, numDims, numDims);
    CinvTgrad.resize(worksetSize, numQPs, numDims);
    CinvTgrad_old.resize(worksetSize, numQPs, numDims);
    CinvTaugrad.resize(worksetSize, numQPs, numDims);

    this->setName("HDiffusionDeformationMatterResidual"+PHX::TypeString<EvalT>::value);

  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void HDiffusionDeformationMatterResidual<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
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
//	 typedef Intrepid::RealSpaceTools<ScalarT> RST;

    Albany::MDArray Clattice_old = (*workset.stateArrayPtr)[ClatticeName];
    Albany::MDArray eqps_old = (*workset.stateArrayPtr)[eqpsName];
    Albany::MDArray CLGrad_old = (*workset.stateArrayPtr)[CLGradName];

    ScalarT dt = deltaTime(0);
    ScalarT temp(0.0);


    // compute artifical diffusivity
    // for 1D this is identical to lumped mass as shown in Prevost's paper.
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	if (dt == 0){
    	     		 artificalDL(cell,qp) = 0;
    	} else {
        temp = elementLength(cell,qp)*elementLength(cell,qp)/6.0*Dstar(cell,qp)/DL(cell,qp)/dt;
        artificalDL(cell,qp) = stab_param_*
          //  (temp) // temp - DL is closer to the limit ...if lumped mass is preferred..
          std::abs(temp) // should be 1 but use 0.5 for safety
          *(0.5 + 0.5*std::tanh( (temp-1)/DL(cell,qp)  ))
          *DL(cell,qp);
    	}
        stabilizedDL(cell,qp) = artificalDL(cell,qp)/( DL(cell,qp) + artificalDL(cell,qp) );
      }
    }

    // compute the 'material' flux
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {

      for (std::size_t qp=0; qp < numQPs; ++qp) {


  		  Intrepid::Tensor<ScalarT> F(numDims, &DefGrad(cell, qp, 0, 0));
  		  Intrepid::Tensor<ScalarT> C_tensor_ = Intrepid::t_dot(F,F);
  		  Intrepid::Tensor<ScalarT> C_inv_tensor_ = Intrepid::inverse(C_tensor_);

  	      Intrepid::Vector<ScalarT> C_grad_(numDims, &CLGrad(cell, qp, 0));
  	      Intrepid::Vector<ScalarT> C_grad_in_ref_ = Intrepid::dot(C_inv_tensor_, C_grad_ );

         for (std::size_t j=0; j<numDims; j++){
        	 Hflux(cell,qp,j) = (1.0 -stabilizedDL(cell,qp))*C_grad_in_ref_(j)*dt;
        }
      }
    }

    FST::integrate<ScalarT>(TResidual, Hflux, wGradBF, Intrepid::COMP_CPP, false); // this also works

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {

                   // Divide the equation by DL to avoid ill-conditioned tangent
                   temp =  1.0/ ( DL(cell,qp)  + artificalDL(cell,qp)  );

                  // Transient Term
                  TResidual(cell,node) +=  Dstar(cell, qp)*
                		                                     (Clattice(cell,qp)- Clattice_old(cell, qp) )*
                                                             wBF(cell, node, qp)*temp;

                 // Strain Rate Term
                 TResidual(cell,node) +=  eqpsFactor(cell,qp)*
                                                           (eqps(cell,qp)- eqps_old(cell, qp))*
                                                            wBF(cell, node, qp)*temp;

                 // hydrostatic stress term
                 for (std::size_t dim=0; dim < numDims; ++dim) {
                         TResidual(cell,node) -= tauFactor(cell,qp)*Clattice(cell,qp)*
                                                                  wGradBF(cell, node, qp, dim)*
                                                                  stressGrad(cell, qp, dim)*dt*temp;
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
        CLPbar += weights(cell,qp)*
                           (Clattice(cell,qp) - Clattice_old(cell, qp)  );
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
          temp =  1.0/ ( DL(cell,qp)  + artificalDL(cell,qp)  );
          TResidual(cell,node) -=  stab_param_*Dstar(cell, qp)*temp*
                                                   (-Clattice(cell,qp) + Clattice_old(cell, qp)+pterm(cell,qp))*
                                                    wBF(cell, node, qp);
        }
      }
    }


}
  //**********************************************************************
}


