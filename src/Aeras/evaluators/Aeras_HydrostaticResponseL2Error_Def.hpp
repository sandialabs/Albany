//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AERAS_HYDROSTATIC_RESPONSE_L2ERROR_DEF_HPP_
#define AERAS_HYDROSTATIC_RESPONSE_L2ERROR_DEF_HPP_

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx.hpp"
#include "PHAL_Utilities.hpp"
#include "Aeras_ShallowWaterConstants.hpp"
#include "Aeras_Eta.hpp" 

namespace Aeras {
template<typename EvalT, typename Traits>
HydrostaticResponseL2Error<EvalT, Traits>::
HydrostaticResponseL2Error(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  sphere_coord("Lat-Long", dl->qp_gradient),
  weighted_measure("Weights", dl->qp_scalar),
  velocity("Velx",  dl->qp_vector_level),
  temperature("Temperature",dl->qp_scalar_level),
  spressure("SPressure",dl->qp_scalar),
  numLevels(dl->node_scalar_level->dimension(2)), 
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  *out << "in Hydrostatic_Response_L2Error! \n";

  // number of quad points per cell and dimension of space
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;

  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  *out << "numQPs, numDims, numLevels: " << numQPs << ", " << numDims << ", " << numLevels << std::endl;

  //User-specified parameters
  refSolName = plist->get<std::string>("Reference Solution Name"); //no reference solution by default.
  *out << "Reference Solution Name for Aeras::HydrostaticResponseL2Error response: " << refSolName << std::endl;
  inputData = plist->get<double>("Reference Solution Data", 0.0);

  if (refSolName == "Zero")
    ref_sol_name = ZERO;
  else if (refSolName == "Baroclinic Instabilities Unperturbed")
    ref_sol_name  = BAROCLINIC_UNPERTURBED;
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown reference solution name " << ref_sol_name <<
      "!" << std::endl;);
  }

 
  this->addDependentField(sphere_coord);
  this->addDependentField(weighted_measure);
  this->addDependentField(velocity);
  this->addDependentField(temperature);
  this->addDependentField(spressure);

  this->setName("Aeras Hydrostatic Response L2 Error");

  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = "Local Response Aeras Hydrostatic Response L2 Error";
  std::string global_response_name = "Global Response Aeras Hydrostatic Response L2 Error";
  int worksetSize = scalar_dl->dimension(0);

  //FIXME: extend responseSize to have tracers 
  responseSize = 3*numLevels + 1; //there are 2 velocities and 1 temperature variable on each level
                                      //surface pressure is on 1st level only
                                      //the ordering is: Sp0, u0, v0, T0, u1, v1, T1, etc

  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(
      new MDALayout<Cell,Dim>(worksetSize, responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name,
                                       local_response_layout);
  p.set("Local Response Field Tag", local_response_tag);

  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(
      new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> global_response_tag(global_response_name,
                                        global_response_layout);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);

}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Error<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphere_coord,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(velocity,fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(spressure,fm);

  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Error<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  const int imax = this->global_response.size();

  PHAL::set(this->global_response, 0.0);

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Error<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  *out << "HydrostaticResponseL2Error evaluateFields() \n" << std::endl;

  //Zero out local response 
  PHAL::set(this->local_response, 0.0);

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> spressure_ref(workset.numCells, numQPs); //spressure_ref (exact solution) at quad points
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> spressure_err(workset.numCells, numQPs); //spressure error at quad points
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> velocity_ref(workset.numCells, numQPs, numLevels, 2); //velocity_ref (exact solution) at quad points
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> velocity_err(workset.numCells, numQPs, numLevels, 2); //velocity error at quad points
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> temperature_ref(workset.numCells, numQPs, numLevels); //temperature_ref (exact solution) at quad points
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> temperature_err(workset.numCells, numQPs, numLevels); //temperature error at quad points

  //Get time from workset.  This is for setting time-dependent exact solution.
  const RealType time  = workset.current_time;
  *out << "time = " << time << std::endl;

  //Set reference solution at quadrature points
  if (ref_sol_name == ZERO) { //zero reference solution 
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        spressure_ref(cell,qp) = 0.0;
        for (std::size_t level=0; level < numLevels; ++level) {
          for (std::size_t i=0; i<2; ++i) {
            velocity_ref(cell,qp,level,i) = 0.0; 
          }
          temperature_ref(cell,qp,level) = 0.0; 
        }
      }
    }
  }
  else if (ref_sol_name == BAROCLINIC_UNPERTURBED) {
    *out << "Setting baroclinic unperturbed reference solution!" << std::endl; 
    //IKT, 5/23/16: the values/expressions here are from AAdapt_AnalyticFunction.cpp.
    //Warning: if the values/expressions change in AAdapt_AnalyticFunctions.cpp, they need to 
    //be changed here as well. 
    const double u0 = 35.0;
    const double SP0 = 1.0e5;
    const double P0 = SP0; 
    const double Eta0 = 0.252, Etas=1.0, Etat=0.2, TT0=288.0,
                 Gamma = 0.005, deltaT = 4.8E+5, Rd = 287.04;
    const double Ptop = 219.4067;
    const double a = Aeras::ShallowWaterConstants::self().earthRadius;
    const double omega = Aeras::ShallowWaterConstants::self().omega;
    const double g = Aeras::ShallowWaterConstants::self().gravity;
    double a_omega      = a*omega;
    const double constPi = Aeras::ShallowWaterConstants::self().pi; 
    const Aeras::Eta<DoubleType> &EP = Aeras::Eta<DoubleType>::self(Ptop,P0,numLevels);
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //FIXME: SP_ref = SP0 is IC but it does not stay constant in time.  What should it be in the error computation?
        spressure_ref(cell,qp) = SP0;
        //The following 2 definitions are for the 1st velocity component solution
        const MeshScalarT theta = sphere_coord(cell, qp, 1);
        const double sin2Theta = std::sin(2.0*theta);
        for (std::size_t level=0; level < numLevels; ++level) {
          //first component of velocity.
          const double Eta =  EP.eta(level);
          const double cosEtav = std::cos((Eta-Eta0)*constPi/2.0);
          velocity_ref(cell,qp,level,0) = u0 * std::pow(cosEtav,1.5) * std::pow(sin2Theta,2.0);
          //second component of velocity.
          //FIXME: v = 0 is IC but it does not stay constant in time.  What should it be in the error computation?
          velocity_ref(cell,qp,level,1) = 0.0;
          double Tavg =  TT0 * std::pow(Eta, Rd*Gamma/g);
          if( Eta <= Etat ) Tavg += deltaT * std::pow(Etat - Eta, 5.0);
          double factor       = Eta*constPi*u0/Rd;
          double phi_vertical = (Eta - Eta0) * 0.5 *constPi;
          double t_deviation = factor*1.5* std::sin(phi_vertical) * std::pow(std::cos(phi_vertical),0.5) *
                 ((-2.* std::pow(std::sin(theta),6.) * ( std::pow(std::cos(theta),2.) + 1./3.) + 10./63.)*
                 u0 * std::pow(std::cos(phi_vertical),1.5)  +
                 (8./5.*std::pow(std::cos(theta),3.) * (std::pow(std::sin(theta),2.) + 2./3.) - constPi/4.)*a_omega*0.5 );
          temperature_ref(cell,qp,level) = Tavg + t_deviation; //Tavg + TT0 * (TT1 + TT2);
        }
      }
    }
  }

  //Calculate L2 error at all the quad points.  We do not need to interpolate flow_state_field from nodes
  //to QPs because nodes = QPs for Aeras spectral elements.
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      spressure_err(cell,qp) = spressure(cell,qp) - spressure_ref(cell,qp); 
      for (std::size_t level=0; level < numLevels; ++level) {
        temperature_err(cell,qp,level) = temperature(cell,qp,level) - temperature_ref(cell,qp,level);
        /*if (cell == 0) 
           std::cout << "qp, level, temp, temp_ref, temp_err: " << qp << ", " << level << ", " << temperature(cell,qp,level) 
                     << ", " << temperature_ref(cell,qp,level) << ", " << temperature_err(cell,qp,level) << std::endl;  
        */
        for (std::size_t i=0; i<2; ++i) {
          velocity_err(cell,qp,level,i) = velocity(cell,qp,level,i) - velocity_ref(cell,qp,level,i);
        }
      }
    }
  }
  
  //Calculate absolute L2 error squared (for now) 
  //FIXME: calculate norm of reference solution squared for relative error calculation
  ScalarT wm; //weighted measure
  ScalarT spressure_err_sq = 0.0; 
  ScalarT temperature_err_sq = 0.0; 
  ScalarT uvelocity_err_sq = 0.0; 
  ScalarT vvelocity_err_sq = 0.0; 
  std::size_t dim; 
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp)  {
      wm = weighted_measure(cell,qp);
      //surface pressure field: dof 0
      dim = 0;
      spressure_err_sq = spressure_err(cell,qp)*spressure_err(cell,qp); 
      this->local_response(cell,dim) += wm*spressure_err_sq;
      this->global_response(dim) += wm*spressure_err_sq; 
      for (std::size_t level=0; level < numLevels; ++level) {
        //u-velocity field: dof 1, 4, 7, ...
        dim = 1 + level*3; 
        uvelocity_err_sq = velocity_err(cell,qp,level,0)*velocity_err(cell,qp,level,0); 
        this->local_response(cell,dim) += wm*uvelocity_err_sq; //velocity(cell,qp,level,0)*velocity(cell,qp,level,0);  
        this->global_response(dim) += wm*uvelocity_err_sq; //velocity(cell,qp,level,0)*velocity(cell,qp,level,0); 
        //v-velocity field: dof 2, 5, 8, ...
        dim = 2 + level*3; 
        vvelocity_err_sq = velocity_err(cell,qp,level,1)*velocity_err(cell,qp,level,1); 
        this->local_response(cell,dim) += wm*vvelocity_err_sq;  
        this->global_response(dim) += wm*vvelocity_err_sq;  
        //temperature field: dof 3, 6, 9, .... 
        dim = 3 + level*3; 
        temperature_err_sq = temperature_err(cell,qp,level)*temperature_err(cell,qp,level); 
        this->local_response(cell,dim) += wm*temperature_err_sq; 
        this->global_response(dim) += wm*temperature_err_sq;
        //FIXME: ultimately, will want to add tracers. 
      } 
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void HydrostaticResponseL2Error<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  *out << "HydrostaticResponseL2Error postEvaluate() \n" << std::endl;
#if 0
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));

  Teuchos::reduceAll(
    *workset.comm, *serializer, Teuchos::REDUCE_SUM,
    this->global_response.size(), &partial_response[0],
    &this->global_response[0]);
#else
  //amb reduceAll workaround.
  PHAL::reduceAll(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);
#endif

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);

#if 0
#else
  PHAL::MDFieldIterator<ScalarT> gr(this->global_response);
  for (int i=0; i < responseSize; ++i) {
    ScalarT abs_err_sq = *gr; 
    *gr = sqrt(abs_err_sq);  
    ++gr; 
  } 
#endif
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
HydrostaticResponseL2Error<EvalT,Traits>::getValidResponseParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL =
        rcp(new Teuchos::ParameterList("Valid Hydrostatic Response L2 Error Params"));;
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Name", "", "Scalar field from which to compute center of mass");
  validPL->set<std::string>("Description", "", "Description of this response used by post processors");
  validPL->set<std::string>("Reference Solution Name", "", "Name of reference solution");
  validPL->set<double>("Reference Solution Data", 0.0, "Data needed to specifying reference solution");

  return validPL;
}

}
#endif /* AERAS_HYDROSTATIC_RESPONSE_L2ERROR_DEF_HPP_ */
