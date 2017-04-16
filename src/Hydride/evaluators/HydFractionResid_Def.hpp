//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
HydFractionResid<EvalT, Traits>::
HydFractionResid(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  dl_(dl),

  wBF         (p.get<std::string>                   ("Weighted BF Name"), dl->node_qp_scalar),
  Temperature (p.get<std::string>                   ("Temperature Name"), dl->qp_scalar),
  Tdot        (p.get<std::string>                   ("Temp Time Derivative Name"), dl->qp_scalar),
  Fh          (p.get<std::string>                   ("QP Variable Name"), dl->qp_scalar),
  Fhdot       (p.get<std::string>                   ("QP Time Derivative Variable Name"), dl->qp_scalar),
  JThermCond  (p.get<std::string>                   ("J Conductivity Name"), dl->qp_scalar),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"), dl->node_qp_vector),
  TGrad       (p.get<std::string>                   ("Temp Gradient Variable Name"), dl->qp_vector),
  FhResidual  (p.get<std::string>                   ("Residual Name"), dl->node_scalar)
{

  Teuchos::ParameterList* hyd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
               this->getValidHydFractionParameters();

  hyd_list->validateParameters(*reflist, 0, 
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  std::string ebName = 
    p.get<std::string>("Element Block Name", "Missing");

  type = hyd_list->get("Material Parameters Type", "Block Dependent");

  if (type == "Block Dependent") 
  {
    // We have a multiple material problem and need to map element blocks to material data

    if(p.isType<Teuchos::RCP<Albany::MaterialDatabase> >("MaterialDB")){
       materialDB = p.get< Teuchos::RCP<Albany::MaterialDatabase> >("MaterialDB");
    }
    else {
       TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Must specify a material database if using block dependent " << 
		     "material properties" << std::endl);
    }

    // Get the sublist for thermal conductivity for the element block in the mat DB (the material in the
    // elem block ebName.

    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "C_HHyd");

    std::string typ = subList.get("C_HHyd Type", "Constant");

    if (typ == "Constant") {

       C_HHyd = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "R");

    std::string typ = subList.get("R Type", "Constant");

    if (typ == "Constant") {

       R = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "CTSo");

    std::string typ = subList.get("CTSo Type", "Constant");

    if (typ == "Constant") {

       CTSo = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "delQ");

    std::string typ = subList.get("delQ Type", "Constant");

    if (typ == "Constant") {

       delQ = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "delWm");

    std::string typ = subList.get("delWm Type", "Constant");

    if (typ == "Constant") {

       delWm = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "stoi");

    std::string typ = subList.get("stoi Type", "Constant");

    if (typ == "Constant") {

       stoi = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "Vh");

    std::string typ = subList.get("Vh Type", "Constant");

    if (typ == "Constant") {

       Vh = subList.get("Value", 0.0);

    }
    }
    {

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "delG");

    std::string typ = subList.get("delG Type", "Constant");

    if (typ == "Constant") {

       delG = subList.get("Value", 0.0);

    }
    }
  } // Block dependent

  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Must specify material parameters in the material database" << type);
  } 


  this->addDependentField(wBF);
  this->addDependentField(Temperature);
  this->addDependentField(Tdot);
  this->addDependentField(Fhdot);
  this->addDependentField(Fh);
  this->addDependentField(JThermCond);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  this->addEvaluatedField(FhResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_->node_qp_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("HydFractionResid"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydFractionResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(Temperature,fm);
  this->utils.setFieldData(Tdot,fm);
  this->utils.setFieldData(Fhdot,fm);
  this->utils.setFieldData(Fh,fm);
  this->utils.setFieldData(JThermCond,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(TGrad,fm);
  
  // Allocate workspace
  JGrad = Kokkos::createDynRankView(TGrad.get_view(), "JGrad", worksetSize, numQPs, numDims);
  fh_coef = Kokkos::createDynRankView(Temperature.get_view(), "fh_coef", worksetSize, numQPs, numDims);
  fh_time_term = Kokkos::createDynRankView(Fhdot.get_view(), "fh_time_term", worksetSize, numQPs, numDims);
  CHZr_coef = Kokkos::createDynRankView(Temperature.get_view(), "CHZr_coef", worksetSize, numQPs, numDims);
  CH_time_term = Kokkos::createDynRankView(Temperature.get_view(), "CH_time_term", worksetSize, numQPs, numDims);

  this->utils.setFieldData(FhResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydFractionResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  // First, multiply the coefficient (JThermConductivity) by the temperature gradient to get the JGrad field

  FST::scalarMultiplyDataData<ScalarT> (JGrad, JThermCond.get_view(), TGrad.get_view());

  /* Now, integrate this JGrad term into the residual statement, which gives us the RHS term:

   JGrad = 0

  */

  FST::integrate(FhResidual.get_view(), JGrad, wGradBF.get_view(), false); // "false" overwrites

  /*
     Now, build the coefficient for \partial f_H / \partial t
  */

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
         fh_coef(cell, qp) = C_HHyd - CTSo * std::exp(- delQ / (R * Temperature(cell, qp))) *
              std::exp(delWm / (stoi * R * Temperature(cell, qp))) * 
              std::exp(Vh * delG / (R * Temperature(cell, qp)));
      }
    }

    // multiply by Fhdot

    FST::scalarMultiplyDataData<ScalarT> (fh_time_term, fh_coef, Fhdot.get_view());

    // integrate and sum into residual

    FST::integrate(FhResidual.get_view(), fh_time_term, wBF.get_view(), true); // "true" sums into

  /*
     Finally, build the coefficient for \partial C_H,Zr / \partial t
  */

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
         CHZr_coef(cell, qp) = -(delWm + stoi * Vh * delG) / (stoi * R * Temperature(cell, qp) * Temperature(cell, qp));
         CHZr_coef(cell, qp) *= CTSo * std::exp(- delQ / (R * Temperature(cell, qp))) *
              std::exp(delWm / (stoi * R * Temperature(cell, qp))) * 
              std::exp(Vh * delG / (R * Temperature(cell, qp)));
         CHZr_coef(cell, qp) *= (1.0 - Fh(cell, qp));
      }
    }

    // multiply by Tdot

    FST::scalarMultiplyDataData<ScalarT> (CH_time_term, CHZr_coef, Tdot.get_view());

    // integrate and sum into residual

    FST::integrate(FhResidual.get_view(), CH_time_term, wBF.get_view(), true); // "true" sums into

}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
HydFractionResid<EvalT, Traits>::
getValidHydFractionParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Hyd Fraction Params"));;

  validPL->set<std::string>("C_HHyd Type", "Constant", 
               "Constant C_HHyd across the entire domain");
  validPL->set<std::string>("R Type", "Constant", 
               "Constant R across the entire domain");
  validPL->set<std::string>("CTSo Type", "Constant", 
               "Constant CTSo across the entire domain");
  validPL->set<std::string>("delQ Type", "Constant", 
               "Constant delQ across the entire domain");
  validPL->set<std::string>("delWm Type", "Constant", 
               "Constant delWm across the entire domain");
  validPL->set<std::string>("stoi Type", "Constant", 
               "Constant stoi across the entire domain");
  validPL->set<std::string>("Vh Type", "Constant", 
               "Constant Vh across the entire domain");
  validPL->set<std::string>("delG Type", "Constant", 
               "Constant delG across the entire domain");
  validPL->set<double>("Value", 1.0, "Constant material parameter value");

  return validPL;
}


//**********************************************************************
}

