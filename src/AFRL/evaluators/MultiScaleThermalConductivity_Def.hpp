//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <sstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include <zmq.h>

namespace AFRL {

template<typename EvalT, typename Traits>
MultiScaleThermalConductivity<EvalT, Traits>::
MultiScaleThermalConductivity(Teuchos::ParameterList& p) :
  thermalCond(p.get<std::string>("QP Variable Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout"))
{
  randField = CONSTANT;

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidThermalCondParameters();

  // Check the parameters contained in the input file. Do not check the defaults
  // set programmatically
  cond_list->validateParameters(*reflist, 0,
    Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  std::string ebName =
    p.get<std::string>("Element Block Name", "Missing");

  type = cond_list->get("Thermal Conductivity Type", "Constant");

  if (type == "Constant") {

    ScalarT value = cond_list->get("Value", 1.0);
    init_constant(value, p);

  }

#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion" || type == "Log Normal RF") {

    init_KL_RF(type, *cond_list, p);

  }
#endif

  else if (type == "Block Dependent")
  {
    // We have a multiple material problem and need to map element blocks to material data

    if(p.isType<Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB")){
       materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");
    }
    else {
       TEUCHOS_TEST_FOR_EXCEPTION(
         true, Teuchos::Exceptions::InvalidParameter,
         std::endl <<
         "Error! Must specify a material database if using block dependent " <<
         "thermal conductivity" << std::endl);
    }

    // Get the sublist for thermal conductivity for the element block in the mat DB (the material in the
    // elem block ebName.

    Teuchos::ParameterList& subList = materialDB->getElementBlockSublist(ebName, "Thermal Conductivity");

    std::string typ = subList.get("Thermal Conductivity Type", "Constant");

    if (typ == "Constant") {

       ScalarT value = subList.get("Value", 1.0);
       init_constant(value, p);

    }
    else if (typ == "Compute from RVE") {
      std::string mat = materialDB->getElementBlockParam<std::string>(ebName, "material");
      std::string remoteAddress = cond_list->get("Remote Address", "");
      std::string descriptionFile = subList.get("RVE Description File", "");
      int descriptionId = subList.get("RVE ID", -1);
      init_remote(mat, remoteAddress, descriptionFile, descriptionId, p);
    }
#ifdef ALBANY_STOKHOS
    else if (typ == "Truncated KL Expansion" || typ == "Log Normal RF") {

       init_KL_RF(typ, subList, p);

    }
#endif
  } // Block dependent

  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Invalid thermal conductivity type " << type);
  }

  this->addEvaluatedField(thermalCond);
  this->setName("Thermal Conductivity" );
}

// template<typename EvalT, typename Traits>
// MultiScaleThermalConductivity<EvalT, Traits>::
// ~MultiScaleThermalConductivity()
// {

// }

template<typename EvalT, typename Traits>
void
MultiScaleThermalConductivity<EvalT, Traits>::
init_constant(ScalarT value, Teuchos::ParameterList& p){

    computeMode = Constant;
    randField = CONSTANT;

    constant_value = value;

    // Add thermal conductivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

    this->registerSacadoParameter("Thermal Conductivity", paramLib);

} // init_constant

#ifdef ALBANY_STOKHOS
template<typename EvalT, typename Traits>
void
MultiScaleThermalConductivity<EvalT, Traits>::
init_KL_RF(std::string &type, Teuchos::ParameterList& sublist, Teuchos::ParameterList& p){

    computeMode = Series;

    if (type == "Truncated KL Expansion")
      randField = UNIFORM;
    else if (type == "Log Normal RF")
      randField = LOGNORMAL;

    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl =
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(sublist));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Thermal Conductivity KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = sublist.get(ss, 0.0);
    }

} // (type == "Truncated KL Expansion" || type == "Log Normal RF")
#endif

template<typename EvalT, typename Traits>
void
MultiScaleThermalConductivity<EvalT, Traits>::
init_remote(std::string &type, std::string& remoteAddress,
            std::string& descriptionFile, int id, Teuchos::ParameterList& p){

    computeMode = Remote;
    constant_value = 1.;

    RVE.material = type;
    RVE.descriptionfile = descriptionFile;
    RVE.id = id;
    RVE.context = zmq_ctx_new();
    RVE.socket = zmq_socket(RVE.context, ZMQ_REQ);
    int rc = zmq_connect(RVE.socket, remoteAddress.c_str());
    std::cout<<"connection to socket "<<remoteAddress<<": "<<rc<<std::endl;

    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      temp(p.get<std::string> ("Variable Name"), scalar_dl);
    temperature = temp;
    this->addDependentField(temperature);

    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim>
      gradTemp(p.get<std::string> ("Variable Gradient Name"), vector_dl);
    gradTemperature = gradTemp;
    this->addDependentField(gradTemperature);

    // Add thermal conductivity as a Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib =
      p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

    this->registerSacadoParameter("Thermal Conductivity", paramLib);

} // init_remote

namespace
{
  template <class ScalarT>
  double val(const ScalarT& v)
  {
    return Sacado::ScalarValue<Sacado::Fad::DFad<ScalarT> >::eval(v);
  }
}

template<typename EvalT,typename Traits>
double MultiScaleThermalConductivity<EvalT,Traits>::get_remote(
  double time, double previousTime, const ScalarT& temperature,
  const Teuchos::Array<ScalarT>& gradT) const
{
  // std::cout<<RVE.material<<":"<<std::endl;
  // std::cout<<"description file: "<<RVE.descriptionfile<<std::endl;
  // std::cout<<"id: "<<RVE.id<<std::endl;

  // std::cout<<"time: "<<time<<std::endl;
  // std::cout<<"previous time: "<<previousTime<<std::endl;
  // std::cout<<"temperature: "<<val(temperature)<<std::endl;
  // std::cout<<"gradient temp: "<<val(gradT[0])<<" "<<val(gradT[1])<<" "<<val(gradT[2])<<std::endl;

  // Query remote system for thermal conductivity
  std::stringstream s;
  s << RVE.material << "," << RVE.descriptionfile << ","
    << time << "," << previousTime << "," << val(temperature) << ","
    << val(gradT[0]) << "," << val(gradT[1]) << "," << val(gradT[2]);
  std::cout<<"sending <"<<s.str()<<">"<<" to "<<RVE.socket<<std::endl;
  zmq_send(RVE.socket, s.str().c_str(), s.str().size(), 0);

  s.clear(); s.str("");

  char buffer[100];
  zmq_recv(RVE.socket, buffer, 100, 0);
  s.str(buffer);

  double thermalConductivity;
  s >> thermalConductivity;

  return thermalConductivity;
}

// **********************************************************************
template<typename EvalT, typename Traits>
void MultiScaleThermalConductivity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(thermalCond,fm);
  // this->utils.setFieldData(time,fm);
  // this->utils.setFieldData(deltaTime,fm);
  if (computeMode == Series)
    this->utils.setFieldData(coordVec,fm);
  if (computeMode == Remote)
  {
    this->utils.setFieldData(temperature,fm);
    this->utils.setFieldData(gradTemperature,fm);
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void MultiScaleThermalConductivity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (computeMode == Constant) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
         thermalCond(cell,qp) = constant_value;
      }
    }
  }
#ifdef ALBANY_STOKHOS
  else if (computeMode == Series) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
          Teuchos::Array<MeshScalarT> point(numDims);
          for (std::size_t i=0; i<numDims; i++)
              point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
          if (randField == UNIFORM)
              thermalCond(cell,qp) = exp_rf_kl->evaluate(point, rv);
          else if (randField == LOGNORMAL)
              thermalCond(cell,qp) = std::exp(exp_rf_kl->evaluate(point, rv));
      }
    }
  }
#endif
  else if (computeMode == Remote) {

    ScalarT meanTemp = 0.;
    Teuchos::Array<ScalarT> meanGradTemp(numDims);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        meanTemp += Sacado::ScalarValue<ScalarT>::eval(temperature(cell,qp));
        for (std::size_t i=0; i<numDims; i++) {
          meanGradTemp[i] += Sacado::ScalarValue<ScalarT>::eval(gradTemperature(cell,qp,i));
        }
      }
    }

    meanTemp /= (workset.numCells*numQPs);
    for (std::size_t i=0; i<numDims; i++) {
      meanGradTemp[i] /= (workset.numCells*numQPs);
    }


    double thermalConductivity = get_remote(workset.current_time,
                                            workset.previous_time,
                                            meanTemp,
                                            meanGradTemp);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        thermalCond(cell,qp) = thermalConductivity;
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename MultiScaleThermalConductivity<EvalT,Traits>::ScalarT&
MultiScaleThermalConductivity<EvalT,Traits>::getValue(const std::string &n)
{
  if (computeMode == Constant) {
    return constant_value;
  }
#ifdef ALBANY_STOKHOS
  else if (computeMode == Series) {
    for (int i=0; i<rv.size(); i++) {
      if (n == Albany::strint("Thermal Conductivity KL Random Variable",i))
        return rv[i];
    }
  }
#endif
  else if (computeMode == Remote) {
    return constant_value;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error! Logic error in getting paramter " << n
                     << " in MultiScaleThermalConductivity::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
MultiScaleThermalConductivity<EvalT,Traits>::getValidThermalCondParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
       rcp(new Teuchos::ParameterList("Valid Thermal Conductivity Params"));;

  validPL->set<std::string>("Thermal Conductivity Type", "Constant",
               "Constant thermal conductivity across the entire domain");
  validPL->set<std::string>("Remote Address", "",
               "Address to send/recieve microscale simulation data");
  validPL->set<double>("Value", 1.0, "Constant thermal conductivity value");

// Truncated KL parameters

  validPL->set<int>("Number of KL Terms", 2, "");
  validPL->set<double>("Mean", 0.2, "");
  validPL->set<double>("Standard Deviation", 0.1, "");
  validPL->set<std::string>("Domain Lower Bounds", "{0.0 0.0}", "");
  validPL->set<std::string>("Domain Upper Bounds", "{1.0 1.0}", "");
  validPL->set<std::string>("Correlation Lengths", "{1.0 1.0}", "");

// Remote parameters

  validPL->set<std::string>("Microscale Executable", "",
               "Executable for computing thermal conductivity from the microscale");

  return validPL;
}

// **********************************************************************
// **********************************************************************
}
