//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "PHAL_Utilities.hpp"
#include <Intrepid2_MiniTensor.h>
#include "Albany_Utils.hpp"

template<typename EvalT, typename Traits>
ATO::TensorPNormResponse<EvalT, Traits>::
TensorPNormResponse(Teuchos::ParameterList& p,
		    const Teuchos::RCP<Albany::Layouts>& dl,
		    const Albany::MeshSpecsStruct* meshSpecs) :
  qp_weights ("Weights", dl->qp_scalar),
  BF         ("BF",      dl->node_qp_scalar)
{
  using Teuchos::RCP;


  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string tLayout = responseParams->get<std::string>("Tensor Field Layout");

  Teuchos::RCP<PHX::DataLayout> layout;
  if(tLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(tLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Tensor Field Layout " << tLayout <<
                               "!" << std::endl << "Options are (QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _tensor(responseParams->get<std::string>("Tensor Field Name"), layout);
  tensor = _tensor;

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  pVal = responseParams->get<double>("Exponent");

  Teuchos::RCP<TopologyArray>
    topologies = paramsFromProblem->get<Teuchos::RCP<TopologyArray> >("Topologies");

  TEUCHOS_TEST_FOR_EXCEPTION(
    topologies->size() != 1,
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorPNormResponse not implemented for multiple topologies." << std::endl);

  topology = (*topologies)[0];

  if(responseParams->isType<int>("Penalty Function")){
    functionIndex = responseParams->get<int>("Penalty Function");
  } else functionIndex = 0;

  if(responseParams->isType<int>("Barlat Exponent")){
    R = responseParams->get<int>("Barlat Exponent");
  } else R = 2;

  if(responseParams->isType<double>("Yield Stress")){
    yieldStress = responseParams->get<double>("Yield Stress");
  } else yieldStress = 1.0;

  TEUCHOS_TEST_FOR_EXCEPTION(
    (yieldStress<=0.0), Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorPNormResponse requires a positive yield stress." << std::endl);

  int matDim = 0;
  if(responseParams->isType<int>("Anisotropic Dimensions"))
      matDim = responseParams->get<int>("Anisotropic Dimensions");

  TEUCHOS_TEST_FOR_EXCEPTION(
    (matDim>3||matDim<0), Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorPNormResponse only supports 2D or 3D anisotropy." << std::endl);

  //Grab linear transforms from input data. Supports 2D/3D transforms.
  Cp.set_dimensions(6,6);
  Cp.fill(0.0);
  Cpp.set_dimensions(6,6);
  Cpp.fill(0.0);
  if (matDim==3) {
    bool formatTest = ( responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 0")
                && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 1")
                && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 2")
                && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 3")
                && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 4")
                && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 5")

                && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 0")
                && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 1")
                && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 2")
                && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 3")
                && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 4")
                && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 5"));

    Teuchos::Array<double> Cpp_in;
    Teuchos::Array<double> Cp_in;
    for (int i=0; i<6 && formatTest; i++) {
      Cp_in = responseParams->get<Teuchos::Array<double> >(Albany::strint("Transform 1 Row",i,' '));
      Cpp_in = responseParams->get<Teuchos::Array<double> >(Albany::strint("Transform 2 Row",i,' '));
      formatTest=(formatTest && Cp_in.size()==6 && Cpp_in.size()==6);
      for (int j=0; j<6 && formatTest; j++) {
        Cp(i,j)=Cp_in[j];
        Cpp(i,j)=Cpp_in[j];
      }
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
      (!formatTest), Teuchos::Exceptions::InvalidParameter, std::endl
      << "Error!  Anisotropic transforms do not match dimensions given." << std::endl);

  } else if (matDim==2) {
    bool formatTest = ( responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 0")
                     && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 1")
                     && responseParams->isType<Teuchos::Array<double> >("Transform 1 Row 2")

                     && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 0")
                     && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 1")
                     && responseParams->isType<Teuchos::Array<double> >("Transform 2 Row 2"));
    if (formatTest) {
      Teuchos::Array<double> Cp_in_0 = responseParams->get<Teuchos::Array<double> >("Transform 1 Row 0");
      Teuchos::Array<double> Cp_in_1 = responseParams->get<Teuchos::Array<double> >("Transform 1 Row 1");
      Teuchos::Array<double> Cp_in_2 = responseParams->get<Teuchos::Array<double> >("Transform 1 Row 2");
      formatTest = (formatTest && Cp_in_0.size()==3 && Cp_in_1.size()==3 && Cp_in_2.size()==3);
      Teuchos::Array<double> Cpp_in_0 = responseParams->get<Teuchos::Array<double> >("Transform 2 Row 0");
      Teuchos::Array<double> Cpp_in_1 = responseParams->get<Teuchos::Array<double> >("Transform 2 Row 1");
      Teuchos::Array<double> Cpp_in_2 = responseParams->get<Teuchos::Array<double> >("Transform 2 Row 2");
      formatTest = (formatTest && Cpp_in_0.size()==3 && Cpp_in_1.size()==3 && Cpp_in_2.size()==3);
      if (formatTest) {
        Cp(0,0)=Cp_in_0 [0];
        Cp(1,0)=Cp_in_1 [0];
        Cp(0,1)=Cp_in_0 [1];
        Cp(1,1)=Cp_in_1 [1];
        Cp(5,0)=Cp_in_2 [0];
        Cp(5,1)=Cp_in_2 [1];
        Cp(0,5)=Cp_in_0 [2];
        Cp(1,5)=Cp_in_1 [2];
        Cp(5,5)=Cp_in_2 [2];

        Cp(3,3)=1.0;
        Cp(4,4)=1.0;
        Cp(2,1)=-1.0;
        Cp(1,2)=-1.0;
        Cp(0,2)=-1.0;
        Cp(2,0)=-1.0;

        Cpp(0,0)=Cpp_in_0 [0];
        Cpp(1,0)=Cpp_in_1 [0];
        Cpp(0,1)=Cpp_in_0 [1];
        Cpp(1,1)=Cpp_in_1 [1];
        Cpp(5,0)=Cpp_in_2 [0];
        Cpp(5,1)=Cpp_in_2 [1];
        Cpp(0,5)=Cpp_in_0 [2];
        Cpp(1,5)=Cpp_in_1 [2];
        Cpp(5,5)=Cpp_in_2 [2];

        Cpp(3,3)=1.0;
        Cpp(4,4)=1.0;
        Cpp(2,1)=-1.0;
        Cpp(1,2)=-1.0;
        Cpp(0,2)=-1.0;
        Cpp(2,0)=-1.0;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(
            (!formatTest), Teuchos::Exceptions::InvalidParameter, std::endl
            << "Error!  Anisotropic transforms do not match dimensions given." << std::endl);

  } else {
    //If 2D/3D isn't recognized, just do the Von Mises transform.
    Cp(1,0) = -1.0;
    Cp(2,0) = -1.0;
    Cp(0,1) = -1.0;
    Cp(2,1) = -1.0;
    Cp(0,2) = -1.0;
    Cp(1,2) = -1.0;
    Cp(3,3) = 1.0;
    Cp(4,4) = 1.0;
    Cp(5,5) = 1.0;

    Cpp(1,0) = -1.0;
    Cpp(2,0) = -1.0;
    Cpp(0,1) = -1.0;
    Cpp(2,1) = -1.0;
    Cpp(0,2) = -1.0;
    Cpp(1,2) = -1.0;
    Cpp(3,3) = 1.0;
    Cpp(4,4) = 1.0;
    Cpp(5,5) = 1.0;
  }


  TEUCHOS_TEST_FOR_EXCEPTION(
    topology->getEntityType() != "Distributed Parameter",
    Teuchos::Exceptions::InvalidParameter, std::endl
    << "Error!  TensorPNormResponse requires 'Distributed Parameter' based topology" << std::endl);

  topo = PHX::MDField<ParamScalarT,Cell,Node>(topology->getName(),dl->node_scalar);

  this->pStateMgr = p.get< Albany::StateManager* >("State Manager Ptr");
  this->pStateMgr->registerStateVariable("Effective Stress",
                                         dl->cell_scalar, dl->dummy, 
                                         "all", "scalar", 0.0, false, true);

  this->addDependentField(qp_weights);
  this->addDependentField(BF);
  this->addDependentField(tensor);
  this->addDependentField(topo);

  // Create tag
  objective_tag =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Tensor PNorm", dl->dummy));
  this->addEvaluatedField(*objective_tag);
  
  std::string responseID = "ATO Tensor PNorm";
  this->setName(responseID);

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);

  int responseSize = 1;
  int worksetSize = dl->qp_scalar->dimension(0);
  Teuchos::RCP<PHX::DataLayout> 
    global_response_layout = Teuchos::rcp(new PHX::MDALayout<Dim>(responseSize));
  Teuchos::RCP<PHX::DataLayout> 
    local_response_layout  = Teuchos::rcp(new PHX::MDALayout<Cell,Dim>(worksetSize, responseSize));

  std::string local_response_name  = FName + " Local Response";
  std::string global_response_name = FName + " Global Response";

  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  p.set("Local Response Field Tag", local_response_tag);

  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Global Response Field Tag", global_response_tag);

  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qp_weights,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(tensor,fm);
  this->utils.setFieldData(topo,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  std::vector<int> dims;
  tensor.dimensions(dims);
  int size = dims.size();

  ScalarT pNorm=0.0;

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = dims[2];
  int numNodes = topo.dimension(1);

  Albany::MDArray effStress;
  effStress = (*workset.stateArrayPtr)["Effective Stress"];

  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT devNorm = 0.0;
        for(int i=0; i<numDims; i++)
          devNorm += tensor(cell,qp,i)*tensor(cell,qp,i);
        ScalarT P = topology->Penalize(functionIndex, topoVal);
        devNorm = P*sqrt(devNorm);
        ScalarT dS = pow(devNorm,pVal) * qp_weights(cell,qp);
        this->local_response(cell,0) += dS;
        pNorm += dS;
      }
    }
  } else
  if( size == 4 && numDims == 2 ){
    for(int cell=0; cell<numCells; cell++){
      ScalarT responseAve(0.0);
      RealType el_weight = 0.0;
      for(int qp=0; qp<numQPs; qp++){
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex, topoVal);
        ScalarT response_eff = 0.0;
        TransformResponse(cell,qp,response_eff);
        responseAve += response_eff*qp_weights(cell,qp);
        el_weight += qp_weights(cell,qp);
        ScalarT devNorm = P*response_eff/yieldStress;
        ScalarT dS = pow(devNorm,pVal) * qp_weights(cell,qp);
        this->local_response(cell,0) += dS;
        pNorm += dS;
      }
      saveState(effStress(cell,0), responseAve/el_weight);
    }
  } else
  if( size == 4 && numDims == 3 ){
    for(int cell=0; cell<numCells; cell++){
      ScalarT responseAve(0.0);
      RealType el_weight = 0.0;
      for(int qp=0; qp<numQPs; qp++){
        ScalarT topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex, topoVal);
        ScalarT response_eff = 0.0;
        TransformResponse(cell,qp,response_eff);
        responseAve += response_eff*qp_weights(cell,qp);
        el_weight += qp_weights(cell,qp);
        ScalarT devMag = P*response_eff/yieldStress;
        ScalarT dS = pow(devMag,pVal) * qp_weights(cell,qp);
        this->local_response(cell,0) += dS;
        pNorm += dS;
      }
      saveState(effStress(cell,0), responseAve/el_weight);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>4, Teuchos::Exceptions::InvalidParameter,
      "Unexpected array dimensions in Tensor PNorm Objective:" << size << std::endl);
  }

  this->global_response[0] += pNorm;

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
TransformResponse(int Cell, int QP, ScalarT& response_eff)
{
  //This class receives a field tensor, two (3D) linear transformations, and an exponent
  //It transforms the input into an effective response scalar using the Barlat Yield Model
  //As defined in Barlat et. al, 2004
  std::vector<int> dims;
  tensor.dimensions(dims);
  int size = dims.size();
  int numDims  = dims[2];

  ScalarT nullVal = 0.0;
  //Transform the tensor into Voigt notation.
  //This transform generalizes 2d/3d by leaving blanks, which is still correct.
  Intrepid2::Vector<ScalarT> voigt_tensor(6);

  voigt_tensor[0] = tensor(Cell,QP,0,0);
  voigt_tensor[1] = tensor(Cell,QP,1,1);
  voigt_tensor[2] = 0.0;
  voigt_tensor[3] = 0.0;
  voigt_tensor[4] = 0.0;
  voigt_tensor[5] = tensor(Cell,QP,0,1);
  if(numDims==3){
    voigt_tensor[2] = tensor(Cell,QP,2,2);
    voigt_tensor[3] = tensor(Cell,QP,1,2);
    voigt_tensor[4] = tensor(Cell,QP,0,2);
  }


  //Transform input tensor into its deviator. T is a constant transformation matrix.
  Intrepid2::Matrix<double> T (6,6);
  for (int i=0;i<6;i++)
    for (int j=0;j<6;j++)
      if (i<3 && j<3)
        if (i==j)
          T(i,j)=2.0/3.0;
        else
          T(i,j)=-1.0/3.0;
      else
        if (i==j)
          T(i,j)=1.0;
        else
          T(i,j)=0.0;

  Intrepid2::Vector<ScalarT> _s = T*voigt_tensor;

  //Use supplied linear transformations to get effective deviators
  Intrepid2::Vector<ScalarT> _sp =Cp*_s;
  Intrepid2::Vector<ScalarT> _spp =Cpp*_s;

  //Get invariates of the deviators. These are analytic equations, given in appendix A of Barlat, 2004.
  ScalarT Hp1 = (_sp[0]+_sp[1]+_sp[2]) / 3.0;
  ScalarT Hp2 = (_sp[3]*_sp[3]+_sp[4]*_sp[4]+_sp[5]*_sp[5]-_sp[1]*_sp[2]-_sp[2]*_sp[0]-_sp[0]*_sp[1]) / 3.0;
  ScalarT Hp3 = (2.0*_sp[3]*_sp[4]*_sp[5]+_sp[0]*_sp[1]*_sp[2]-_sp[0]*_sp[3]*_sp[3]-_sp[1]*_sp[4]*_sp[4]-_sp[2]*_sp[5]*_sp[5]) / 2.0;
  ScalarT Hpp1 = (_spp[0]+_spp[1]+_spp[2]) / 3.0;
  ScalarT Hpp2 = (_spp[3]*_spp[3]+_spp[4]*_spp[4]+_spp[5]*_spp[5]-_spp[1]*_spp[2]-_spp[2]*_spp[0]-_spp[0]*_spp[1]) / 3.0;
  ScalarT Hpp3 = (2.0*_spp[3]*_spp[4]*_spp[5]+_spp[0]*_spp[1]*_spp[2]-_spp[0]*_spp[3]*_spp[3]-_spp[1]*_spp[4]*_spp[4]-_spp[2]*_spp[5]*_spp[5]) / 2.0;

  //Get interim values for p, q, and theta. These are defined analytically in the Barlat text.
  ScalarT Pp = 0.0;
  if (Hp1*Hp1+Hp2>0.0)
    Pp = Hp1*Hp1+Hp2;
  ScalarT Qp = (2.0*Hp1*Hp1*Hp1 + 3.0*Hp1*Hp2 + 2.0*Hp3) / 2.0;
  ScalarT Thetap = acos(Qp/pow(Pp,3.0/2.0));

  ScalarT Ppp = 0.0;
  if (Hpp1*Hpp1+Hpp2>0.0)
    Ppp = Hpp1*Hpp1+Hpp2;
  ScalarT Qpp = (2.0*Hpp1*Hpp1*Hpp1 + 3.0*Hpp1*Hpp2 + 2.0*Hpp3) / 2.0;
  ScalarT Thetapp = acos( Qpp / pow(Ppp,3.0/2.0) );

  //Apply the analytic solutions for the SVD stress deviators
  ScalarT Sp1 = (Hp1 + 2.0*sqrt(Hp1*Hp1+Hp2)*cos(Thetap/3.0));
  ScalarT Sp2 = (Hp1 + 2.0*sqrt(Hp1*Hp1+Hp2)*cos((Thetap+4.0*acos(-1.0))/3.0));
  ScalarT Sp3 = (Hp1 + 2.0*sqrt(Hp1*Hp1+Hp2)*cos((Thetap+2.0*acos(-1.0))/3.0));

  ScalarT Spp1 = (Hpp1 + 2.0*sqrt(Hpp1*Hpp1+Hpp2)*cos(Thetapp/3.0));
  ScalarT Spp2 = (Hpp1 + 2.0*sqrt(Hpp1*Hpp1+Hpp2)*cos((Thetapp+4.0*acos(-1.0))/3.0));
  ScalarT Spp3 = (Hpp1 + 2.0*sqrt(Hpp1*Hpp1+Hpp2)*cos((Thetapp+2.0*acos(-1.0))/3.0));

  //Apply the Barlat yield function to get effective response.
  response_eff = pow((pow(Sp1-Spp1,R)+pow(Sp1-Spp2,R)+pow(Sp1-Spp3,R)+pow(Sp2-Spp1,R)+pow(Sp2-Spp2,R)+pow(Sp2-Spp3,R)+pow(Sp3-Spp1,R)+pow(Sp3-Spp2,R)+pow(Sp3-Spp3,R)) / 4.0 , 1.0/R);
}

// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::Residual, Traits>::
saveState(RealType& to, ScalarT from) { to = from; }

// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::Jacobian, Traits>::
saveState(RealType& to, ScalarT from) { to = from.val(); }

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponse<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
    // Add contributions across processors
    PHAL::reduceAll<ScalarT>(*workset.comm, Teuchos::REDUCE_SUM, this->global_response);

    TensorPNormResponseSpec<EvalT,Traits>::postEvaluate(workset);

    // Do global scattering
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ATO::TensorPNormResponseSpec<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);
}

// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);
}


// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::Jacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  double gVal = this->global_response[0].val();
  double scale = pow(gVal,1.0/pVal-1.0)/pVal;

  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxT = workset.overlapped_dgdxT;
  if (overlapped_dgdxT != Teuchos::null) overlapped_dgdxT->scale(scale);

  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdotT = workset.overlapped_dgdxdotT;
  if (overlapped_dgdxdotT != Teuchos::null) overlapped_dgdxdotT->scale(scale);
}

// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  double gVal = this->global_response[0].val();
  double scale = pow(gVal,1.0/pVal-1.0)/pVal;

  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdp = workset.overlapped_dgdp;
  if(overlapped_dgdp != Teuchos::null) overlapped_dgdp->Scale(scale);
#endif
}

#ifdef ALBANY_SG
// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::SGJacobian, Traits>::l
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdx_sg = workset.overlapped_sg_dgdx;
  if(overlapped_dgdx_sg != Teuchos::null){
    for(int block=0; block<overlapped_dgdx_sg->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdx_sg)[block].Scale(scale);
    }
  }

  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> overlapped_dgdxdot_sg = workset.overlapped_sg_dgdxdot;
  if(overlapped_dgdxdot_sg != Teuchos::null){
    for(int block=0; block<overlapped_dgdxdot_sg->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdxdot_sg)[block].Scale(scale);
    }
  }
}

#endif 
#ifdef ALBANY_ENSEMBLE 
// **********************************************************************
template<typename Traits>
void ATO::TensorPNormResponseSpec<PHAL::AlbanyTraits::MPJacobian, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  this->global_response[0] = pow(this->global_response[0],1.0/pVal);

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdx_mp = workset.overlapped_mp_dgdx;
  if(overlapped_dgdx_mp != Teuchos::null){
    for(int block=0; block<overlapped_dgdx_mp->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdx_mp)[block].Scale(scale);
    }
  }

  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> overlapped_dgdxdot_mp = workset.overlapped_mp_dgdxdot;
  if(overlapped_dgdxdot_mp != Teuchos::null){
    for(int block=0; block<overlapped_dgdxdot_mp->size(); block++){
      typename PHAL::Ref<ScalarT>::type gVal = this->global_response[0];
      double scale = pow(gVal.val().coeff(block),1.0/pVal-1.0)/pVal;
      (*overlapped_dgdxdot_mp)[block].Scale(scale);
    }
  }
}
#endif
