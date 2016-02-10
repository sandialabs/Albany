//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHASEPROBLEM_HPP
#define PHASEPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace Albany {

///
/// \brief Definition for the Phase problem
///
class PhaseProblem : public AbstractProblem
{
public:
 
  PhaseProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
	       const Teuchos::RCP<ParamLib>& param_lib,
	       const int num_dims,
 	       Teuchos::RCP<const Teuchos::Comm<int> >& commT);

  ~PhaseProblem();

  virtual 
  int spatialDimension() const { return num_dims_; }

  virtual
  void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr);

  virtual 
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  Teuchos::RCP<const Teuchos::ParameterList> 
  getValidProblemParameters() const;

private:

  PhaseProblem(const PhaseProblem&);
    
  PhaseProblem& operator=(const PhaseProblem&);

public:

  template <typename EvalT> 
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(
      const std::vector<std::string>& nodeSetIDs);
    
  void constructNeumannEvaluators(
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  int num_dims_;

  Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

  Teuchos::RCP<Albany::Layouts> dl_;

};

}

//******************************************************************************

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_SaveStateField.hpp"

#include "RhoCp.hpp"
#include "Phi.hpp"
#include "Psi.hpp"
#include "ThermalCond.hpp"
#include "PhaseSource.hpp"
#include "LaserSource.hpp"
#include "PhaseResidual.hpp"
#include "EnergyDot.hpp"
#include "AMP_Time.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::PhaseProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;
  using PHAL::AlbanyTraits;

  // // Problem is transient
  // TEUCHOS_TEST_FOR_EXCEPTION(
  //     number_of_time_deriv != 1,
  //     std::logic_error,
  //     "PhaseProblem must be defined as a transient calculation.");

  const CellTopologyData* const elem_top = &meshSpecs.ctd;

  std::string eb_name = meshSpecs.ebName;
 // check name of element blocks. Must be Powder and Solid
  // M. Juha: I am forcing this here because I need to pass the
  // properties of both materials to ThermalCond evaluator. Anyway,
  // for this analysis we will always have powder and solid (substrate) as
  // materials. Please, if you find a better way to do it, then change it!
  // check element block names
  if ((eb_name != "Powder_Region") && (eb_name != "Solid_Region")){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"*** Element block names must be Powder_Region or Solid_Region ***\n")
  }
  
 // check names of material names
  std::string material_name;
  material_name = material_db_->getElementBlockParam<std::string>(eb_name,"material");
  if ( (material_name != "Solid") && (material_name != "Powder") ){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"*** Material name must be Powder or Solid ***\n")
  }
 
  RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > >
    intrepid_basis = Albany::getIntrepid2Basis(*elem_top);

  RCP<shards::CellTopology> elem_type = 
    rcp(new shards::CellTopology (elem_top));

  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cub_factory;

  RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > elem_cubature = 
    cub_factory.create(*elem_type, meshSpecs.cubatureDegree);

  const int workset_size = meshSpecs.worksetSize;
  const int num_vertices = elem_type->getNodeCount();
  const int num_nodes = intrepid_basis->getCardinality();
  const int num_qps = elem_cubature->getNumPoints();

  *out << "Field Dimensions: Workset=" << workset_size 
       << ", Vertices= "               << num_vertices
       << ", Nodes= "                  << num_nodes
       << ", QuadPts= "                << num_qps
       << ", Dim= "                    << num_dims_ 
       << std::endl;

  dl_ = rcp(new Albany::Layouts(
        workset_size,num_vertices,num_nodes,num_qps,num_dims_));

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);

  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  Teuchos::ArrayRCP<string> dof_names(1);
  Teuchos::ArrayRCP<string> dof_names_dot(1);
  Teuchos::ArrayRCP<string> resid_names(1);

  dof_names[0] = "Temperature";
  dof_names_dot[0] = "Temperature_dot";
  resid_names[0] = "Temperature Residual";

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(
      false,dof_names,dof_names_dot));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(
      false,resid_names));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator(
      elem_type,elem_cubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(
      elem_type,intrepid_basis,elem_cubature));

  { // Time
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Time"));
   
    // Input
    p->set<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout",
        dl_->workset_scalar);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    // Output
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
   
    // Register evaluator
    ev = Teuchos::rcp(new AMP::Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Register state variable
    p = stateMgr.registerStateVariable("Time", dl_->workset_scalar,
        dl_->dummy, eb_name, "scalar", 0.0, true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  
  { // Temperature
     
     double temp(0.0);
     if (material_db_->isElementBlockParam(eb_name, "Initial Temperature")) {
       temp = material_db_->
           getElementBlockParam<double>(eb_name, "Initial Temperature");
     }
     Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
         new Teuchos::ParameterList("Save Temperature"));
     p = stateMgr.registerStateVariable("Temperature",
         dl_->qp_scalar,
         dl_->dummy,
         eb_name,
         "scalar",
         temp,
         true,
         false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     
  }

  { //Phi
    Teuchos::RCP<ParameterList> p = rcp(new ParameterList("Phi parameters"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Initial Phi");

    // Input
    p->set<string>("Temperature Name","Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Phi Name","Phi");
    p->set<string>("Psi Name","Psi");

    ev = rcp(new AMP::Phi<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
    
    p = stateMgr.registerStateVariable("Phi", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", 0.0, true);
    
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 

  }

  { //Psi
    RCP<ParameterList> p = rcp(new ParameterList("Psi parameters"));
    
    double psi_initial(0.0);
    if (material_db_->isElementBlockSublist(eb_name, "Initial Psi")) 
      {
	Teuchos::ParameterList& param = 
	  material_db_->getElementBlockSublist(eb_name, "Initial Psi"); 
        psi_initial = param.get<double>("Psi");
      }

    Teuchos::ParameterList& param_list = 
      material_db_->getElementBlockSublist(eb_name, "Initial Psi"); 

    // Input
    p->set<string>("Phi Name","Phi");
    p->set<string>("Temperature Name","Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list); 

    //Output
    p->set<string>("Psi Name","Psi");

    ev = rcp(new AMP::Psi<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
    

    p = stateMgr.registerStateVariable("Psi", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", psi_initial, true);
    
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 
  }


  { // Thermal Conductivity
    RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity"));
    
   
    // get data for powder
    Teuchos::ParameterList& param_list_powder =
      material_db_->getElementBlockSublist("Powder_Region", "Thermal Conductivity");  
    
    // get data for substrate (solid))
    Teuchos::ParameterList& param_list_solid =
      material_db_->getElementBlockSublist("Solid_Region", "Thermal Conductivity");  

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<Teuchos::ParameterList*>("Powder Parameter List", &param_list_powder);
    p->set<Teuchos::ParameterList*>("Solid Parameter List", &param_list_solid);

    //Output
    p->set<string>("Thermal Conductivity Name", "k");
    p->set<string>("Psi Name", "Psi");

    ev = rcp(new AMP::ThermalCond<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Rho Cp
    RCP<ParameterList> p = rcp(new ParameterList("Specific Heat"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Rho Cp");    

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Rho Cp Name", "Rho Cp");

    ev = rcp(new AMP::RhoCp<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Source Function
    RCP<ParameterList> p = rcp(new ParameterList("Source Function"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Source");

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Source Name", "Source");
    
    ev = rcp(new AMP::PhaseSource<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  
  { // Laser Source Function
    RCP<ParameterList> p = rcp(new ParameterList("Laser Source Function"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Laser Source"); 

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Laser Source Name", "Laser Source");
    
    ev = rcp(new AMP::LaserSource<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }  
  
  
  { // Energy dot
    RCP<ParameterList> p = rcp(new ParameterList("Energy Rate Params"));

    // take phase change parameter list
    Teuchos::ParameterList& param_list_phase =
      material_db_->getElementBlockSublist(eb_name, "Phase Change Properties"); 
    
    //Input
    p->set<string>("Temperature Name","Temperature");
    p->set<string>("Temperature Time Derivative Name","Temperature_dot");
    p->set<string>("Phi Name","Phi");
    p->set<string>("Psi Name","Psi");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    p->set<string>("Rho Cp Name", "Rho Cp");
    p->set<Teuchos::ParameterList*>("Phase Change Parameter List", &param_list_phase);
    
    // take initial Psi parameter list
    Teuchos::ParameterList& param_list_psi =
      material_db_->getElementBlockSublist(eb_name, "Initial Phi");
    p->set<Teuchos::ParameterList*>("Initial Phi Parameter List", &param_list_psi);
    
    //Output
    p->set<string>("Energy Rate Name", "Energy Rate");

    ev = rcp(new AMP::EnergyDot<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  

  { // Phase Residual
    RCP<ParameterList> p = rcp(new ParameterList("u Resid"));

    //Input
    p->set<string>("Weighted BF Name","wBF");
    p->set<string>("Weighted Gradient BF Name","wGrad BF");
    p->set<string>("Temperature Name","Temperature");
    p->set<string>("Temperature Gradient Name","Temperature Gradient");
    p->set<string>("Temperature Time Derivative Name","Temperature_dot");
    p->set<string>("Thermal Conductivity Name","k");
    p->set<string>("Rho Cp Name","Rho Cp");
    p->set<string>("Source Name","Source");
    p->set<string>("Laser Source Name","Laser Source");
    p->set<string>("Phi Name","Phi");
    p->set<string>("Psi Name","Psi");
    p->set<string>("Energy Rate Name", "Energy Rate");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");

    //Output
    p->set<string>("Residual Name", "Temperature Residual");

    ev = rcp(new AMP::PhaseResidual<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}

#endif
