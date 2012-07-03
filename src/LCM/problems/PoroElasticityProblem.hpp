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


#ifndef POROELASTICITYPROBLEM_HPP
#define POROELASTICITYPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

  /*!
   * \brief Problem definition for Poro-Elasticity
   */
  class PoroElasticityProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    PoroElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
			  const Teuchos::RCP<ParamLib>& paramLib,
			  const int numEq);

    //! Destructor
    virtual ~PoroElasticityProblem();

     //Set problem information for computation of rigid body modes (in src/Albany_SolverFactory.cpp)
    void getRBMInfoForML(
         int& numPDEs, int& numElasticityDim, int& numScalar, int& nullSpaceDim);

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(
         Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
	 Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
	 ) const;

  private:

    //! Private to prohibit copying
    PoroElasticityProblem(const PoroElasticityProblem&);
    
    //! Private to prohibit copying
    PoroElasticityProblem& operator=(const PoroElasticityProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> 
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

    std::string matModel;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState;
  };
}

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "Time.hpp"
#include "Strain.hpp"
#include "AssumedStrain.hpp"
#include "StabParameter.hpp"
#include "PHAL_SaveStateField.hpp"
#include "Porosity.hpp"
#include "BiotCoefficient.hpp"
#include "BiotModulus.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "KCPermeability.hpp"
#include "ElasticModulus.hpp"
#include "ShearModulus.hpp"
#include "PoissonsRatio.hpp"
#include "TotalStress.hpp"
#include "Stress.hpp"
#include "PoroElasticityResidMomentum.hpp"
#include "PHAL_Source.hpp"
#include "PoroElasticityResidMass.hpp"

#include "PHAL_NSMaterialProperty.hpp"

// Plasticity model from Q.Chen
#include "CapModelStress.hpp"
#include "GursonSDStress.hpp"


template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::PoroElasticityProblem::constructEvaluators(
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
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;


   // Construct standard FEM evaluators with standard field names                              
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Mechanics problems assume vecDim = numDim");
   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   string scatterName="Scatter PoreFluid";


   // ----------------------setup the solution field ---------------//

   // Displacement Variable
   Teuchos::ArrayRCP<string> dof_names(1);
     dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<string> resid_names(1);
     resid_names[0] = dof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, X_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names, X_offset));

  // Pore Pressure Variable
   Teuchos::ArrayRCP<string> tdof_names(1);
     tdof_names[0] = "Pore Pressure";
   Teuchos::ArrayRCP<string> tdof_names_dot(1);
     tdof_names_dot[0] = tdof_names[0]+"_dot";
   Teuchos::ArrayRCP<string> tresid_names(1);
     tresid_names[0] = tdof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFInterpolationEvaluator(tdof_names[0]));

     (evalUtils.constructDOFInterpolationEvaluator(tdof_names_dot[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFGradInterpolationEvaluator(tdof_names[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherSolutionEvaluator(false, tdof_names, tdof_names_dot, T_offset));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(false, tresid_names, T_offset, scatterName));

   // ----------------------setup the solution field ---------------//

   // General FEM stuff
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

   { // Time
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<string>("Time Name", "Time");
     p->set<string>("Delta Time Name", " Delta Time");
     p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
     p->set<RCP<ParamLib> >("Parameter Library", paramLib);
     p->set<bool>("Disable Transient", true);

     ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy,"scalar", 0.0, true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Spatial Stabilization Parameter Field
        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<string>("Stabilization Parameter Name", "Stabilization Parameter");
        p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

        p->set<RCP<ParamLib> >("Parameter Library", paramLib);
        Teuchos::ParameterList& paramList = params->sublist("Stabilization Parameter");
        p->set<Teuchos::ParameterList*>("Parameter List", &paramList);


        // Additional information to construct stabilization parameter field
        p->set<string>("Gradient QP Variable Name", "Pore Pressure Gradient");
		p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

		p->set<string>("Gradient BF Name", "Grad BF");
		p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

		p->set<string>("Diffusive Parameter Name", "Kozeny-Carman Permeability");
		p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

        ev = rcp(new LCM::StabParameter<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);

        p = stateMgr.registerStateVariable("Stabilization Parameter",dl->qp_scalar, dl->dummy,"scalar", 5.0e-7, true);
        ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }


   { // Strain
     RCP<ParameterList> p = rcp(new ParameterList("Strain"));

     //Input
     p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
     p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

     //Output
     p->set<string>("Strain Name", "Strain"); //dl->qp_tensor also

     ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Strain",dl->qp_tensor, dl->dummy,"scalar", 0.0,true);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

   { // Assumed Strain
       RCP<ParameterList> p = rcp(new ParameterList("Assumed Strain"));

       //Inputs: flags, weights, GradU
       const bool avgJ = params->get("avgJ", false);
       p->set<bool>("avgJ Name", avgJ);
       const bool volavgJ = params->get("volavgJ", false);
       p->set<bool>("volavgJ Name", volavgJ);
       const bool weighted_Volume_Averaged_J = params->get("weighted_Volume_Averaged_J", false);
       p->set<bool>("weighted_Volume_Averaged_J Name", weighted_Volume_Averaged_J);
       p->set<string>("Weights Name","Weights");
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
       p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
       p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

       //Outputs: F, J
       p->set<string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also
       p->set<string>("Assumed Strain Name", "Assumed Strain"); //dl->qp_tensor also
       p->set<string>("DetDefGrad Name", "Jacobian");
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

       ev = rcp(new LCM::AssumedStrain<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
       p = stateMgr.registerStateVariable("Assumed Strain",dl->qp_tensor, dl->dummy,"scalar", 0.0,true);
       ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
       fm0.template registerEvaluator<EvalT>(ev);
     }

   {  // Porosity
      RCP<ParameterList> p = rcp(new ParameterList);

	  p->set<string>("Porosity Name", "Porosity");
	  p->set<string>("QP Coordinate Vector Name", "Coord Vec");
	  p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	  Teuchos::ParameterList& paramList = params->sublist("Porosity");
	  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

	  // Setting this turns on dependence of strain and pore pressure)
	  p->set<string>("Strain Name", "Strain");
	  p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
//	  p->set<string>("QP Pore Pressure Name", "Pore Pressure");
//	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
//	  p->set<string>("Biot Coefficient Name", "Biot Coefficient");

          ev = rcp(new LCM::Porosity<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
          p = stateMgr.registerStateVariable("Porosity",dl->qp_scalar, dl->dummy,"scalar", 0.4, true);
          ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
     }



   { // Biot Coefficient
      RCP<ParameterList> p = rcp(new ParameterList);

	  p->set<string>("Biot Coefficient Name", "Biot Coefficient");
	  p->set<string>("QP Coordinate Vector Name", "Coord Vec");
	  p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	  Teuchos::ParameterList& paramList = params->sublist("Biot Coefficient");
	  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

	  // Setting this turns on linear dependence on porosity
	  p->set<string>("Porosity Name", "Porosity");

          ev = rcp(new LCM::BiotCoefficient<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
          p = stateMgr.registerStateVariable("Biot Coefficient",dl->qp_scalar, dl->dummy,"scalar", 1.0);
          ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
  }

   { // Biot Modulus
         RCP<ParameterList> p = rcp(new ParameterList);

   	  p->set<string>("Biot Modulus Name", "Biot Modulus");
   	  p->set<string>("QP Coordinate Vector Name", "Coord Vec");
   	  p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
   	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
   	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

   	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
   	  Teuchos::ParameterList& paramList = params->sublist("Biot Modulus");
   	  p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

   	  // Setting this turns on linear dependence on porosity and Biot's coeffcient
   	  p->set<string>("Porosity Name", "Porosity");
          p->set<string>("Biot Coefficient Name", "Biot Coefficient");

          ev = rcp(new LCM::BiotModulus<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
          p = stateMgr.registerStateVariable("Biot Modulus",dl->qp_scalar, dl->dummy,"scalar", 1.0e10);
          ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
     }

  { // Thermal conductivity
   RCP<ParameterList> p = rcp(new ParameterList);

   p->set<string>("QP Variable Name", "Thermal Conductivity");
   p->set<string>("QP Coordinate Vector Name", "Coord Vec");
   p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
   p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
   p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

   p->set<RCP<ParamLib> >("Parameter Library", paramLib);
   Teuchos::ParameterList& paramList = params->sublist("Thermal Conductivity");
   p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::ThermalConductivity<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Kozeny-Carman Permeaiblity
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<string>("Kozeny-Carman Permeability Name", "Kozeny-Carman Permeability");
     p->set<string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib> >("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Kozeny-Carman Permeability");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     // Setting this turns on Kozeny-Carman relation
     p->set<string>("Porosity Name", "Porosity");
     p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

     ev = rcp(new LCM::KCPermeability<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
     p = stateMgr.registerStateVariable("Kozeny-Carman Permeability",dl->qp_scalar, dl->dummy,"scalar", 0.0);
     ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
    }

  // Skeleton parameter

  { // Elastic Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Elastic Modulus");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Elastic Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set<string>("Porosity Name", "Porosity"); // porosity is defined at Cubature points
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Shear Modulus
     RCP<ParameterList> p = rcp(new ParameterList);

     p->set<string>("QP Variable Name", "Shear Modulus");
     p->set<string>("QP Coordinate Vector Name", "Coord Vec");
     p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
     p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
     p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

     p->set<RCP<ParamLib> >("Parameter Library", paramLib);
     Teuchos::ParameterList& paramList = params->sublist("Shear Modulus");
     p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

     ev = rcp(new LCM::ShearModulus<EvalT,AlbanyTraits>(*p));
     fm0.template registerEvaluator<EvalT>(ev);
   }

  { // Poissons Ratio 
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("QP Variable Name", "Poissons Ratio");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on linear dependence of nu on T, nu = nu_ + dnudT*T)
    //p->set<string>("QP Pore Pressure Name", "Pore Pressure");

    ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (matModel == "CapModel")
  {
	{ // Cap model stress
	  RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<string>("Strain Name", "Assumed Strain");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      double A = params->get("A", 1.0);
      double B = params->get("B", 1.0);
      double C = params->get("C", 1.0);
      double theta = params->get("theta", 1.0);
      double R = params->get("R", 1.0);
      double kappa0 = params->get("kappa0", 1.0);
      double W = params->get("W", 1.0);
      double D1 = params->get("D1", 1.0);
      double D2 = params->get("D2", 1.0);
      double calpha = params->get("calpha", 1.0);
      double psi = params->get("psi", 1.0);
      double N = params->get("N", 1.0);
      double L = params->get("L", 1.0);
      double phi = params->get("phi", 1.0);
      double Q = params->get("Q", 1.0);

      p->set<double>("A Name", A);
      p->set<double>("B Name", B);
      p->set<double>("C Name", C);
      p->set<double>("Theta Name", theta);
      p->set<double>("R Name", R);
      p->set<double>("Kappa0 Name", kappa0);
      p->set<double>("W Name", W);
      p->set<double>("D1 Name", D1);
      p->set<double>("D2 Name", D2);
      p->set<double>("Calpha Name", calpha);
      p->set<double>("Psi Name", psi);
      p->set<double>("N Name", N);
      p->set<double>("L Name", L);
      p->set<double>("Phi Name", phi);
      p->set<double>("Q Name", Q);


      //Output
      p->set<string>("Stress Name", "Stress"); //dl->qp_tensor also
      p->set<string>("Back Stress Name", "backStress"); //dl->qp_tensor also
      p->set<string>("Cap Parameter Name", "capParameter"); //dl->qp_tensor also

      //Declare what state data will need to be saved (name, layout, init_type)
      ev = rcp(new LCM::CapModelStress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy,"scalar", 0.0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("backStress",dl->qp_tensor, dl->dummy,"scalar", 0.0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("capParameter",dl->qp_scalar, dl->dummy,"scalar", kappa0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
	}
  }

  else if (matModel == "GursonSD")
  {
	{ // Gurson small deformation stress
	  RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<string>("Strain Name", "Assumed Strain");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      double f0 = params->get("f0", 0.0);
      double Y0 = params->get("Y0", 100.0);
      double kw = params->get("kw", 0.0);
      double N = params->get("N", 1.0);
      double q1 = params->get("q1", 1.0);
      double q2 = params->get("q2", 1.0);
      double q3 = params->get("q3", 1.0);
      double eN = params->get("eN", 0.1);
      double sN = params->get("sN", 0.1);
      double fN = params->get("fN", 0.1);
      double fc = params->get("fc", 1.0);
      double ff = params->get("ff", 1.0);
      double flag = params->get("flag", 1.0);

      p->set<double>("f0 Name", f0);
      p->set<double>("Y0 Name", Y0);
      p->set<double>("kw Name", kw);
      p->set<double>("N Name", N);
      p->set<double>("q1 Name", q1);
      p->set<double>("q2 Name", q2);
      p->set<double>("q3 Name", q3);
      p->set<double>("eN Name", eN);
      p->set<double>("sN Name", sN);
      p->set<double>("fN Name", fN);
      p->set<double>("fc Name", fc);
      p->set<double>("ff Name", ff);
      p->set<double>("flag Name", flag);

      //Output
      p->set<string>("Stress Name", "Stress"); //dl->qp_tensor also
      p->set<string>("Void Volume Name", "voidVolume"); //dl->qp_scalar also
      p->set<string>("ep Name", "ep"); //dl->qp_scalar also
      p->set<string>("Yield Strength Name", "yieldStrength"); //dl->qp_scalar also

      //Declare what state data will need to be saved (name, layout, init_type)
      ev = rcp(new LCM::GursonSDStress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy,"scalar", 0.0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("voidVolume",dl->qp_scalar, dl->dummy,"scalar", f0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("ep",dl->qp_scalar, dl->dummy,"scalar", 0.0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("yieldStrength",dl->qp_scalar, dl->dummy,"scalar", Y0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
	}
  }

  else
  {
	{ // Linear elasticity stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<string>("Strain Name", "Assumed Strain");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      //Output
      p->set<string>("Stress Name", "Stress"); //dl->qp_tensor also

      ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy,"scalar", 0.0);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
	}
  }

  { // Total Stress
    RCP<ParameterList> p = rcp(new ParameterList("Total Stress"));

    //Input
    p->set<string>("Effective Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Biot Coefficient Name", "Biot Coefficient");  // dl->qp_scalar also
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("QP Variable Name", "Pore Pressure");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    //Output
    p->set<string>("Total Stress Name", "Total Stress"); //dl->qp_tensor also

    ev = rcp(new LCM::TotalStress<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Total Stress",dl->qp_tensor, dl->dummy,"scalar", 0.0);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Pore Pressure",dl->qp_scalar, dl->dummy,"scalar", 0.0, true);
	ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
	fm0.template registerEvaluator<EvalT>(ev);

  }


/*  { // Total Stress
    RCP<ParameterList> p = rcp(new ParameterList("Total Stress"));

    //Input
    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Elastic Modulus Name", "Elastic Modulus");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

    p->set<string>("Biot Coefficient Name", "Biot Coefficient");  // dl->qp_scalar also
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("QP Variable Name", "Pore Pressure");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    //Output
    p->set<string>("Total Stress Name", "Total Stress"); //dl->qp_tensor also

    ev = rcp(new LCM::TotalStress<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Total Stress",dl->qp_tensor, dl->dummy,"scalar", 0.0);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Pore Pressure",dl->qp_scalar, dl->dummy,"scalar", 0.0, true);
	ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
	fm0.template registerEvaluator<EvalT>(ev);

  } */

  if (haveSource) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<string>("Source Name", "Source");
    p->set<string>("QP Variable Name", "Pore Pressure");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
    p->set<string>("Total Stress Name", "Total Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<bool>("Disable Transient", true);


    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::PoroElasticityResidMomentum<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }



  { // Pore Pressure Resid
    RCP<ParameterList> p = rcp(new ParameterList("Pore Pressure Resid"));

    //Input

    // Input from nodal points
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<bool>("Have Source", false);
    p->set<string>("Source Name", "Source");

    p->set<bool>("Have Absorption", false);

    // Input from cubature points
    p->set<string>("QP Pore Pressure Name", "Pore Pressure");
	p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

	p->set<string>("QP Time Derivative Variable Name", "Pore Pressure");

	p->set<string>("Material Property Name", "Stabilization Parameter");
    p->set<string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<string>("Porosity Name", "Porosity");
    p->set<string>("Kozeny-Carman Permeability Name", "Kozeny-Carman Permeability");
    p->set<string>("Biot Coefficient Name", "Biot Coefficient");
    p->set<string>("Biot Modulus Name", "Biot Modulus");

    p->set<string>("Gradient QP Variable Name", "Pore Pressure Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    p->set<string>("Weights Name","Weights");

    p->set<string>("Delta Time Name", " Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);

    //Output
    p->set<string>("Residual Name", "Pore Pressure Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new LCM::PoroElasticityResidMass<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);

    PHX::Tag<typename EvalT::ScalarT> res_tag2(scatterName, dl->dummy);
    fm0.requireField<EvalT>(res_tag2);

    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}

#endif // POROELASTICITYPROBLEM_HPP
