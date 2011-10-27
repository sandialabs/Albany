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

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void 
    buildProblem(
       Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
       StateManager& stateMgr,
       Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

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

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT>
    void constructEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr,
            Albany::FieldManagerChoice fmchoice,
            Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses);

    //! Interface for Residual (PDE) field manager
    template <typename EvalT>
    void constructResidEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr)
    {
      Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> > junk;
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, BUILD_RESID_FM, junk);
    }

    //! Interface for Response field manager, except for residual type
    template <typename EvalT>
    void constructResponseEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr)
    {
      Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> > junk;
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, BUILD_RESPONSE_FM, junk);
    }

    //! Interface for Response field manager, Residual type.
    // This version loads the responses variable, that needs to be constructed just once
    template <typename EvalT>
    void constructResponseEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr,
            Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
    {
      constructEvaluators<EvalT>(fm0, meshSpecs, stateMgr, BUILD_RESPONSE_FM, responses);
    }
    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int T_offset;  //Position of T unknown in nodal DOFs
    int X_offset;  //Position of X unknown in nodal DOFs, followed by Y,Z
    int numDim;    //Number of spatial dimensions and displacement variable 

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
#include "PHAL_SaveStateField.hpp"
#include "Porosity.hpp"
#include "BiotCoefficient.hpp"
#include "BiotModulus.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "KCPermeability.hpp"
#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "TotalStress.hpp"
#include "PoroElasticityResidMomentum.hpp"
#include "PHAL_Source.hpp"
#include "PoroElasticityResidMass.hpp"


template <typename EvalT>
void Albany::PoroElasticityProblem::constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& meshSpecs,
        Albany::StateManager& stateMgr,
        Albany::FieldManagerChoice fieldManagerChoice,
        Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
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
   const int numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;


   // Construct standard FEM evaluators with standard field names                              
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
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
     p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy,"zero", true);
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
     p = stateMgr.registerStateVariable("Strain",dl->qp_tensor, dl->dummy,"zero",true);
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

	  // Setting this turns on linear dependence of E on T, E = E_ + dEdT*T)
	  p->set<string>("Strain Name", "Strain");
	  p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
	  p->set<string>("QP Pore Pressure Name", "Pore Pressure");
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

          ev = rcp(new LCM::Porosity<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
          p = stateMgr.registerStateVariable("Porosity",dl->qp_scalar, dl->dummy,"zero", true);
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
          p = stateMgr.registerStateVariable("Biot Coefficient",dl->qp_scalar, dl->dummy,"zero");
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
          p = stateMgr.registerStateVariable("Biot Modulus",dl->qp_scalar, dl->dummy,"zero");
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
     p = stateMgr.registerStateVariable("Kozeny-Carman Permeability",dl->qp_scalar, dl->dummy,"zero");
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



//  if (haveSource) { // Source
 //   TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
 //                      "Error!  Sources not implemented in Elasticity yet!");
//
  //  RCP<ParameterList> p = rcp(new ParameterList);
//
  //  p->set<string>("Source Name", "Source");
 //   p->set<string>("Variable Name", "Displacement");
 //   p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
//
 //   p->set<RCP<ParamLib> >("Parameter Library", paramLib);
 //   Teuchos::ParameterList& paramList = params->sublist("Source Functions");
 //   p->set<Teuchos::ParameterList*>("Parameter List", &paramList);
//
 //   ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
 //   fm0.template registerEvaluator<EvalT>(ev);
//  }



  { // Total Stress
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
    p = stateMgr.registerStateVariable("Total Stress",dl->qp_tensor, dl->dummy,"zero");
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Pore Pressure",dl->qp_scalar, dl->dummy,"zero", true);
	ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
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





  { // Pore Pressure Resid
    RCP<ParameterList> p = rcp(new ParameterList("Pore Pressure Resid"));

    //Input

    // Input from nodal points
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<string>("QP Pore Pressure Name", "Pore Pressure");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("QP Time Derivative Variable Name", "Pore Pressure");

    p->set<bool>("Have Source", false);
    p->set<string>("Source Name", "Source");

    p->set<bool>("Have Absorption", false);

    // Input from cubature points
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
  }
  else {
    //Construct Responses
    Teuchos::ParameterList& responseList = params->sublist("Response Functions");
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    respUtils.constructResponses(fm0, responses, responseList, stateMgr);
  }
}

#endif // POROELASTICITYPROBLEM_HPP
