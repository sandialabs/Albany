//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ATO_Utils.hpp"
#include "ATO_Stress.hpp"
#include "ATO_Mixture.hpp"
#include "ATO_Mixture_DistParam.hpp"
#include "ATO_BodyForce.hpp"
#include "ATO_FixedFieldTerm.hpp"
#include "ATO_NeumannTerm.hpp"
#include "ATO_DirichletTerm.hpp"
#include "ATO_CreateField.hpp"
#include "ATO_TopologyFieldWeighting.hpp"
#include "ATO_TopologyWeighting.hpp"
#include "ATO_ScaleVector.hpp"
#include "Albany_Utils.hpp"
#include "Albany_DataTypes.hpp"
#include "ElasticityResid.hpp"

#ifdef ALBANY_STOKHOS
#include "ATO_ResidualStrain.hpp"
#endif

#include "PHAL_SaveStateField.hpp"
#include "PHAL_SaveCellStateField.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"

template<typename EvalT, typename Traits>
ATO::Utils<EvalT,Traits>::Utils(
     Teuchos::RCP<Albany::Layouts> dl_, int numDim_ ) :
     dl(dl_), numDim(numDim_)
{
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::SaveCellStateField(
       PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &variableName,
       const std::string &elementBlockName,
       const Teuchos::RCP<PHX::DataLayout>& dataLayout)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p;
    Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

    //
    // QUAD POINT SCALARS
    if( dataLayout == dl->qp_scalar ){

      // save cell average for output
      p = stateMgr.registerStateVariable(variableName+"_ave",
          dl->cell_scalar, dl->dummy, elementBlockName, "scalar",
          0.0, false, true);
      p->set("Field Layout", dl->qp_scalar);
      p->set("Field Name", variableName);
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Weights Name", "Weights");
      ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else

    //
    // QUAD POINT VECTORS
    if(dataLayout == dl->qp_vector){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++){
        std::string varname(variableName);
        varname += " ";
        varname += cn[i];
        varname += "_ave ";
        p = stateMgr.registerStateVariable(varname,
            dl->cell_scalar, dl->dummy, elementBlockName, "scalar",
            0.0, false, true);
        p->set("Field Layout", dl->qp_vector);
        p->set("Field Name", variableName);
        p->set("Weights Layout", dl->qp_scalar);
        p->set("Weights Name", "Weights");
        p->set("component i", i);
        ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    } else

    //
    // QUAD POINT TENSORS
    if(dataLayout == dl->qp_tensor){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++)
        for(int j=0; j< numDim; j++){
          std::string varname(variableName);
          varname += " ";
          varname += cn[i];
          varname += cn[j];
          varname += "_ave ";
          p = stateMgr.registerStateVariable(varname,
              dl->cell_scalar, dl->dummy, elementBlockName, "scalar",
              0.0, false, true);
          p->set("Field Layout", dl->qp_tensor);
          p->set("Field Name", variableName);
          p->set("Weights Layout", dl->qp_scalar);
          p->set("Weights Name", "Weights");
          p->set("component i", i);
          p->set("component j", j);
          ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
    }
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructFluxEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName,
       std::string fluxName, std::string gradName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;

    Teuchos::RCP<ParameterList> p = rcp(new ParameterList(fluxName));

    //Input
    p->set<std::string>("Input Vector Name", gradName);
    p->set< RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    if( params->isType<int>("Add Cell Problem Forcing") )
      p->set<int>("Cell Forcing Column",params->get<int>("Add Cell Problem Forcing") );

    // check for multiple element block specs
    Teuchos::ParameterList& configParams = params->sublist("Configuration");

    if( configParams.isSublist("Element Blocks") ){
      Teuchos::ParameterList& blocksParams = configParams.sublist("Element Blocks");
      int nblocks = blocksParams.get<int>("Number of Element Blocks");
      bool blockFound = false;
      for(int ib=0; ib<nblocks; ib++){
        Teuchos::ParameterList& blockParams = blocksParams.sublist(Albany::strint("Element Block", ib));
        std::string blockName = blockParams.get<std::string>("Name");
        if( blockName != elementBlockName ) continue;
        blockFound = true;

        // user can specify a material or a mixture
        if( blockParams.isSublist("Material") ){
          // parse material
          Teuchos::ParameterList& materialParams = blockParams.sublist("Material",false);
          if( materialParams.isSublist("Homogenized Constants") ){
            Teuchos::ParameterList& homogParams = p->sublist("Homogenized Constants",false);
            homogParams.setParameters(materialParams.sublist("Homogenized Constants",true));
            p->set<Albany::StateManager*>("State Manager", &stateMgr);
            p->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
          } else {
            p->set<double>("Coefficient", materialParams.get<double>("Isotropic Modulus"));
          }
          //Output
          p->set<std::string>("Output Vector Name", fluxName);

          ev = rcp(new ATO::ScaleVector<EvalT,Traits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          // state the strain in the state manager so for ATO
          p = stateMgr.registerStateVariable(fluxName,dl->qp_vector, dl->dummy,
                                             elementBlockName, "scalar", 0.0, false, false);
          ev = rcp(new PHAL::SaveStateField<EvalT,Traits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          //if(some input stuff)
          SaveCellStateField(fm0, stateMgr, fluxName, elementBlockName, dl->qp_vector);

        } else
        if( blockParams.isSublist("Mixture") ){
          // parse mixture
          Teuchos::ParameterList& mixtureParams = blockParams.sublist("Mixture",false);
          int nmats = mixtureParams.get<int>("Number of Materials");

          //-- create individual materials --//
          for(int imat=0; imat<nmats; imat++){
            Teuchos::RCP<ParameterList> pmat = rcp(new ParameterList(*p));
            Teuchos::ParameterList& materialParams = mixtureParams.sublist(Albany::strint("Material", imat));
            if( materialParams.isSublist("Homogenized Constants") ){
              Teuchos::ParameterList& homogParams = pmat->sublist("Homogenized Constants",false);
              homogParams.setParameters(materialParams.sublist("Homogenized Constants",true));
              pmat->set<Albany::StateManager*>("State Manager", &stateMgr);
              pmat->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
            } else {
              pmat->set<double>("Coefficient", materialParams.get<double>("Isotropic Modulus"));
            }
            //Output
            std::string outName = Albany::strint(fluxName, imat);
            pmat->set<std::string>("Output Vector Name", outName);

            ev = rcp(new ATO::ScaleVector<EvalT,Traits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);

            // state the strain in the state manager so for ATO
            pmat = stateMgr.registerStateVariable(outName,dl->qp_vector, dl->dummy,
                                                  elementBlockName, "scalar", 0.0, false, false);
            ev = rcp(new PHAL::SaveStateField<EvalT,Traits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);

            //if(some input stuff)
            SaveCellStateField(fm0, stateMgr, outName, elementBlockName, dl->qp_vector);
          }

          //-- create mixture --//
          TEUCHOS_TEST_FOR_EXCEPTION( !mixtureParams.isSublist("Mixed Fields"), std::logic_error,
                                  "'Mixture' requested but no 'Fields' defined"  << std::endl <<
                                  "Add 'Fields' list");
          {
            Teuchos::ParameterList& fieldsParams = mixtureParams.sublist("Mixed Fields",false);
            int nfields = fieldsParams.get<int>("Number of Mixed Fields");


            //-- create individual mixture field evaluators --//
            for(int ifield=0; ifield<nfields; ifield++){
              Teuchos::ParameterList& fieldParams = fieldsParams.sublist(Albany::strint("Mixed Field", ifield));
              std::string fieldName = fieldParams.get<std::string>("Field Name");

              Teuchos::RCP<ParameterList> p = rcp(new ParameterList(fieldName + " Mixed Field"));

              std::string fieldLayout = fieldParams.get<std::string>("Field Layout");
              p->set<std::string>("Field Layout", fieldLayout);

              std::string mixtureRule = fieldParams.get<std::string>("Rule Type");

              // currently only SIMP-type mixture is implemented
              Teuchos::ParameterList& simpParams = fieldParams.sublist(mixtureRule);

              Teuchos::RCP<ATO::TopologyArray>
                topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");
              p->set<Teuchos::RCP<ATO::TopologyArray> > ("Topologies", topologyArray);

              // topology and function indices
              p->set<Teuchos::Array<int> >("Topology Indices",
                                           simpParams.get<Teuchos::Array<int> >("Topology Indices"));
              p->set<Teuchos::Array<int> >("Function Indices",
                                           simpParams.get<Teuchos::Array<int> >("Function Indices"));

              // constituent var names
              Teuchos::Array<int> matIndices = simpParams.get<Teuchos::Array<int> >("Material Indices");
              int nMats = matIndices.size();
              Teuchos::Array<int> topoIndices = simpParams.get<Teuchos::Array<int> >("Topology Indices");
              int nTopos = topoIndices.size();
              TEUCHOS_TEST_FOR_EXCEPTION(nMats != nTopos+1, std::logic_error, std::endl <<
                                        "For SIMP Mixture, 'Materials' list must be 1 longer than 'Topologies' list"
                                        << std::endl);
              Teuchos::Array<std::string> constituentNames(nMats);
              for(int imat=0; imat<nmats; imat++){
                std::string constituentName = Albany::strint(fieldName, matIndices[imat]);
                constituentNames[imat] = constituentName;
              }
              p->set<Teuchos::Array<std::string> >("Constituent Variable Names", constituentNames);

              // mixture var name
              p->set<std::string>("Mixture Variable Name",fieldName);

              // basis functions
              p->set<std::string>("BF Name", "BF");

              TEUCHOS_TEST_FOR_EXCEPTION(topologyArray->size() == 0, std::logic_error, std::endl <<
                                        "Mixture requested with no topologies defined!" << std::endl);
              Teuchos::RCP<ATO::Topology> topology = (*topologyArray)[0];

              if( topology->getEntityType() == "Distributed Parameter" ){
                ev = rcp(new ATO::Mixture_DistParam<EvalT,Traits>(*p,dl));
              } else {
                ev = rcp(new ATO::Mixture<EvalT,Traits>(*p,dl));
              }
              fm0.template registerEvaluator<EvalT>(ev);
            }
          }
        } else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "'Material' or 'Mixture' not specified for '"
                                  << elementBlockName << "'");
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!blockFound, std::logic_error,
                                 "Material definition for block named '" << elementBlockName << "' not found");
    } else {

      if( params->isSublist("Homogenized Constants") ){
        Teuchos::ParameterList& homogParams = p->sublist("Homogenized Constants",false);
        homogParams.setParameters(params->sublist("Homogenized Constants",true));
        p->set<Albany::StateManager*>("State Manager", &stateMgr);
        p->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
      } else {
        p->set<double>("Coefficient", params->get<double>("Isotropic Modulus"));
      }

      //Output
      p->set<std::string>("Output Vector Name", fluxName);

      ev = rcp(new ATO::ScaleVector<EvalT,Traits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      p = stateMgr.registerStateVariable(fluxName,dl->qp_vector, dl->dummy,
                                         elementBlockName, "scalar", 0.0, false, false);
      ev = rcp(new PHAL::SaveStateField<EvalT,Traits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      SaveCellStateField(fm0, stateMgr, fluxName, elementBlockName, dl->qp_vector);
    }

}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructStressEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName,
       std::string stressName, std::string strainName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;

    Teuchos::RCP<ParameterList> p = rcp(new ParameterList(stressName));

    //Input
    p->set<std::string>("Strain Name", strainName);
    p->set< RCP<PHX::DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    if( params->isType<int>("Add Cell Problem Forcing") )
      p->set<int>("Cell Forcing Column",params->get<int>("Add Cell Problem Forcing") );

    // check for multiple element block specs
    Teuchos::ParameterList& configParams = params->sublist("Configuration");

    if( configParams.isSublist("Element Blocks") ){
      Teuchos::ParameterList& blocksParams = configParams.sublist("Element Blocks");
      int nblocks = blocksParams.get<int>("Number of Element Blocks");
      bool blockFound = false;
      for(int ib=0; ib<nblocks; ib++){
        Teuchos::ParameterList& blockParams = blocksParams.sublist(Albany::strint("Element Block", ib));
        std::string blockName = blockParams.get<std::string>("Name");
        if( blockName != elementBlockName ) continue;
        blockFound = true;

        // user can specify a material or a mixture
        if( blockParams.isSublist("Material") ){
          // parse material
          Teuchos::ParameterList& materialParams = blockParams.sublist("Material",false);
          if( materialParams.isSublist("Homogenized Constants") ){
            Teuchos::ParameterList& homogParams = p->sublist("Homogenized Constants",false);
            homogParams.setParameters(materialParams.sublist("Homogenized Constants",true));
            p->set<Albany::StateManager*>("State Manager", &stateMgr);
            p->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
          } else {
            p->set<double>("Elastic Modulus", materialParams.get<double>("Elastic Modulus"));
            p->set<double>("Poissons Ratio",  materialParams.get<double>("Poissons Ratio"));
          }
          //Output
          p->set<std::string>("Stress Name", stressName);

          ev = rcp(new ATO::Stress<EvalT,Traits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          // state the strain in the state manager so for ATO
          p = stateMgr.registerStateVariable(stressName,dl->qp_tensor, dl->dummy,
                                             elementBlockName, "scalar", 0.0, false, false);
          ev = rcp(new PHAL::SaveStateField<EvalT,Traits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          //if(some input stuff)
          SaveCellStateField(fm0, stateMgr, stressName, elementBlockName, dl->qp_tensor);

        } else
        if( blockParams.isSublist("Mixture") ){
          // parse mixture
          Teuchos::ParameterList& mixtureParams = blockParams.sublist("Mixture",false);
          int nmats = mixtureParams.get<int>("Number of Materials");

          //-- create individual materials --//
          for(int imat=0; imat<nmats; imat++){
            Teuchos::RCP<ParameterList> pmat = rcp(new ParameterList(*p));
            Teuchos::ParameterList& materialParams = mixtureParams.sublist(Albany::strint("Material", imat));
            if( materialParams.isSublist("Homogenized Constants") ){
              Teuchos::ParameterList& homogParams = pmat->sublist("Homogenized Constants",false);
              homogParams.setParameters(materialParams.sublist("Homogenized Constants",true));
              pmat->set<Albany::StateManager*>("State Manager", &stateMgr);
              pmat->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
            } else {
              pmat->set<double>("Elastic Modulus", materialParams.get<double>("Elastic Modulus"));
              pmat->set<double>("Poissons Ratio",  materialParams.get<double>("Poissons Ratio"));
            }
            //Output
            std::string outName = Albany::strint(stressName, imat);
            pmat->set<std::string>("Stress Name", outName);

            ev = rcp(new ATO::Stress<EvalT,Traits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);

            // state the strain in the state manager so for ATO
            pmat = stateMgr.registerStateVariable(outName,dl->qp_tensor, dl->dummy,
                                                  elementBlockName, "scalar", 0.0, false, false);
            ev = rcp(new PHAL::SaveStateField<EvalT,Traits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);

            //if(some input stuff)
            SaveCellStateField(fm0, stateMgr, outName, elementBlockName, dl->qp_tensor);
          }

          //-- create mixture --//
          TEUCHOS_TEST_FOR_EXCEPTION( !mixtureParams.isSublist("Mixed Fields"), std::logic_error,
                                  "'Mixture' requested but no 'Fields' defined"  << std::endl <<
                                  "Add 'Fields' list");
          {
            Teuchos::ParameterList& fieldsParams = mixtureParams.sublist("Mixed Fields",false);
            int nfields = fieldsParams.get<int>("Number of Mixed Fields");


            //-- create individual mixture field evaluators --//
            for(int ifield=0; ifield<nfields; ifield++){
              Teuchos::ParameterList& fieldParams = fieldsParams.sublist(Albany::strint("Mixed Field", ifield));
              std::string fieldName = fieldParams.get<std::string>("Field Name");

              Teuchos::RCP<ParameterList> p = rcp(new ParameterList(fieldName + " Mixed Field"));

              std::string fieldLayout = fieldParams.get<std::string>("Field Layout");
              p->set<std::string>("Field Layout", fieldLayout);

              std::string mixtureRule = fieldParams.get<std::string>("Rule Type");

              // currently only SIMP-type mixture is implemented
              Teuchos::ParameterList& simpParams = fieldParams.sublist(mixtureRule);

              Teuchos::RCP<ATO::TopologyArray>
                topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");
              p->set<Teuchos::RCP<ATO::TopologyArray> > ("Topologies", topologyArray);

              // topology and function indices
              p->set<Teuchos::Array<int> >("Topology Indices",
                                           simpParams.get<Teuchos::Array<int> >("Topology Indices"));
              p->set<Teuchos::Array<int> >("Function Indices",
                                           simpParams.get<Teuchos::Array<int> >("Function Indices"));

              // constituent var names
              Teuchos::Array<int> matIndices = simpParams.get<Teuchos::Array<int> >("Material Indices");
              int nMats = matIndices.size();
              Teuchos::Array<int> topoIndices = simpParams.get<Teuchos::Array<int> >("Topology Indices");
              int nTopos = topoIndices.size();
              TEUCHOS_TEST_FOR_EXCEPTION(nMats != nTopos+1, std::logic_error, std::endl <<
                                        "For SIMP Mixture, 'Materials' list must be 1 longer than 'Topologies' list"
                                        << std::endl);
              Teuchos::Array<std::string> constituentNames(nMats);
              for(int imat=0; imat<nmats; imat++){
                std::string constituentName = Albany::strint(fieldName, matIndices[imat]);
                constituentNames[imat] = constituentName;
              }
              p->set<Teuchos::Array<std::string> >("Constituent Variable Names", constituentNames);

              // mixture var name
              p->set<std::string>("Mixture Variable Name",fieldName);

              // basis functions
              p->set<std::string>("BF Name", "BF");

             TEUCHOS_TEST_FOR_EXCEPTION(topologyArray->size() == 0, std::logic_error, std::endl <<
                                       "Mixture requested with no topologies defined!" << std::endl);
             Teuchos::RCP<ATO::Topology> topology = (*topologyArray)[0];

             if( topology->getEntityType() == "Distributed Parameter" ){
               ev = rcp(new ATO::Mixture_DistParam<EvalT,Traits>(*p,dl));
             } else {
               ev = rcp(new ATO::Mixture<EvalT,Traits>(*p,dl));
             }
             fm0.template registerEvaluator<EvalT>(ev);
            }
          }
        } else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "'Material' or 'Mixture' not specified for '"
                                  << elementBlockName << "'");
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!blockFound, std::logic_error,
                                 "Material definition for block named '" << elementBlockName << "' not found");
    } else {

      if( params->isSublist("Homogenized Constants") ){
        Teuchos::ParameterList& homogParams = p->sublist("Homogenized Constants",false);
        homogParams.setParameters(params->sublist("Homogenized Constants",true));
        p->set<Albany::StateManager*>("State Manager", &stateMgr);
        p->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
      } else {
        p->set<double>("Elastic Modulus", params->get<double>("Elastic Modulus"));
        p->set<double>("Poissons Ratio",  params->get<double>("Poissons Ratio"));
      }

      //Output
      p->set<std::string>("Stress Name", stressName);

      ev = rcp(new ATO::Stress<EvalT,Traits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // state the strain in the state manager so for ATO
      p = stateMgr.registerStateVariable(stressName,dl->qp_tensor, dl->dummy,
                                         elementBlockName, "scalar", 0.0, false, false);
      ev = rcp(new PHAL::SaveStateField<EvalT,Traits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      //if(some input stuff)
      SaveCellStateField(fm0, stateMgr, stressName, elementBlockName, dl->qp_tensor);
    }

}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructBoundaryConditionEvaluators(
       const Teuchos::ParameterList& bcSpec,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string& boundaryName,
       std::string boundaryForceName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;

    Teuchos::RCP<ParameterList> p = rcp(new ParameterList(boundaryForceName));

    // Parse boundary conditions and create the appropriate evaluators
    //
    int numDirichlet = bcSpec.get<int>("Number of Dirichlet BCs");
    for(int iDirichlet=0; iDirichlet<numDirichlet; iDirichlet++){
      const Teuchos::ParameterList& dirichletSpec = bcSpec.sublist(Albany::strint("Dirichlet BC",iDirichlet));

      if(dirichletSpec.get<std::string>("Boundary") != boundaryName) continue;

      RCP<ParameterList> p = rcp(new ParameterList("Dirichlet Penalty Term"));

      p->set<double>("Penalty Coefficient",dirichletSpec.get<double>("Penalty Coefficient"));
      p->set<std::string>("Variable Name",dirichletSpec.get<std::string>("Variable Name"));

      p->set<std::string>("Dirichlet Name", boundaryForceName);
      if(dirichletSpec.get<std::string>("Layout") == "QP Scalar"){
        p->set< RCP<PHX::DataLayout> >("Data Layout", dl->qp_scalar);
        if(dirichletSpec.isType<double>("Scalar")) p->set<double>("Scalar",dirichletSpec.get<double>("Scalar"));
        ev = rcp(new ATO::DirichletScalarTerm<EvalT,Traits>(*p));
      } else
      if(dirichletSpec.get<std::string>("Layout") == "QP Vector"){
        p->set< RCP<PHX::DataLayout> >("Data Layout", dl->qp_vector);
        if(dirichletSpec.isType<double>("X")) p->set<double>("X",dirichletSpec.get<double>("X"));
        if(dirichletSpec.isType<double>("Y")) p->set<double>("Y",dirichletSpec.get<double>("Y"));
        if(dirichletSpec.isType<double>("Z")) p->set<double>("Z",dirichletSpec.get<double>("Z"));
        ev = rcp(new ATO::DirichletVectorTerm<EvalT,Traits>(*p));
      } else
        TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error,
          "'Layout' in Dirichlet BCs ParameterList can be 'QP Vector' or 'QP Scalar'"  << std::endl);

      fm0.template registerEvaluator<EvalT>(ev);
    }



    int numNeumann = bcSpec.get<int>("Number of Neumann BCs");
    for(int iNeumann=0; iNeumann<numNeumann; iNeumann++){
      const Teuchos::ParameterList& neumannSpec = bcSpec.sublist(Albany::strint("Neumann BC",iNeumann));

      if(neumannSpec.get<std::string>("Boundary") != boundaryName) continue;

      RCP<ParameterList> p = rcp(new ParameterList("Neumann Term"));

      p->set<std::string>("Neumann Name", boundaryForceName);

      if(neumannSpec.get<std::string>("Layout") == "QP Vector"){
        p->set< RCP<PHX::DataLayout> >("Data Layout", dl->qp_vector);
        p->set<Teuchos::Array<double> >("Vector",
          neumannSpec.get<Teuchos::Array<double> >("Vector"));
        ev = rcp(new ATO::NeumannVectorTerm<EvalT,Traits>(*p));
      } else
      if(neumannSpec.get<std::string>("Layout") == "QP Scalar"){
        p->set< RCP<PHX::DataLayout> >("Data Layout", dl->qp_scalar);
        p->set<double>("Scalar",neumannSpec.get<double>("Scalar"));
        ev = rcp(new ATO::NeumannScalarTerm<EvalT,Traits>(*p));
      } else
        TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::logic_error,
          "'Layout' in Neumann BCs ParameterList can be 'QP Vector' or 'QP Scalar'"  << std::endl);
      fm0.template registerEvaluator<EvalT>(ev);
    }
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructFixedFieldTermEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName,
       std::string dof_name,
       std::string fixedFieldTermName)
{
    if( ! params->isSublist("Fixed Field") ) return;

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;

    Teuchos::RCP<ParameterList> p = rcp(new ParameterList(fixedFieldTermName));

    p->set<std::string>("Output Name", fixedFieldTermName);
    p->set<std::string>("Field Name", dof_name);
    p->set< RCP<PHX::DataLayout> >("Data Layout", dl->qp_scalar);

    Teuchos::ParameterList& ffParams = params->sublist("Fixed Field",false);

    p->set<double>("Penalty Coefficient", ffParams.get<double>("Penalty Coefficient"));
    p->set<double>("Fixed Value", ffParams.get<double>("Fixed Value"));

    ev = rcp(new ATO::FixedFieldTerm<EvalT,Traits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructBodyForceEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName,
       std::string bodyForceName)
{
    if( ! params->isSublist("Body Force") ) return;

    std::string densityName("Density");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;


    Teuchos::RCP<ParameterList> p = rcp(new ParameterList(bodyForceName));

    // check for multiple element block specs
    Teuchos::ParameterList& configParams = params->sublist("Configuration");

    if( configParams.isSublist("Element Blocks") ){
      Teuchos::ParameterList& blocksParams = configParams.sublist("Element Blocks");
      int nblocks = blocksParams.get<int>("Number of Element Blocks");
      bool blockFound = false;
      for(int ib=0; ib<nblocks; ib++){
        Teuchos::ParameterList& blockParams = blocksParams.sublist(Albany::strint("Element Block", ib));
        std::string blockName = blockParams.get<std::string>("Name");
        if( blockName != elementBlockName ) continue;
        blockFound = true;

        // material spec given
        //
        if( blockParams.isSublist("Material") ){
          Teuchos::RCP<ParameterList> pDensity = rcp(new ParameterList(densityName));
          // parse material
          Teuchos::ParameterList& materialParams = blockParams.sublist("Material",false);
          double density = materialParams.get<double>("Density");
          pDensity->set<std::string>("Field Name", densityName);
          pDensity->set<double>("Field Value", density);
          pDensity->set< RCP<PHX::DataLayout> >("Field Data Layout", dl->qp_scalar);
          ev = rcp(new ATO::CreateField<EvalT,Traits>(*pDensity));
          fm0.template registerEvaluator<EvalT>(ev);

        } else

        // mixture spec given
        //
        if( blockParams.isSublist("Mixture") ){
          // parse mixture
          Teuchos::ParameterList& mixtureParams = blockParams.sublist("Mixture",false);
          int nmats = mixtureParams.get<int>("Number of Materials");

          //-- create individual materials --//
          for(int imat=0; imat<nmats; imat++){
            Teuchos::RCP<ParameterList> pmat = rcp(new ParameterList(densityName));
            Teuchos::ParameterList& materialParams = mixtureParams.sublist(Albany::strint("Material", imat));
            double density = materialParams.get<double>("Density");
            pmat->set<std::string>("Field Name", Albany::strint(densityName, imat));
            pmat->set<double>("Field Value", density);
            pmat->set< RCP<PHX::DataLayout> >("Field Data Layout", dl->qp_scalar);
            ev = rcp(new ATO::CreateField<EvalT,Traits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);
          }

          //-- create mixture --//
          TEUCHOS_TEST_FOR_EXCEPTION( !mixtureParams.isSublist("Mixed Parameters"), std::logic_error,
                                  "'Mixture' requested but no 'Parameters' defined"  << std::endl <<
                                  "Add 'Parameters' list");
          {
            Teuchos::ParameterList& paramsParams = mixtureParams.sublist("Mixed Parameters",false);
            int nparams = paramsParams.get<int>("Number of Mixed Parameters");

            //-- create individual mixture param evaluators --//
            for(int iparam=0; iparam<nparams; iparam++){
              Teuchos::ParameterList& paramParams = paramsParams.sublist(Albany::strint("Mixed Parameter", iparam));
              std::string paramName = paramParams.get<std::string>("Parameter Name");

              Teuchos::RCP<ParameterList> p = rcp(new ParameterList(paramName + " Mixed Parameter"));

              p->set<std::string>("Field Layout", "QP Scalar");

              std::string mixtureRule = paramParams.get<std::string>("Rule Type");

              // currently only SIMP-type mixture is implemented
              Teuchos::ParameterList& simpParams = paramParams.sublist(mixtureRule);

              Teuchos::RCP<ATO::TopologyArray>
                topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");
              p->set<Teuchos::RCP<ATO::TopologyArray> > ("Topologies", topologyArray);

              // topology and function indices
              p->set<Teuchos::Array<int> >("Topology Indices",
                                           simpParams.get<Teuchos::Array<int> >("Topology Indices"));
              p->set<Teuchos::Array<int> >("Function Indices",
                                           simpParams.get<Teuchos::Array<int> >("Function Indices"));

              // constituent var names
              Teuchos::Array<int> matIndices = simpParams.get<Teuchos::Array<int> >("Material Indices");
              int nMats = matIndices.size();
              Teuchos::Array<int> topoIndices = simpParams.get<Teuchos::Array<int> >("Topology Indices");
              int nTopos = topoIndices.size();
              TEUCHOS_TEST_FOR_EXCEPTION(nMats != nTopos+1, std::logic_error, std::endl <<
                                        "For SIMP Mixture, 'Materials' list must be 1 longer than 'Topologies' list"
                                        << std::endl);
              Teuchos::Array<std::string> constituentNames(nMats);
              for(int imat=0; imat<nmats; imat++){
                std::string constituentName = Albany::strint(paramName, matIndices[imat]);
                constituentNames[imat] = constituentName;
              }
              p->set<Teuchos::Array<std::string> >("Constituent Variable Names", constituentNames);

              // mixture var name
              p->set<std::string>("Mixture Variable Name",paramName);

              // basis functions
              p->set<std::string>("BF Name", "BF");

              TEUCHOS_TEST_FOR_EXCEPTION(topologyArray->size() == 0, std::logic_error, std::endl <<
                                        "Mixture requested with no topologies defined!" << std::endl);
              Teuchos::RCP<ATO::Topology> topology = (*topologyArray)[0];

              if( topology->getEntityType() == "Distributed Parameter" ){
                ev = rcp(new ATO::Mixture_DistParam<EvalT,Traits>(*p,dl));
              } else {
                ev = rcp(new ATO::Mixture<EvalT,Traits>(*p,dl));
              }
              fm0.template registerEvaluator<EvalT>(ev);
            }
          }
        } else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "'Material' or 'Mixture' not specified for '"
                                  << elementBlockName << "'");
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!blockFound, std::logic_error,
                                 "Material definition for block named '" << elementBlockName << "' not found");
    } else {

      Teuchos::RCP<ParameterList> pDensity = rcp(new ParameterList("Density"));
      double density = params->get<double>("Density");
      pDensity->set<std::string>("Field Name", densityName);
      pDensity->set<double>("Field Value", density);
      pDensity->set< RCP<PHX::DataLayout> >("Field Data Layout", dl->qp_scalar);
      ev = rcp(new ATO::CreateField<EvalT,Traits>(*pDensity));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    { // Body forces
      RCP<ParameterList> p = rcp(new ParameterList("Body Force"));

      Teuchos::ParameterList& configParams = params->sublist("Configuration");
      p->set<Teuchos::ParameterList>("Configuration",configParams);

      p->set<std::string>("Body Force Name", bodyForceName);
      p->set< RCP<PHX::DataLayout> >("Body Force Data Layout", dl->qp_vector);

      p->set<std::string>("Density Field Name", densityName);
      p->set< RCP<PHX::DataLayout> >("Density Field Data Layout", dl->qp_scalar);

      Teuchos::ParameterList& bfParams = params->sublist("Body Force",false);
      p->set<Teuchos::Array<double> >("Body Force Direction",
         bfParams.get<Teuchos::Array<double> >("Body Force Direction"));
      p->set<double>("Body Force Magnitude",
         bfParams.get<double>("Body Force Magnitude"));

      ev = rcp(new ATO::BodyForce<EvalT,Traits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    SaveCellStateField(fm0, stateMgr, "Density", elementBlockName, dl->qp_scalar);
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructResidualStressEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName,
       std::string residForceName)
{
#ifdef ALBANY_STOKHOS

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;


    std::string residStrain("Residual Strain");
    std::string residStress("Residual Stress");

    // create KL strain
    //
    Teuchos::RCP<ParameterList> p = rcp(new ParameterList("Residual Strain"));
    ParameterList& residStrainParams = params->sublist("Residual Strain");
    p->set<Teuchos::ParameterList*>("Parameter List", &residStrainParams);

    p->set<std::string>("QP Variable Name", residStrain);
    p->set<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    ev = rcp(new ATO::ResidualStrain<EvalT,Traits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    SaveCellStateField(fm0, stateMgr, residStrain, elementBlockName, dl->qp_tensor);

    // compute KL stress from KL strain
    //
    constructStressEvaluators(params, fm0, stateMgr, elementBlockName, residStress, residStrain);

    SaveCellStateField(fm0, stateMgr, residStress, elementBlockName, dl->qp_tensor);
#else
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error, "The evaluator ATO::ResiduaalStrain requires ALBANY_STOKHOS to be defined.\n");
#endif
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::constructWeightedFieldEvaluators(
       const Teuchos::RCP<Teuchos::ParameterList>& params,
       PHX::FieldManager<Traits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &elementBlockName,
//       Teuchos::RCP<PHX::DataLayout> layout,
       std::string layoutName,
       std::string& inputFieldName)
{


  if(params->isType<Teuchos::RCP<ATO::TopologyArray> >("Topologies"))
  {

    Teuchos::RCP<PHX::Evaluator<Traits> > ev;

    Teuchos::RCP<ATO::TopologyArray>
      topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");

    Teuchos::ParameterList& wfParams = params->sublist("Apply Topology Weight Functions");
    bool foundField = false;
    int nfields = wfParams.get<int>("Number of Fields");
    for(int ifield=0; ifield<nfields; ifield++){
      Teuchos::ParameterList& fieldParams = wfParams.sublist(Albany::strint("Field", ifield));

      std::string fieldName  = fieldParams.get<std::string>("Name");

      if(fieldName != inputFieldName) continue;

      foundField = true;

      int topoIndex  = fieldParams.get<int>("Topology Index");
      int functionIndex      = fieldParams.get<int>("Function Index");

      Teuchos::RCP<ATO::Topology> topology = (*topologyArray)[topoIndex];

      // Get distributed parameter
      if( topology->getEntityType() == "Distributed Parameter" ){
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Distributed Parameter"));
        p->set<std::string>("Parameter Name", topology->getName());
        ev = Teuchos::rcp(new PHAL::GatherScalarNodalParameter<EvalT,PHAL::AlbanyTraits>(*p, dl) );
        fm0.template registerEvaluator<EvalT>(ev);
      }

      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("TopologyWeighting"));

      p->set<Teuchos::RCP<ATO::Topology> >("Topology",topology);

      p->set<std::string>("BF Name", "BF");
      p->set<std::string>("Unweighted Variable Name", inputFieldName);
      inputFieldName += "_Weighted";
      p->set<std::string>("Weighted Variable Name", inputFieldName);
      p->set<std::string>("Variable Layout", layoutName);
      p->set<int>("Function Index", functionIndex);

      if( topology->getEntityType() == "Distributed Parameter" )
        ev = Teuchos::rcp(new ATO::TopologyFieldWeighting<EvalT,PHAL::AlbanyTraits>(*p,dl));
      else
        ev = Teuchos::rcp(new ATO::TopologyWeighting<EvalT,PHAL::AlbanyTraits>(*p,dl));

      fm0.template registerEvaluator<EvalT>(ev);

      break;
    }
    if( !foundField ){
      // error out.  Instructions for weighting requested field not found.
    }
  } else {
    // error out.  Topology weighting requested without defining topology.
  }

}
