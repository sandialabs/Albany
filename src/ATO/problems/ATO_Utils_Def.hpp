//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ATO_Utils.hpp"
#include "Albany_DataTypes.hpp"

#include "PHAL_SaveCellStateField.hpp"

template<typename EvalT, typename Traits>
ATO::Utils<EvalT,Traits>::Utils(
     Teuchos::RCP<Albany::Layouts> dl_) :
     dl(dl_)
{
}

template<typename EvalT, typename Traits>
void ATO::Utils<EvalT,Traits>::SaveCellStateField(
       PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
       Albany::StateManager& stateMgr,
       const std::string &variableName,
       const std::string &elementBlockName,
       const Teuchos::RCP<PHX::DataLayout>& dataLayout, int numDim)
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
