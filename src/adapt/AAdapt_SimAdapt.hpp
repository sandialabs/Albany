//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_AbstractAdapterT.hpp"

namespace AAdapt {

struct SimAdaptImpl;

class SimAdapt : public AbstractAdapterT {
public:
  SimAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
           const Teuchos::RCP<ParamLib>& paramLib_,
           const Albany::StateManager& StateMgr_,
           const Teuchos::RCP<const Teuchos_Comm>& commT_);
  virtual bool queryAdaptationCriteria(int iteration);
  // virtual bool adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
  //                        const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);
  virtual bool adaptMesh();
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;
private:
  double errorBound;
};

}
