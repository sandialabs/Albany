//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_AbstractAdapterT.hpp"
#include "Teuchos_RCP.hpp"

/* BRD */
class SGModel;
/* BRD */

namespace AAdapt {

struct SimAdaptImpl;

class SimLayerAdapt : public AbstractAdapterT {
public:
  SimLayerAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
           const Teuchos::RCP<ParamLib>& paramLib_,
           const Albany::StateManager& StateMgr_,
           const Teuchos::RCP<const Teuchos_Comm>& commT_);
  virtual bool queryAdaptationCriteria(int iteration);
  // virtual bool adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
  //                        const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution);
  virtual bool adaptMesh();
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;
/* BRD */
  virtual ~SimLayerAdapt() {
    if (Simmetrix_numLayers > 0)
      delete []  Simmetrix_layerTimes;
  };
protected:
  void computeLayerTimes();
  double *Simmetrix_layerTimes;
  int    Simmetrix_numLayers;
  int    Simmetrix_currentLayer;
  SGModel *Simmetrix_model;
/* BRD */
private:
  //! Output stream, defaults to printing just Proc 0
  Teuchos::RCP<Teuchos::FancyOStream> out;
  // When added a new layer we assign a uniform temperature to it
  double initTempNewLayer;
};

}
