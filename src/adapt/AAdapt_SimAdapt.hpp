//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_AbstractAdapterT.hpp"
#include "Teuchos_VerboseObject.hpp"
/* BRD */
class SGModel;
/* BRD */

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
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters();
/* BRD */
  virtual ~SimAdapt() {
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
  double errorBound;
  
  // output stream
  Teuchos::RCP<Teuchos::FancyOStream> out;
};

}
