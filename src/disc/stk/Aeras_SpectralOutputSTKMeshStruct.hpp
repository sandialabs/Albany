//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AERAS_STK_FROM_SPECTRAL_MESHSTRUCT_HPP
#define AERAS_STK_FROM_SPECTRAL_MESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_AbstractDiscretization.hpp"

//#include <Ionit_Initializer.h>

namespace Aeras {

  class SpectralOutputSTKMeshStruct : public Albany::GenericSTKMeshStruct {

    public:

//Constructor 
    SpectralOutputSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const Teuchos::RCP<const Teuchos_Comm>& commT, 
                  const int numDim_, const int worksetSize_, 
                  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& wsElNodeID_, 
                  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& coords_,
                  const Teuchos::RCP<const Tpetra_Map>& node_mapT_, const int points_per_edge_);



    ~SpectralOutputSTKMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize);

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return false; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return -1.0; }
    
    //Is this necessary here? 
    const bool getInterleavedOrdering() const {return this->interleavedOrdering;}

    private:
    //Ioss::Init::Initializer ioInit;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    bool contigIDs; //boolean specifying if node / element / face IDs are contiguous; only relevant for 1 processor run 
    Teuchos::RCP<Tpetra_Map> node_mapT; //node map
    const int numDim;  
    const int points_per_edge;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type wsElNodeID;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;
     

    protected:
  };

}
#endif
