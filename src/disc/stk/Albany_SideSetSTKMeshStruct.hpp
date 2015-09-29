//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SIDE_SET_STK_MESH_STRUCT_HPP
#define ALBANY_SIDE_SET_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

namespace Albany
{

class SideSetSTKMeshStruct : public GenericSTKMeshStruct
{
public:

  SideSetSTKMeshStruct (const MeshSpecsStruct& inputMeshSpecs,
                        const Teuchos::RCP<Teuchos::ParameterList>& params,
                        const Teuchos::RCP<const Teuchos_Comm>& commT);

  virtual ~SideSetSTKMeshStruct();

  void setFieldAndBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                            const Teuchos::RCP<Teuchos::ParameterList>& params,
                            const unsigned int neq_,
                            const AbstractFieldContainer::FieldContainerRequirements& req,
                            const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                            const unsigned int worksetSize,
                            const Teuchos::RCP<std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> > >& ss_sis = Teuchos::null);

  void extractEntitiesFromSTKMesh(const AbstractSTKMeshStruct& inputMeshStruct,
                                  const std::string& sideSetName);

  bool hasRestartSolution () const {return false;}
  double restartDataTime () const {return 0.;}
};

} // Namespace Albany

#endif // ALBANY_SIDE_SET_STK_MESH_STRUCT_HPP
