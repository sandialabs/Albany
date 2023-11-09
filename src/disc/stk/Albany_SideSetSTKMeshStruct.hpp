//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SRC_DISC_STK_ALBANY_SIDESETSTKMESHSTRUCT_HPP_
#define SRC_DISC_STK_ALBANY_SIDESETSTKMESHSTRUCT_HPP_

#include "Albany_GenericSTKMeshStruct.hpp"

namespace Albany
{

class SideSetSTKMeshStruct : public GenericSTKMeshStruct
{
public:

  SideSetSTKMeshStruct (const MeshSpecsStruct& inputMeshSpecs,
                        const Teuchos::RCP<Teuchos::ParameterList>& params,
                        const Teuchos::RCP<const Teuchos_Comm>& commT,
                        const int numParams);

  virtual ~SideSetSTKMeshStruct() = default;

  void setFieldData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                     const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                     const unsigned int worksetSize,
                     const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {});

  void setBulkData (const Teuchos::RCP<const Teuchos_Comm>& comm,
                    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                    const unsigned int worksetSize,
                    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {});

  void setParentMeshInfo (const AbstractSTKMeshStruct& parentMeshStruct_,
                          const std::string& sideSetName);

  bool hasRestartSolution () const {return false;}
  double restartDataTime () const {return 0.;}

  Teuchos::RCP<const Teuchos::ParameterList> getValidDiscretizationParameters() const;

private:

  Teuchos::RCP<const AbstractSTKMeshStruct>  parentMeshStruct; // Weak ptr
  std::string                                parentMeshSideSetName;
};

} // Namespace Albany

#endif /* SRC_DISC_STK_ALBANY_SIDESETSTKMESHSTRUCT_HPP_ */
