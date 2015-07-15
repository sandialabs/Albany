#include "Albany_SimOutput.hpp"
#include <apfSIM.h>
#include <SimModel.h>
#include <SimPartitionedMesh.h>
#include <SimField.h>

static std::string removeExtension(std::string const& s)
{
  std::size_t pos = s.find_last_of('.');
  if (pos == std::string::npos)
    return s;
  return s.substr(0, pos);
}

Albany::SimOutput::SimOutput(
    const Teuchos::RCP<APFMeshStruct>& meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_):
  commT(commT_),
  index(1)
{
  filename = removeExtension(meshStruct_->outputFileName);
  mesh = dynamic_cast<apf::MeshSIM*>(meshStruct_->getMesh());
}

Albany::SimOutput::~SimOutput()
{
}

void Albany::SimOutput::writeFile(const double time_value)
{
  (void) time_value;
  assert(commT->getSize() == 1);
  pParMesh spm = mesh->getMesh();
  pMesh sm = PM_mesh(spm, 0);
  std::string meshname;
  {
    std::stringstream ss;
    ss << filename << '_' << index << ".sms";
    meshname = ss.str();
  }
  PM_write(spm, meshname.c_str(), sthreadNone, 0);
  for (int i = 0; i < mesh->countFields(); ++i) {
    apf::Field* field = mesh->getField(i);
    /* this is a cheap filter for non-nodal fields */
    if (apf::getShape(field) != mesh->getShape())
      continue;
    pField sf = apf::getSIMField(field);
    std::string fieldname;
    {
      std::stringstream ss;
      ss << filename << '_' << apf::getName(field) << '_' << index << ".fld";
      fieldname = ss.str();
    }
    Field_write(sf, fieldname.c_str(), 0, 0, 0);
  }
  ++index;
}

void Albany::SimOutput::setFileName(const std::string& fname)
{
  filename = removeExtension(fname);
}
