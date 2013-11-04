//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Catalyst_TeuchosArrayRCPDataArrayDecl.hpp"

#include "Shards_BasicTopologies.hpp"

#include "Teuchos_TestForException.hpp"

#include "vtkIdList.h"
#include "vtkIdTypeArray.h"
#include "vtkObjectFactory.h"
#include "vtkVariantCast.h"

#include <cassert>
#include <exception>

namespace Albany {
namespace Catalyst {

// Can't use vtkStandardNewMacro on a templated class:
template <typename Scalar>
TeuchosArrayRCPDataArray<Scalar> * TeuchosArrayRCPDataArray<Scalar>::New()
{
  VTK_STANDARD_NEW_BODY(TeuchosArrayRCPDataArray<Scalar>)
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::PrintSelf(ostream &os, vtkIndent indent)
{
  this->TeuchosArrayRCPDataArray<Scalar>::Superclass::PrintSelf(os, indent);
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
SetArrayRCP(const Teuchos::ArrayRCP<Scalar> &array, int numComps)
{
  this->Data = array;
  this->NumberOfComponents = numComps;
  delete [] this->TmpArray;
  this->TmpArray = new Scalar[numComps];
  this->Size = static_cast<vtkIdType>(array.size());
  this->MaxId = this->Size - 1;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::Initialize()
{
  this->Data = Teuchos::ArrayRCP<Scalar>();
  this->Size = 0;
  this->MaxId = -1;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::GetTuples(vtkIdList *ptIds,
                                                 vtkAbstractArray *output)
{
  vtkDataArray *da = vtkDataArray::FastDownCast(output);
  TEUCHOS_TEST_FOR_EXCEPTION(!da, std::runtime_error,
                             "GetTuples must be given a vtkDataArray arg.");

  da->Reset();
  da->SetNumberOfComponents(this->NumberOfComponents);
  da->Allocate(this->NumberOfComponents * ptIds->GetNumberOfIds());

  vtkIdType *begin = ptIds->GetPointer(0);
  vtkIdType *end = ptIds->GetPointer(ptIds->GetNumberOfIds());

  // Use type appropriate API if possible:
  if (vtkTypedDataArray<Scalar> *ta
      = vtkTypedDataArray<Scalar>::FastDownCast(da)) {
    while (begin != end) {
      ta->InsertNextTupleValue(
            &this->Data[*(begin++) * this->NumberOfComponents]);
    }
  }
  else { // otherwise, use the double interface:
    double *tuple = new double[this->NumberOfComponents];
    while (begin != end) {
      this->GetTuple(*(begin++), tuple);
      da->InsertNextTuple(tuple);
    }
    delete [] tuple;
  }
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::GetTuples(vtkIdType p1, vtkIdType p2,
                                                 vtkAbstractArray *output)
{
  vtkDataArray *da = vtkDataArray::FastDownCast(output);
  TEUCHOS_TEST_FOR_EXCEPTION(!da, std::runtime_error,
                             "GetTuples must be given a vtkDataArray arg.");

  da->Reset();
  da->SetNumberOfComponents(this->NumberOfComponents);
  da->Allocate(this->NumberOfComponents * (p2 - p1 + 1));

  // Use type appropriate API if possible:
  if (vtkTypedDataArray<Scalar> *ta
      = vtkTypedDataArray<Scalar>::FastDownCast(da)) {
    while (p1 <= p2)
      ta->InsertNextTupleValue(&this->Data[p1++ * this->NumberOfComponents]);
  }
  else { // otherwise, use the double interface:
    double *tuple = new double[this->NumberOfComponents];
    while (p1 <= p2) {
      this->GetTuple(p1++, tuple);
      da->InsertNextTuple(tuple);
    }
    delete [] tuple;
  }
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::Squeeze()
{
}

template <typename Scalar>
vtkArrayIterator *TeuchosArrayRCPDataArray<Scalar>::NewIterator()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Method not implemented.");
  return NULL;
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::LookupValue(vtkVariant value)
{
  bool valid = false;
  double tmp = vtkVariantCast<double>(value, &valid);
  if (valid)
    return this->Lookup(tmp, 0);
  return -1;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
LookupValue(vtkVariant value, vtkIdList *ids)
{
  bool valid = false;
  double tmp = vtkVariantCast<double>(value, &valid);
  ids->Reset();
  if (valid) {
    vtkIdType index = 0;
    while ((index = this->Lookup(tmp, index)) >= 0)
      ids->InsertNextId(index++);
  }
}

template <typename Scalar>
vtkVariant TeuchosArrayRCPDataArray<Scalar>::GetVariantValue(vtkIdType idx)
{
  return vtkVariant(this->Data[idx]);
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::ClearLookup()
{
  // no-op, no fast lookup implemented.
}

template <typename Scalar>
double *TeuchosArrayRCPDataArray<Scalar>::GetTuple(vtkIdType i)
{
  int numComps = this->NumberOfComponents;
  double *out = this->TmpArray;
  typename ContainerType::const_iterator in =
      this->Data.begin() + (i * numComps);
  while (numComps-- > 0)
    *(out++) = static_cast<double>(*(in++));

  return this->TmpArray;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::GetTuple(vtkIdType i, double *tuple)
{
  int numComps = this->NumberOfComponents;
  typename ContainerType::const_iterator in
      = this->Data.begin() + (i * numComps);
  while (numComps-- > 0)
    *(tuple++) = static_cast<double>(*(in++));
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::LookupTypedValue(ValueType value)
{
  return this->Lookup(value, 0);
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
LookupTypedValue(ValueType value, vtkIdList *ids)
{
  ids->Reset();
  vtkIdType index = 0;
  while ((index = this->Lookup(value, index)) >= 0)
    ids->InsertNextId(index++);
}

template <typename Scalar>
int TeuchosArrayRCPDataArray<Scalar>::Allocate(vtkIdType sz, vtkIdType ext)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return 0;
}

template <typename Scalar>
int TeuchosArrayRCPDataArray<Scalar>::Resize(vtkIdType numTuples)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return 0;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::SetNumberOfTuples(vtkIdType number)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
SetTuple(vtkIdType i, vtkIdType j, vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::SetTuple(vtkIdType i,
                                                const float *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::SetTuple(vtkIdType i,
                                                const double *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InsertTuple(vtkIdType i, vtkIdType j, vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InsertTuple(vtkIdType i, const float *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InsertTuple(vtkIdType i, const double *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InsertTuples(vtkIdList *dstIds, vtkIdList *srcIds, vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::
InsertNextTuple(vtkIdType j, vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::InsertNextTuple(const float *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::
InsertNextTuple(const double *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::DeepCopy(vtkAbstractArray *aa)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::DeepCopy(vtkDataArray *da)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InterpolateTuple(vtkIdType i, vtkIdList *ptIndices, vtkAbstractArray *source,
                 double *weights)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InterpolateTuple(vtkIdType i, vtkIdType id1, vtkAbstractArray *source1,
                 vtkIdType id2, vtkAbstractArray *source2, double t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
SetVariantValue(vtkIdType idx, vtkVariant value)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::RemoveTuple(vtkIdType id)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::RemoveFirstTuple()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::RemoveLastTuple()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
SetTupleValue(vtkIdType i, const ValueType *t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::
InsertTupleValue(vtkIdType i, const ValueType *t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::
InsertNextTupleValue(const ValueType *t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::SetValue(vtkIdType idx, ValueType value)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::InsertNextValue(ValueType v)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

template <typename Scalar>
void TeuchosArrayRCPDataArray<Scalar>::InsertValue(vtkIdType idx, ValueType v)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

template <typename Scalar>
TeuchosArrayRCPDataArray<Scalar>::TeuchosArrayRCPDataArray()
  : Data(Teuchos::ArrayRCP<Scalar>()),
    TmpArray(NULL)
{
}

template <typename Scalar>
TeuchosArrayRCPDataArray<Scalar>::~TeuchosArrayRCPDataArray()
{
  delete TmpArray;
}

template <typename Scalar>
vtkIdType TeuchosArrayRCPDataArray<Scalar>::
Lookup(double val, vtkIdType index)
{
  while (index <= this->MaxId) {
    if (this->Data[index] == val)
      return index;
    ++index;
  }
  return -1;
}

} // namespace Catalyst
} // namespace Albany
