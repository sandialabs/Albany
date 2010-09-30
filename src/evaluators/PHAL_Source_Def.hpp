/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include <cmath>
#include <sstream>
#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_VerboseObject.hpp"

namespace Source_Functions {

template <typename EvalT, typename Traits>
class Source_Base {
public :
  Source_Base(){};
  virtual ~Source_Base(){};
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)                     = 0;
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)                     = 0;
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm) = 0;
  virtual void evaluateFields (typename Traits::EvalData workset)                                         = 0;
};


template<typename EvalT, typename Traits>
class Quadratic : public Source_Base<EvalT,Traits>, public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  Quadratic(Teuchos::ParameterList& p);
  virtual ~Quadratic(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n) { return m_factor;};
private :
  ScalarT     m_factor;
  double      m_constant;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_baseField;
};

template<typename EvalT,typename Traits>
bool Quadratic<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Quadratic");
  return exists;
}

template<typename EvalT,typename Traits>
Quadratic<EvalT,Traits>::Quadratic(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Quadratic");
  m_factor   = paramList.get("Nonlinear Factor", 0.0);
  m_constant = paramList.get("Constant", false);
  // Add the factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  new Sacado::ParameterRegistration<EvalT, SPL_Traits> ("Quadratic Nonlinear Factor", this, paramLib);
}

template<typename EvalT,typename Traits>
void Quadratic<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::ParameterList& paramList = m_source_list->sublist("Quadratic");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}
template<typename EvalT,typename Traits>
void Quadratic<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::ParameterList& paramList = m_source_list->sublist("Quadratic");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Variable Name"), dl);
  m_baseField = f;
  source.addDependentField(m_baseField);
}
template<typename EvalT,typename Traits>
void Quadratic<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_baseField,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void Quadratic<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset){
  int numCells = workset.numCells;

  // Loop over cells, quad points: compute Quadratic Source Term
  if (m_constant) {
    for (std::size_t cell = 0; cell < numCells; ++cell) {
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
        m_source(cell, iqp) = m_factor;
    }
  } else {
    for (std::size_t cell = 0; cell < numCells; ++cell) {
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
        m_source(cell, iqp) = m_factor * m_baseField(cell,iqp) * m_baseField(cell,iqp);
    }
  }
}

template<typename EvalT, typename Traits>
class MVQuadratic : public Source_Base<EvalT,Traits>, public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  MVQuadratic(Teuchos::ParameterList& p);
  virtual ~MVQuadratic(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n);
private :
  std::vector<ScalarT>     m_factor;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_baseField;
};

template<typename EvalT,typename Traits>
bool MVQuadratic<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Multivariate Quadratic");
  return exists;
}

template<typename EvalT,typename Traits>
MVQuadratic<EvalT,Traits>::MVQuadratic(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Quadratic");
  int num_vars = paramList.get("Dimension", 1);
  m_factor.resize(num_vars);
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  for (int i=0; i<num_vars; i++) {
    std::stringstream ss;
    ss << "Nonlinear Factor " << i;
    m_factor[i] = paramList.get(ss.str(), 0.0);

    // Add the factor as a Sacado-ized parameter
    std::stringstream ss2;
    ss2 << "Multivariate Quadratic Nonlinear Factor " << i;
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss2.str(), this, paramLib);
  }
}

template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Quadratic");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}
template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Quadratic");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Variable Name"), dl);
  m_baseField = f;
  source.addDependentField(m_baseField);
}
template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_baseField,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset){
  int numCells = workset.numCells;

  ScalarT a = 0.0;
  for (unsigned int j=0; j<m_factor.size(); j++) {
    a += m_factor[j];
  }
  a /= static_cast<double>(m_factor.size());

  // Loop over cells, quad points: compute MVQuadratic Source Term
  for (std::size_t cell = 0; cell < numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
      m_source(cell, iqp) = a * m_baseField(cell,iqp) * m_baseField(cell,iqp);
  }
}

template<typename EvalT,typename Traits>
typename MVQuadratic<EvalT,Traits>::ScalarT& 
MVQuadratic<EvalT,Traits>::getValue(const std::string &n)
{
  for (unsigned int i=0; i<m_factor.size(); i++) {
    std::stringstream ss;
    ss << "Multivariate Quadratic Nonlinear Factor " << i;
    if (n == ss.str())
      return m_factor[i];
  }
  return m_factor[0];
}

template<typename EvalT, typename Traits>
class MVExponential : public Source_Base<EvalT,Traits>, public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  MVExponential(Teuchos::ParameterList& p);
  virtual ~MVExponential(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n);
private :
  std::vector<ScalarT>     m_factor;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_baseField;
};

template<typename EvalT,typename Traits>
bool MVExponential<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Multivariate Exponential");
  return exists;
}

template<typename EvalT,typename Traits>
MVExponential<EvalT,Traits>::MVExponential(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Exponential");
  int num_vars = paramList.get("Dimension", 1);
  m_factor.resize(num_vars);
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  for (int i=0; i<num_vars; i++) {
    std::stringstream ss;
    ss << "Nonlinear Factor " << i;
    m_factor[i] = paramList.get(ss.str(), 0.0);

    // Add the factor as a Sacado-ized parameter
    std::stringstream ss2;
    ss2 << "Multivariate Exponential Nonlinear Factor " << i;
    new Sacado::ParameterRegistration<EvalT, SPL_Traits> (ss2.str(), this, paramLib);
  }
}

template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Exponential");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}
template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Exponential");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Variable Name"), dl);
  m_baseField = f;
  source.addDependentField(m_baseField);
}
template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_baseField,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset){
  int numCells = workset.numCells;

  ScalarT a = 0.0;
  for (unsigned int j=0; j<m_factor.size(); j++) {
    a += m_factor[j];
  }
  a /= static_cast<double>(m_factor.size());

  // Loop over cells, quad points: compute MVExponential Source Term
  for (std::size_t cell = 0; cell < numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
      m_source(cell, iqp) = a*std::exp(m_baseField(cell,iqp));
  }
}

template<typename EvalT,typename Traits>
typename MVExponential<EvalT,Traits>::ScalarT& 
MVExponential<EvalT,Traits>::getValue(const std::string &n)
{
  for (unsigned int i=0; i<m_factor.size(); i++) {
    std::stringstream ss;
    ss << "Multivariate Exponential Nonlinear Factor " << i;
    if (n == ss.str())
      return m_factor[i];
  }
  return m_factor[0];
}


// This needs to be templated to get mesh derivatives
template<typename EvalT>
class Spatial_Base {
public :
  virtual ~Spatial_Base(){}
  typedef typename EvalT::MeshScalarT MeshScalarT;
  virtual MeshScalarT evaluateFields  (const std::vector<MeshScalarT> &coords)=0;
};

template<typename EvalT>
class Gaussian : public Spatial_Base<EvalT> {
public :
  static bool check_for_existance(Teuchos::ParameterList &source_list);
  typedef typename EvalT::MeshScalarT MeshScalarT;
  Gaussian(Teuchos::ParameterList &source_list, std::size_t num);
  virtual ~Gaussian(){}
  virtual MeshScalarT evaluateFields(const std::vector<MeshScalarT> &coords);
private :
  double      m_amplitude        ;
  double      m_radius           ;
  double      m_sigma_sq         ;
  double      m_sigma_pi         ;
  Teuchos::Array<double> m_centroid ;
};

template<typename EvalT>
inline bool Gaussian<EvalT>::check_for_existance(Teuchos::ParameterList &source_list)
{
  std::string g("Gaussian");
  bool exists = source_list.getEntryPtr("Type");
  if (exists) exists = g==source_list.get("Type",g);
  return exists;
}

template<typename EvalT>
inline Gaussian<EvalT>::Gaussian(Teuchos::ParameterList &source_list, std::size_t num)
{
  std::stringstream ss;
  ss <<"Center "<<num;
  Teuchos::ParameterList& paramList = source_list.sublist("Spatial",true);
  m_amplitude = paramList.get("Amplitude",      1.0);
  m_radius    = paramList.get("Radius",         1.0);
  m_centroid  = Teuchos::getArrayFromStringParameter<double> (source_list, ss.str());
  m_sigma_sq = 1.0/(2.0*std::pow(m_radius, 2));
  const double pi = 3.1415926535897932385;
  m_sigma_pi = 1.0/(m_radius*std::sqrt(2*pi));
}

template<typename EvalT>
typename EvalT::MeshScalarT Gaussian<EvalT>::
evaluateFields(const std::vector<typename EvalT::MeshScalarT> &coords)
{
  MeshScalarT exponent=0;
  const std::size_t nsd = coords.size();
  for (std::size_t i=0; i<nsd; ++i) {
    exponent += std::pow(m_centroid[i]-coords[i],2);
  }
  exponent *= m_sigma_sq;  
  MeshScalarT x;
  if (nsd==1)
    x = m_amplitude *          m_sigma_pi    * std::exp(-exponent);           
  else if (nsd==2)
    x = m_amplitude * std::pow(m_sigma_pi,2) * std::exp(-exponent);           
  else if (nsd==3)
    x = m_amplitude * std::pow(m_sigma_pi,3) * std::exp(-exponent);           
  return x;
}


class Wavelet_Base {
public :
  virtual ~Wavelet_Base(){}
  virtual RealType evaluateFields  (const RealType time) = 0;
};

class Monotone : public Wavelet_Base {
public :
  static bool check_for_existance(Teuchos::ParameterList &source_list);
  Monotone(Teuchos::ParameterList &source_list);
  virtual ~Monotone(){}
  virtual RealType evaluateFields  (const RealType time);
private :
};

inline bool Monotone::check_for_existance(Teuchos::ParameterList &source_list)
{ 
  std::string g("Monotone");
  bool exists = source_list.getEntryPtr("Type");
  if (exists) exists = g==source_list.get("Type",g);
  return exists;
}

inline Monotone::Monotone(Teuchos::ParameterList &source_list)
{
}

inline RealType Monotone::evaluateFields(const RealType time)
{ return 1.0; }


template<typename EvalT, typename Traits>
class PointSource : public Source_Base<EvalT,Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  PointSource(Teuchos::ParameterList* source_list);
  virtual ~PointSource();
  virtual void EvaluatedFields (Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields (Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData       (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields  (typename Traits::EvalData workset);
private :
  std::size_t                 m_num_qp;
  std::size_t                 m_num_dim;
  Wavelet_Base               *m_wavelet;
  std::vector<Spatial_Base<EvalT> * >  m_spatials;

  Teuchos::ParameterList* m_source_list;
              PHX::MDField<ScalarT,Cell,Point>   m_pressure_source;
  PHX::MDField<MeshScalarT,Cell,Point,Dim> coordVec;
};
template<typename EvalT, typename Traits>
bool PointSource<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Point");
  return exists;
}

template<typename EvalT, typename Traits>
PointSource<EvalT,Traits>::PointSource(Teuchos::ParameterList* source_list) :
  m_num_qp(0), m_wavelet(NULL), m_spatials(), m_source_list(source_list)
{
  Teuchos::ParameterList& paramList = source_list->sublist("Point",true);
  const std::size_t num  = paramList.get("Number", 0);
  Teuchos::ParameterList& spatial_param = paramList.sublist("Spatial",true);
  if (Gaussian<EvalT>::check_for_existance(spatial_param)) {
    for (std::size_t i=0; i<num; ++i) {
      Gaussian<EvalT> *s = new Gaussian<EvalT>(paramList,i);
      m_spatials.push_back(s);
    }
  } else {
    TEST_FOR_EXCEPTION(m_spatials.empty(), std::logic_error,
                       "Point: Did not find a single spatial component.  Specify Gaussian.");
  }
  Teuchos::ParameterList& wavelet_param = paramList.sublist("Time Wavelet",true);
  if (Monotone::check_for_existance(wavelet_param)) {
    Monotone *monotone = new Monotone(wavelet_param);
    m_wavelet = monotone;
  } else {
    TEST_FOR_EXCEPTION(!m_wavelet, std::logic_error,
                       "Point: Did not find a single wavelet component. Specify Monotone.");
  }
}
template<typename EvalT, typename Traits>
PointSource<EvalT,Traits>::~PointSource()
{
  delete m_wavelet;
  for (std::size_t i=0; i<m_spatials.size(); ++i) {
    Spatial_Base<EvalT> *s = m_spatials[i];
    delete s;
  }
  m_spatials.clear();
}

template<typename EvalT, typename Traits>
void PointSource<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> scalar_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f0(p.get<string>("Pressure Source Name"), scalar_qp);
  m_pressure_source       = f0;
  source.addEvaluatedField(m_pressure_source);
}

template<typename EvalT, typename Traits>
void PointSource<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> scalar_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");

  PHX::MDField<MeshScalarT,Cell,Point,Dim> f0
    (p.get<string>("QP Coordinate Vector Name"),  vector_qp);
  coordVec = f0;
  source.addDependentField(coordVec);
}

template<typename EvalT, typename Traits>
void    PointSource<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_pressure_source, fm);
  utils.setFieldData(coordVec,     fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Point,Dim>::size_type > dims;
  coordVec.dimensions(dims);
  m_num_qp  = dims[1];
  m_num_dim = dims[2];
}

template<typename EvalT, typename Traits>
void PointSource<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  const RealType time  = workset.current_time;
  for (std::size_t cell = 0; cell < numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++) {
      std::vector<MeshScalarT> coord;
      for (std::size_t i=0; i<m_num_dim; ++i) {
        const MeshScalarT  x  = coordVec(cell,iqp,i);
        coord.push_back(x);
      }
      ScalarT &p_s = m_pressure_source(cell,iqp);
      p_s = 0;
      const RealType wavelet = m_wavelet->evaluateFields(time);
      for (std::size_t i=0; i<m_spatials.size(); ++i) {
        const MeshScalarT spatial = m_spatials[i]->evaluateFields(coord);
        p_s += wavelet*spatial;
      }
    }
  }
}


}

using namespace Source_Functions;

template<typename EvalT, typename Traits>
Source<EvalT, Traits>::Source(Teuchos::ParameterList& p)
{

  Teuchos::ParameterList* source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  if (Quadratic<EvalT,Traits>::check_for_existance(source_list)) {
    Quadratic<EvalT,Traits>    *q = new Quadratic<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("QuadraticSource");
  }
  if (MVQuadratic<EvalT,Traits>::check_for_existance(source_list)) {
    MVQuadratic<EvalT,Traits> *q = new MVQuadratic<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("MVQuadraticSource");
  }
  if (MVExponential<EvalT,Traits>::check_for_existance(source_list)) {
    MVExponential<EvalT,Traits> *q = new MVExponential<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("MVExponentialSource");
  }
  if (PointSource<EvalT,Traits>::check_for_existance(source_list)) {
    PointSource<EvalT,Traits>       *s = new PointSource<EvalT,Traits>(source_list);
    Source_Base<EvalT,Traits> *sb = s;
    m_sources.push_back(sb);
    this->setName("PointSource");
  }
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    sb->DependentFields(*this, p);
    sb->EvaluatedFields(*this, p);
  }

}

template<typename EvalT, typename Traits>
Source<EvalT, Traits>::~Source() {
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    delete sb;
  }
  m_sources.clear();
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Source<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    sb->FieldData(this->utils, fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Source<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    sb->evaluateFields(workset);
  }
  return;
}

//**********************************************************************
