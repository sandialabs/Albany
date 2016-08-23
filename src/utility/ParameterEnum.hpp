//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ParameterEnum_hpp)
#define ParameterEnum_hpp

#include <string>
#include <sstream>
#include <initializer_list>
#include <exception>

#include "Teuchos_ParameterList.hpp"

namespace utility
{
	template<typename T>
	class ParameterEnum
	{
	public:
		
		using MapType = std::map<std::string, T>;
		using ValueType = typename MapType::value_type;
		
		ParameterEnum() : m_default(T()) {}
		ParameterEnum(std::string const &name, T def,
									std::initializer_list<ValueType> init);

		T get(std::string const & key) const;
		T get(Teuchos::ParameterList const * p) const;

	private:

		template<typename U>
		friend class BadParameterEnumException;

		MapType			m_map;
		T						m_default;
		std::string	m_name;
	};

	template<typename T>
	class BadParameterEnumException : std::exception
	{
	public:

		explicit BadParameterEnumException(std::string const & key,
																			 ParameterEnum<T> const & e);

		virtual const char *what() const noexcept { return m_msg.c_str(); }

	private:

		std::string m_msg;
	};
}

template<typename T>
utility::ParameterEnum<T>::ParameterEnum(std::string const &name, T def,
																				 std::initializer_list<ValueType> init)
	: m_map(init), m_default(def), m_name(name)
{
	
}

template<typename T>
T
utility::ParameterEnum<T>::get(std::string const & key) const
{
	auto pos = m_map.find(key);

	if (std::end(m_map) == pos)
		throw BadParameterEnumException<T>(key, *this);

	return pos->second;
}

template<typename T>
T
utility::ParameterEnum<T>::get(Teuchos::ParameterList const * p) const
{
	if (!p->isParameter(m_name))
		return m_default;

	return get(p->get<std::string>(m_name));
}

template<typename T>
utility::BadParameterEnumException<T>::BadParameterEnumException(std::string const &key,
																																 ParameterEnum<T> const &e)
{
	std::stringstream ss;
	ss << "\n**** Bad Parameter Enum: invalid value \"" << key << "\" for enum \""
		 << e.m_name << "\". Must be one of the following:\n";

	for (auto &&iter : e.m_map)
		ss << "\t" << iter.first << "\n";

	m_msg = ss.str();
}

#endif

