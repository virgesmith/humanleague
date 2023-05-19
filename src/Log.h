
#pragma once

#include <string>
#include <iostream>

using namespace std::string_literals;

template<typename T>
std::string to_string_impl(T v)
{
  return std::to_string(v);
}

// print pointer
template<typename T>
std::string to_string_impl(T* p)
{
  constexpr size_t BUF_SIZE = 20;
  static char buf[BUF_SIZE];
  std::snprintf(buf, BUF_SIZE, "0x%016zx", reinterpret_cast<size_t>(p));
  return std::string(buf);
}

template<>
inline std::string to_string_impl(const char* v)
{
  return std::string(v);
}

inline std::string to_string_impl(const std::string& v)
{
  return v;
}

template<typename T>
std::string to_string_impl(const std::vector<T>& v)
{
  if (v.empty())
    return "[]";
  std::string result = "[" + to_string_impl(v[0]);

  for (size_t i = 1; i < v.size(); ++i)
    result += ", " + to_string_impl(v[i]);
  result += "]";

  return result;
}


// need an rvalue ref as might/will be a temporary
template<typename T>
std::string operator%(std::string&& str, T value)
{
  size_t s = str.find("%%");
  if (s != std::string::npos)
  {
    str.replace(s, 2, to_string_impl(value));
  }
  return std::move(str);
}

