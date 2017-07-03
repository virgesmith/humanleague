#pragma once


#include "3rdParty/json/src/json.hpp"

using JSON = nlohmann::json;

namespace v8 
{
  class Value;
  template<typename T> class FunctionCallbackInfo;
}


//// conversion of scalar types from JSON to T (falling back on converting via a string using function provided)
//// (JSON values from url query params will always be strings)
//template<typename T> 
//T convertWithFallback(const JSON& json, std::function<T(const std::string&)> func)
//{
//  if (json.is_string())
//  {
//    return func(json.get<std::string>());
//  }
//  return json.get<T>();
//}

void sobolSequence(const v8::FunctionCallbackInfo<v8::Value>& args);
void synthPop(const v8::FunctionCallbackInfo<v8::Value>& args);
void synthPopR(const v8::FunctionCallbackInfo<v8::Value>& args);

