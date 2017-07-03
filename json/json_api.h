#pragma once


#include "3rdParty/json/src/json.hpp"

using JSON = nlohmann::json;

namespace v8 
{
  class Value;
  template<typename T> class FunctionCallbackInfo;
}

void sobolSequence(const v8::FunctionCallbackInfo<v8::Value>& args);
void synthPop(const v8::FunctionCallbackInfo<v8::Value>& args);
void synthPopR(const v8::FunctionCallbackInfo<v8::Value>& args);

