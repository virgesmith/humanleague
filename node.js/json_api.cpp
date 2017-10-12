
#include "json_api.h"

#include "humanleague/src/Sobol.h"
#include "humanleague/src/QIWS.h"
#include "humanleague/src/RQIWS.h"
#include "humanleague/src/CQIWS.h"

#include <node.h>
#include <v8.h>

void sobolSequence(const v8::FunctionCallbackInfo<v8::Value>& args)
{
  v8::Isolate* isolate = v8::Isolate::GetCurrent();
  JSON response;
    
  try
  {   
    std::string jsonStringInput = *v8::String::Utf8Value(args[0]->ToString());
    //std::cout << "JSON: " << jsonStringInput << std::endl;
    
    const JSON& request = JSON::parse(jsonStringInput);
    
    if (!request.is_object())
      throw std::runtime_error("JSON request should be an object");
  
    size_t dim = request.at("dim");
    size_t length = request.at("length");
    size_t skips = 0;
    //std::cout << dim << ", " << length << std::endl;
    
    Sobol sobol(dim, skips);
    double scale = 0.5 / (1u << 31);
    std::vector<std::vector<double>> seq;
    seq.reserve(length);
    for (size_t i = 0; i < length; ++i)
    {
      std::vector<double> v(dim);
      const std::vector<uint32_t>& b = sobol.buf();
      for (size_t j = 0; j < dim; ++j)
      {
        v[j] = scale * b[j];
      }
      seq.push_back(v);
    }

    response = seq; // nice!!
  }
  catch(const std::exception& e)
  {
    response["fatal error"] = e.what();
  }
  catch(...)
  {
    response["fatal error"] = "unhandled exception";
  } 
  args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, response.dump().c_str()));   
}


void synthPop(const v8::FunctionCallbackInfo<v8::Value>& args)
{
  v8::Isolate* isolate = v8::Isolate::GetCurrent();
  JSON response;
    
  try
  {   
    std::string jsonStringInput = *v8::String::Utf8Value(args[0]->ToString());
    //std::cout << "JSON: " << jsonStringInput << std::endl;
    
    const JSON& request = JSON::parse(jsonStringInput);
    
    if (!request.is_object())
      throw std::runtime_error("JSON request should be an object");

    std::vector<std::vector<uint32_t>> marginals = request.at("marginals"); 
    
    // TODO multi dim
    QIWS<2> qiws(marginals);

    response["conv"] = qiws.solve();
    // returning as list
    //response["result"] = qiws.result();
    response["list"] = listify(qiws.population(), qiws.result());
    response["p-value"] = qiws.pValue().first;
    response["chiSq"] = qiws.chiSq();
    response["pop"] = qiws.population();
  }
  catch(const std::exception& e)
  {
    response["fatal error"] = e.what();
  }
  catch(...)
  {
    response["fatal error"] = "unhandled exception";
  } 
  args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, response.dump().c_str()));   
}


void synthPopC(const v8::FunctionCallbackInfo<v8::Value>& args)
{
  v8::Isolate* isolate = v8::Isolate::GetCurrent();
  JSON response;
    
  try
  {   
    std::string jsonStringInput = *v8::String::Utf8Value(args[0]->ToString());
    //std::cout << "JSON: " << jsonStringInput << std::endl;
    
    const JSON& request = JSON::parse(jsonStringInput);
    
    if (!request.is_object())
      throw std::runtime_error("JSON request should be an object");

    std::vector<std::vector<uint32_t>> marginals = request.at("marginals"); 
    const std::vector<std::vector<bool>>& permitted = request.at("permitted");

    size_t size[2] = { permitted.size(), permitted[0].size() };
    size_t idx[2] = { permitted.size(), permitted[0].size() };
    
    if (size[0] != marginals[0].size() || size[1] != marginals[1].size())
      throw std::runtime_error("permitted states matrix size inconsistent with marginal sizes");
    
    NDArray<2, bool> permittedStates(idx);
    
    for (idx[0] = 0; idx[0] < size[0]; ++idx[0])
      for (idx[1] = 0; idx[1] < size[1]; ++idx[1])
        permittedStates[idx] = permitted[idx[0]][idx[1]];
        
    CQIWS cqiws(marginals, permittedStates);

    response["conv"] = cqiws.solve();
    // returning as list
    //response["result"] = rqiws.result();
    response["list"] = listify(cqiws.population(), cqiws.result());
    response["pop"] = cqiws.population();
  }
  catch(const std::exception& e)
  {
    response["fatal error"] = e.what();
  }
  catch(...)
  {
    response["fatal error"] = "unhandled exception";
  } 
  args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, response.dump().c_str()));   
}


void synthPopR(const v8::FunctionCallbackInfo<v8::Value>& args)
{
  v8::Isolate* isolate = v8::Isolate::GetCurrent();
  JSON response;
    
  try
  {   
    std::string jsonStringInput = *v8::String::Utf8Value(args[0]->ToString());
    //std::cout << "JSON: " << jsonStringInput << std::endl;
    
    const JSON& request = JSON::parse(jsonStringInput);
    
    if (!request.is_object())
      throw std::runtime_error("JSON request should be an object");

    std::vector<std::vector<uint32_t>> marginals = request.at("marginals"); 
    double rho = request.at("rho");
    
    // TODO multi dim
    RQIWS rqiws(marginals, rho);

    response["conv"] = rqiws.solve();
    // returning as list
    //response["result"] = rqiws.result();
    response["list"] = listify(rqiws.population(), rqiws.result());
    response["pop"] = rqiws.population();
  }
  catch(const std::exception& e)
  {
    response["fatal error"] = e.what();
  }
  catch(...)
  {
    response["fatal error"] = "unhandled exception";
  } 
  args.GetReturnValue().Set(v8::String::NewFromUtf8(isolate, response.dump().c_str()));   
}




