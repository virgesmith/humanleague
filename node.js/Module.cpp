
#include "json_api.h"

#include <node.h>
#include <v8.h>

void init(v8::Handle<v8::Object> target)
{
  NODE_SET_METHOD(target, "sobolSequence", sobolSequence);
  //NODE_SET_METHOD(target, "synthPop", synthPop);
}

NODE_MODULE(humanleague, init) 


