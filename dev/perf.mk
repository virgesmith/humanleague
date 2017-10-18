PROJECT:=humanleague_perf

src = perf.cpp ../src/NDArrayUtils.cpp ../src/Index.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp ../src/StatFuncs.cpp ../src/IPF.cpp ../src/QSIPF.cpp ../src/QIWS.cpp
obj = $(src:.cpp=.o)
dep = $(obj:.o=.d)

#$(info src: $(src))
#$(info obj: $(obj))

CXX=g++
# use -m32 to test for LLP64 data model issues (i.e. windows)
CXXFLAGS=-Wall -Werror -g -O2 -std=c++11 -I..
LDFLAGS=

# this doesnt test python and R interfaces!!!
all: humanleague_perf 

humanleague_perf: $(obj)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

# These rules build obj from src/h
-include $(dep)

clean:
	rm -f $(PROJECT) $(obj) $(dep)

.PHONY: all clean

