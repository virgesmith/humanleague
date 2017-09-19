#PROJECT:=humanleague_dev

src = main.cpp ../src/NDArrayUtils.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp
obj = $(src:.cpp=.o)
dep = $(obj:.o=.d)

#$(info src: $(src))
#$(info obj: $(obj))

CXX=g++
# use -m32 to test for LLP64 data model issues (i.e. windows)
CXXFLAGS=-Wall -Werror -g -O0 -std=c++11 -I.. -D_GLIBCXX_DEBUG
LDFLAGS=

# TODO header deps

# but this doesnt test python and R interfaces
all: humanleague_dev 

#humanleague_dev: main.cpp ../src/NDArrayUtils.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp
humanleague_dev: $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

#humanleague_dev32: main.cpp ../src/NDArrayUtils.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp
#	$(CXX) $(CXXFLAGS) -m32 -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

-include $(dep)


clean:
	rm $(PROJECT) $(obj) $(dep)

.PHONY: all clean install

