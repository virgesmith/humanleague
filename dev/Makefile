#PROJECT:=humanleague_dev

CXX=g++
CXXFLAGS=-Wall -Werror -g -O0 -std=c++11 -I.. 
#-D_GLIBCXX_DEBUG
LDFLAGS=

# TODO header deps

# 32 bit build purely to test for LLP64 data model issues (i.e. windows)
# but this doesnt test python and R interfaces
all: humanleague_dev humanleague_dev32

humanleague_dev: main.cpp ../src/NDArrayUtils.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

humanleague_dev32: main.cpp ../src/NDArrayUtils.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp
	$(CXX) $(CXXFLAGS) -m32 -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

clean:
	rm $(PROJECT)

.PHONY: all clean install

