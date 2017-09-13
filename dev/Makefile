#PROJECT:=humanleague_dev

CXX=g++
CXXFLAGS=-Wall -Werror -g -std=c++11 -I.. 
LDFLAGS=

# TODO header deps?

humanleague_dev: main.cpp ../src/NDArrayUtils.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

clean:
	rm $(PROJECT)

.PHONY: all clean install

