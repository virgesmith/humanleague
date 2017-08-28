#PROJECT:=humanleague_dev

CXX=g++
CXXFLAGS=-Wall -Werror -g -std=c++11 -I.. 
LDFLAGS=

# TODO header deps?

humanleague_dev: main.cpp ../src/IPF.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm $(PROJECT)

.PHONY: all clean install

