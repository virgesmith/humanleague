
# pass extra flags, e.g.
# make -B CXXFLAGS="-DVERBOSE -D_GLIBCXX_DEBUG -D_DEBUG"

# make CXX=clang++-4.0 to use clang

target:=humanleague_dev

src = main.cpp ../src/NDArrayUtils.cpp ../src/Index.cpp ../src/Sobol.cpp ../src/SobolImpl.cpp ../src/StatFuncs.cpp \
      ../src/QIS.cpp ../src/QISI.cpp ../src/QIWS.cpp \
			../src/UnitTester.cpp ../src/TestNDArray.cpp ../src/TestIndex.cpp ../src/TestReduce.cpp ../src/TestSlice.cpp \
			../src/TestSobol.cpp ../src/TestStatFuncs.cpp ../src/TestQIWS.cpp

obj = $(src:.cpp=.o)
dep = $(obj:.o=.d)

#$(info src: $(src))
#$(info obj: $(obj))

CXX=g++
# use -m32 to test for LLP64 data model issues (i.e. windows)
# See about override here: https://stackoverflow.com/questions/2129391/append-to-gnu-make-variables-via-command-line
override CXXFLAGS+=-Wall -Werror -pthread -fPIC -g -std=c++11 -I..
# Note: this arg must come AFTER the object files
LDFLAGS=-pthread

# this doesnt test python and R interfaces!!!
all: $(target) 

$(target): $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

# These rules build obj from src/h
-include $(dep)

clean:
	rm -f $(target) $(obj) $(dep)

.PHONY: all clean

