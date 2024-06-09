CXXFLAGS = -Iinclude -Icsrc -fPIC
LIBS = -lstdc++ -lm

test: test.o capi.o noise.o
	$(CC) $^ -o $@ $(LIBS)

noise.o: csrc/noise.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o $@
test.o: test.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o $@
capi.o: csrc/capi.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o $@
