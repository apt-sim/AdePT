CXX = c++
NVCC = nvcc

CXXFLAGS = -O2 -fno-exceptions -fno-rtti -flto -Wall -Wextra -pedantic
NVCCFLAGS = -x cu -O2 -fmad false -rdc true
LDFLAGS = -flto

PROG = shower shower-cuda raymarch raymarch-cuda

all: $(PROG)

install: $(PROG)
	install $(PROG) $(HOME)/bin

debug: all
debug: CXXFLAGS += -g
debug: NVCCFLAGS += -g -G

ifeq ($(DEBUG),1)
CXXFLAGS += -g
NVCCFLAGS += -g -G
else
CXXFLAGS += -DNDEBUG
NVCCFLAGS += -DNDEBUG
endif

field.o: field.cc field.h point.h vector.h
init.o: init.cc init.h particle.h point.h vector.h random.h
loop.o: loop.cc loop.h init.h particle.h point.h vector.h transport.h \
 user.h
main.o: main.cc loop.h
particle.o: particle.cc particle.h point.h vector.h
physics.o: physics.cc physics.h particle.h point.h vector.h random.h
ppm.o: ppm.cc ppm.h
primitives.o: primitives.cc primitives.h point.h vector.h
random.o: random.cc random.h vector.h
raymarch.o: raymarch.cc render.h
render.o: render.cc render.h point.h vector.h ppm.h primitives.h \
 transform.h matrix.h quaternion.h
scoring.o: scoring.cc scoring.h particle.h point.h vector.h
transport.o: transport.cc transport.h particle.h point.h vector.h \
 physics.h user.h verlet.h
user.o: user.cc user.h particle.h point.h vector.h
verlet.o: verlet.cc field.h particle.h point.h vector.h

shower: main.o field.o init.o loop.o particle.o physics.o random.o \
 scoring.o transport.o user.o geometry.o primitives.o verlet.o
	$(CXX) $(LDFLAGS) $^ -o $@

shower-cuda: main.cc field.cc init.cc loop.cc particle.cc physics.cc common.cc \
 random.cc scoring.cc transport.cc user.cc geometry.cc primitives.cc verlet.cc
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $^ -o $@

raymarch: raymarch.o ppm.o primitives.o render.o
	$(CXX) $(LDFLAGS) $^ -o $@

rng: random.cc test-rng.cc
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $^ -o $@

raymarch-cuda: raymarch.cc ppm.cc primitives.cc render.cc
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $^ -o $@

test-raymarch: raymarch raymarch-cuda
	raymarch 1024 768 output.ppm
	raymarch-cuda 1024 768 output-cuda.ppm

test-compare: test-raymarch
	# enforce display of difference by forcing status 0 for compare command
	compare -metric rmse output.ppm output-cuda.ppm diff.png || exit 0
	display -geometry +0+0 diff.png

test-rng: rng
	rng 16 1 # size seed

test-shower: shower shower-cuda
	shower 16384
	shower-cuda 16384

.PHONY: test

test: test-rng test-shower test-raymarch

clean:
	$(RM) -f *.o $(PROG) rng output.ppm output-cuda.ppm diff.png

backup: clean
	cd .. ; tar czvf shower-$(shell date +%d-%m-%Y).tar.gz --exclude-vcs shower
