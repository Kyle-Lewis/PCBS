CU_FILES := $(wildcard src/*.cu)
CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)) $(notdir $(CPP_FILES:.cpp=.o)))
$(info OBJ_FILES is $(OBJ_FILES))
OBJDIR = obj
INCL = common/inc
LIBS = -lcusolver

CC = g++
CFLAGS = --std=c++11

NVCC = nvcc
NVFLAGS = -arch=sm_30 -g --std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D__STRING_ANSI__

run: $(OBJ_FILES)
	$(NVCC) $(NVFLAGS) -o $@ $(OBJ_FILES) $(LIBS) -I$(INCL) -lpng

obj/main.o: src/main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS) -I$(INCL)

obj/PCBS.o: src/PCBS.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS) -I$(INCL)

