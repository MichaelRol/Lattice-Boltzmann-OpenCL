# Makefile

EXE=d2q9-bgk

CC=icc
CFLAGS= -std=c99 -Wall -Ofast -xAVX
LIBS = -lm

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS += -framework OpenCL
else
	LIBS += -lOpenCL
endif

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check1:
	python check/check.py --ref-av-vels-file=check/128x128.av_vels.dat --ref-final-state-file=check/128x128.final_state.dat --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check2:
	python check/check.py --ref-av-vels-file=check/256x256.av_vels.dat --ref-final-state-file=check/256x256.final_state.dat --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check3:
	python check/check.py --ref-av-vels-file=check/1024x1024.av_vels.dat --ref-final-state-file=check/1024x1024.final_state.dat --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

check4:
	python check/check.py --ref-av-vels-file=check/128x256.av_vels.dat --ref-final-state-file=check/128x256.final_state.dat --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)


.PHONY: all check clean

clean:
	rm -f $(EXE)
