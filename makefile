CC=gcc
CLANG=clang
#CFLAGS=-O3 -march=native -fomit-frame-pointer -Wall -Wextra -msse2 -DDSFMT_MEXP=2203 -DHAVE_SSE2
CFLAGS=-O3 -march=native -fomit-frame-pointer -Wall -Wextra -DDSFMT_MEXP=2203
CLIBS=-I $(HOME)/include -L $(HOME)/lib -lutils -lcmdl -lm
EXEC=delaunay
EXEC_CLANG=delaunay-clang
INSTALL=install -m 111
BINDIR=$(HOME)/bin/
GIT=git

all: main.c dSFMT.c
	$(CC) $(CFLAGS) main.c dSFMT.c $(CLIBS) -o $(EXEC)
	#$(CLANG) $(CFLAGS) main.c dSFMT.c $(CLIBS) -o $(EXEC_CLANG)
	#$(GIT) commit -a -m "voronoi_build"`$(GIT) rev-list --count HEAD`

git:
	$(GIT) commit -a -m "voronoi_build"`$(GIT) rev-list --count HEAD`

install:
	$(INSTALL) $(EXEC) $(BINDIR)
