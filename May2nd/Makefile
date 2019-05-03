INCLUDE_DIRS = 
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= 
CPPLIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -lsqlite3 -lpthread

HFILES= 
CFILES= 
CPPFILES= smartcart.cpp

SRCS= ${HFILES} ${CFILES}
CPPOBJS= ${CPPFILES:.cpp=.o}

all:	smartcart

clean:	
	rm smartcart

distclean:
	-rm -f *.o *.d

smartcart: smartcart.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv` $(CPPLIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CC) $(CFLAGS) -c $<

