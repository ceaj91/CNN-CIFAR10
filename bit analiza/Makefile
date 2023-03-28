# SystemC installation directory
SYSTEMC_HOME=/usr/local/systemc

# Compiler to use
CC=g++

# Compiler flags
CFLAGS=-c -Wall -I$(SYSTEMC_HOME)/include

# Linker flags
LDFLAGS=-L$(SYSTEMC_HOME)/lib-linux64 -lsystemc

# Source files
SOURCES=testing19bits.cpp ConvLayer.cpp ../cpp_implementation/denselayer.cpp ../cpp_implementation/flattenlayer.cpp ../cpp_implementation/MaxPoolLayer.cpp

# Object files
OBJECTS=$(SOURCES:.cpp=.o)

# Executable name
EXECUTABLE=testing19bits

# Default target
all: $(SOURCES) $(EXECUTABLE)

# Build target
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# Compile target
.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

# Clean target
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
