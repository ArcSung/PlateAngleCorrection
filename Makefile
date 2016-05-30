main: *.cpp
	g++  -std=c++11 -o main *.cpp `pkg-config --cflags --libs opencv`
clean:
	$(RM) *.o main
