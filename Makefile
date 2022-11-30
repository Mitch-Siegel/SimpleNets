CC = g++
FLAGS = -Og -g -Werror -Wall
programs: nn


SRCS = $(basename $(wildcard *.cpp))
OBJS = $(SRCS:%=%.o)

nn: $(OBJS)
	$(CC) $(FLAGS) -o $@ $^

%.o: %.cpp
	$(CC) -c $(FLAGS) $@ $^

test:
	$(info OBJS="$(OBJS)")

clean:
	rm *.o
