CC = gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra

all: checkmodel

checkmodel: checkmodel.c
	$(CC) $(CFLAGS) -o $@ checkmodel.c

clean:
	rm -f checkmodel

.PHONY: all clean
