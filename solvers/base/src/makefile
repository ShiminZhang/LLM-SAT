all:
	$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build"
kissat:
	$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build" kissat
tissat:
	$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build" tissat
clean:
	rm -f "/home/meru/scratch/LLM-SAT/solvers/base"/makefile
	rm -f "/home/meru/scratch/LLM-SAT/solvers/base"/src/makefile
	-$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build" clean
	rm -rf "/home/meru/scratch/LLM-SAT/solvers/base/build"
coverage:
	$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build" coverage
format:
	$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build" format
test:
	$(MAKE) -C "/home/meru/scratch/LLM-SAT/solvers/base/build" test
.PHONY: all clean coverage format kissat test tissat
