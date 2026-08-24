# Local development work

## Building the project and testing it
```bash
export PATH=$PWD/go/bin:$PATH
export GOROOT=$PWD/go
cd go/src
./all.bash
```

## Code paths
```bash
func.go, scc.g0, scc_test.go, regalloc.go, regallog_bench_test.go

$GOROOT/src/cmd/compile/internal/ssa/
```

## Using the ssa debug output 
```bash
go build -gcflags="-d=ssa/regalloc/debug=3" -count=1 qsort.go >gsort_dump.txt
```
The `-count=1` disables the cache.

# Performance Testing

It's nice that there is an auxiliary PDF with many details, but please write a more complete commit message. See https://go.dev/doc/contribute#commit_messages

As stated there, the commit message should elaborate and provide context for the change and explain what it does.

Having a more complete commit message also makes it easier for a reviewer to pick this up to review the code.

Also, the commit message for a change like this really should have at least some quantitative performance old vs. new results (e.g., comparing wall clock execution times or similar), which I did not immediately see in the PDF.

As noted in the Contribution Guide, the benchstat tool is conventionally used to format benchmark data for commit messages. In some cases, compilebench can be used.

For your particular case, I'm not sure the magnitude of the changes will be picked up by compilebench. Maybe you've already done something like this, but some chance you might need to instrument the code to report useful timing information, in which case you could likely feed that old vs. new timing information into benchstat.

Some links that might be helpful:

https://pkg.go.dev/golang.org/x/perf/cmd/benchstat
https://pkg.go.dev/golang.org/x/tools/cmd/compilebench
https://github.com/golang/go/blob/master/src/cmd/compile/README.md#8-tips
Finally, this is based on just a quick look at the PDF. Sorry if I've missed something, and sorry if this is all things you already know.

## Comparing master versus my branch compiler

```bash
export PATH=$PWD/go/bin:$PATH
cd go/src
git checkout master
./all.bash
toolstash save  # this has the master toolchain
git checkout scc  # working in the scc branch
./all.bash


compilebench -count 20 -run "^Benchmark(?:Compiler|SSA)$" >mybench_new.txt
cd $GOROOT/src/cmd/compile/internal/ssa
go test -bench=BenchmarkComputeLive -count=6 >live_scc_scc.txt  # use new compiler

toolstash restore                                   # restore old compiler

compilebench -count 20 -run "^Benchmark(?:Compiler|SSA)$" >mybench_old.txt
cd $GOROOT/src/cmd/compile/internal/ssa
go test -bench=BenchmarkComputeLive -count=10 >live_master.txt  # use old compiler

benchstat live_master_master.txt live_scc_scc.txt
```

And continue building the new code basis for the new compiler.

## Generating SSA/CFG output for specific tests, functions: (produces an HTML file)

The count argument avoids using cached results.
```bash
cd $GOROOT/src/sort
GOSSAFUNC=partialInsertionSort_func go test -run=^$ -count=1
cd "$GOROOT/src/math/big"
GOSSAFUNC='(*Rat).FloatPrec' go test -run=^$ -count=1
```

Test code quality (master <-> scc) for loopy code:
```bash
cd $GOROOT/src/math/big
GOSSAFUNC='(*Rat).FloatPrec' go test -run=^$ -count=1 -gcflags="-d=ssa/regalloc/debug=3" 2>debug_scc_dfs_inf.txt
GOSSAFUNC='(*Rat).FloatPrec' go test  -count=1 -run=^$
```

And a test from the GO toolchain:
```bash
cd $GOROOT/src/cmd/compile/internal/ssa
go test -run TestSCC -count=1 
go test -run TestSCC -count=1 -gcflags="-d=ssa/build/dump" 2>scc_ssa.txt
go test -run TestLiveControlOps -count=1 -gcflags="-d=ssa/likelyadjust/debug=4"
```