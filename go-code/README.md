# Code path's

<GOROOT>/go/src/cmd/compile/internal/ssa/scc.go
<GOROOT>/go/src/cmd/compile/internal/ssa/scc_test.go
<GOROOT>/go/src/cmd/compile/internal/ssa/dom.go
<GOROOT>/go/src/cmd/compile/internal/ssa/regalloc.go

# Building the project and testing it
cd <GOROOT>/go/src
./all.bash

# use the ssa debug output 
<GOROOT>/go/bin/go build -gcflags="-d=ssa/regalloc/debug=3" qsort.go

# Testing and measuring compilation (produces an HTML file)
GOSSAFUNC=TestSignMessage <GOROOT>/go/bin/go test crypto_test.go