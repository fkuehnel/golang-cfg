# Live variable analysis with GO toolchain

The analysis script contains *.py scripts to run
```python
python3 <script>.py SCC-stats
```

And for live variable debugging:
```bash
cd $GOROOT/src/math/big
GOSSAFUNC='(*Rat).FloatPrec' go test -run=^$ -count=1 -gcflags="-d=ssa/regalloc/debug=3" 2>debug_scc_3pass.txt
```

then a comparison
```python
python3 compare.py debug_master.txt debug_scc_3pass.txt
```

The go-code folder has the relevant code snippets
that implement the tiered algorithm suggested in the publication.