// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

type sccFunc func(f *Func) [][]*Block

// verifySccPartition verifies that the kernel DAG of an SCC are topologically sorted
// and each SCC contains the proper set of nodes.
func verifySccPartition(t *testing.T, fut fun, sccFn sccFunc, sccPartition [][]string) {
	blockNames := map[*Block]string{}
	for n, b := range fut.blocks {
		blockNames[b] = n
	}

	calcScc := sccFn(fut.f)
	expectedNscc := len(sccPartition)
	actualNscc := len(calcScc)
	if actualNscc != expectedNscc {
		t.Errorf("expected %d SCC kernels, found %d", expectedNscc, actualNscc)
	}

	for n, scc := range calcScc {
		expectedScc := sccPartition[n]
		nc := len(scc)
		if nc != len(expectedScc) {
			t.Errorf("expected %v nodes in SCC, found %v", expectedScc, scc)
		}

		// verify that nodes in this SCC are as expected
		nodeSet := make(map[string]bool)
		for _, b := range scc {
			nodeSet[blockNames[b]] = false
		}

		for _, expectedNode := range expectedScc {
			if val, ok := nodeSet[expectedNode]; !ok {
				nodes := make([]string, 0, len(nodeSet))
				for k := range nodeSet {
					nodes = append(nodes, k)
				}
				t.Errorf("expected node %s in %v", expectedNode, nodes)
			} else if val {
				t.Errorf("duplicate expected node %s is invalid", expectedNode)
			}

			nodeSet[expectedNode] = true
		}

		// test if any actual SCC nodes are not as expected
		for k, v := range nodeSet {
			if !v {
				t.Errorf("actual block %s in SCC is unexpected", k)
			}
		}
	}
}

func TestSccPartitionLinear(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Goto("1")),
		Bloc("1",
			Goto("2")),
		Bloc("2",
			Goto("3")),
		Bloc("3",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"1"},
		{"2"},
		{"3"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSccPartitionOneLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			If("p", "a", "b")),
		Bloc("a",
			Goto("c")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			If("p", "b", "exit")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"a"},
		{"b", "c"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSccPartitionInfiniteLoop(t *testing.T) {
	c := testConfig(t)
	// note lack of an exit block
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("a")),
		Bloc("a",
			Goto("b")),
		Bloc("b",
			Goto("a")),
	)

	expectedSccPartition := [][]string{
		{"entry"},
		{"b", "a"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSccPartitionDeadCode(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 0, nil),
			If("p", "b3", "b5")),
		Bloc("b2", Exit("mem")),
		Bloc("b3", Goto("b2")),
		Bloc("b4", Goto("b2")),
		Bloc("b5", Goto("b2")))

	// for this example, the topological order is not unique:
	// order of b3 and b5 is interchangeable
	expectedSccPartition := [][]string{
		{"entry"},
		{"b5"},
		{"b3"},
		{"b2"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSccPartitionTricky(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			If("p", "6", "8")),
		Bloc("1",
			If("p", "exit", "5")),
		Bloc("2",
			If("p", "1", "3")),
		Bloc("3",
			If("p", "5", "2")),
		Bloc("4",
			If("p", "2", "3")),
		Bloc("5",
			Goto("4")),
		Bloc("6",
			If("p", "7", "4")),
		Bloc("7",
			If("p", "6", "8")),
		Bloc("8",
			If("p", "10", "9")),
		Bloc("9",
			Goto("11")),
		Bloc("10",
			If("p", "11", "4")),
		Bloc("11",
			Goto("8")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"6", "7"},
		{"8", "9", "10", "11"},
		{"1", "2", "3", "4", "5"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSCCsEarlyExit(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Goto("1")),
		Bloc("1", Goto("2")),
		Bloc("2", Goto("exit")),
		Bloc("exit", Exit("mem")))

	CheckFunc(fun.f)

	count := 0
	for scc := range fun.f.SCCs() {
		count++
		if len(scc) == 1 && scc[0] == fun.blocks["1"] {
			break
		}
	}

	if count != 2 { // entry, then "1"
		t.Errorf("expected to stop after 2 SCCs, got %d", count)
	}
}
