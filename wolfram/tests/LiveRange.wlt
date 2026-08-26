(* Regression tests for LiveRange`.

   These encode results that previously existed only as saved output in a
   notebook whose kernel state could not be reconstructed.

   Run headless:   wolframscript -file tests/LiveRange.wlt
   Or set LIVERANGE_ROOT to the directory holding LiveRange.wl.
*)

(* ---- locate the package. Several harnesses leave $InputFileName empty, so
        try each candidate and verify LiveRange.wl is actually there. ---- *)
$root = Module[{cands, up},
  up[start_] := Module[{d = start},
    While[StringLength[d] > 1 && ! FileExistsQ[FileNameJoin[{d, "LiveRange.wl"}]],
      d = DirectoryName[StringDrop[d, -1]]];
    If[FileExistsQ[FileNameJoin[{d, "LiveRange.wl"}]], d, Nothing]];
  cands = DeleteCases[Flatten[{
     Environment["LIVERANGE_ROOT"],
     If[$InputFileName =!= "", DirectoryName[DirectoryName[$InputFileName]], Nothing],
     If[$InputFileName =!= "", up[DirectoryName[$InputFileName]], Nothing],
     Directory[], up[Directory[]]}], $Failed | Null | ""];
  SelectFirst[cands, FileExistsQ[FileNameJoin[{#, "LiveRange.wl"}]] &, $Failed]];

If[$root === $Failed,
  Print["FATAL: cannot locate LiveRange.wl. Set LIVERANGE_ROOT."]; Quit[1]];

Get[FileNameJoin[{$root, "LiveRange.wl"}]];
corpus[name_] := LiveRange`ImportCFG[FileNameJoin[{$root, "corpus", name}]];

(* ---- GUARD: fail loudly if the corpus did not load. Without this, an empty
        corpus makes FindDifference compare <||> to <||> and every oracle test
        below passes vacuously. ---- *)

VerificationTest[
  VertexCount[corpus["minimal-scc"]["Graph"]] > 0 &&
  VertexCount[corpus["crypto-des"]["Graph"]] > 0,
  True,
  TestID -> "guard/corpus-actually-loaded"];

(* ---------- parsing ---------- *)

VerificationTest[VertexCount[corpus["minimal-scc"]["Graph"]], 8,
  TestID -> "minimal-scc/parses-8-blocks"];

VerificationTest[corpus["minimal-scc"]["Entry"], Inactive[LiveRange`Blk][4],
  TestID -> "minimal-scc/entry-is-b4"];

VerificationTest[VertexCount[corpus["crypto-des"]["Graph"]], 14,
  TestID -> "crypto-des/parses-14-blocks"];

VerificationTest[Length[corpus["minimal-scc"]["Final"]], 6,
  TestID -> "minimal-scc/six-blocks-have-final-values"];

(* ---------- the oracle: reference implementation vs the Go compiler ---------- *)

iterDiff[name_] := With[{d = corpus[name]},
  LiveRange`FindDifference[
    Last[LiveRange`IterativeSolution[d["Start"], d["Graph"], d["Entry"], d["Context"]]],
    d["Final"]]];

threeDiff[name_] := With[{d = corpus[name]},
  LiveRange`FindDifference[
    Last[LiveRange`ThreePassSolution[d["Start"], d["Graph"], d["Entry"], d["Context"]]],
    d["Final"]]];

VerificationTest[iterDiff["minimal-scc"], <||>,
  TestID -> "minimal-scc/iterative-matches-compiler"];

VerificationTest[iterDiff["crypto-des"], <||>,
  TestID -> "crypto-des/iterative-matches-compiler"];

(* ---------- three-pass: succeeds on one CFG, fails on the other ---------- *)

VerificationTest[threeDiff["crypto-des"], <||>,
  TestID -> "crypto-des/three-pass-matches-compiler"];

(* Three-pass is NOT correct in general. On minimal-scc it drops v37 and v41
   from b9. This pins the known-wrong answer deliberately: if it starts
   returning <||>, three-pass changed and the claim must be revisited. *)
VerificationTest[
  threeDiff["minimal-scc"],
  <|Inactive[LiveRange`Blk][9] -> {Inactive[LiveRange`Val][37], Inactive[LiveRange`Val][41]}|>,
  TestID -> "minimal-scc/three-pass-is-wrong"];

(* ---------- why it fails: pass 2 starts from a sink and covers nothing ---------- *)

poBwdCoverage[name_] := With[{d = corpus[name]},
  Length @ LiveRange`PostOrder[d["Graph"], First @ LiveRange`PostOrder[d["Graph"], d["Entry"]]]];

VerificationTest[poBwdCoverage["minimal-scc"], 1,
  TestID -> "minimal-scc/poBwd-covers-only-one-block"];

VerificationTest[poBwdCoverage["crypto-des"], 14,
  TestID -> "crypto-des/poBwd-covers-all-blocks"];

(* ---------- the compiler's SCC sweep scheme (SccSweepSolution) ----------

   These pin the Wolfram twin of computeLiveWithSccs. The 2026-08-25 census
   over std+cmd found 23.35% of real SCCs still changing after 3 sweeps and
   86.84% needing more than 2; these corpus cases pin the small end of that
   gradient (crypto-des: 2, minimal-scc: 3) and the exact values a 2-sweep
   cap loses. *)

VerificationTest[
  LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["minimal-scc"]]["State"],
    corpus["minimal-scc"]["Final"]],
  <||>,
  TestID -> "scc-sweeps/minimal-scc-converged-matches-compiler"];

VerificationTest[
  LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["crypto-des"]]["State"],
    corpus["crypto-des"]["Final"]],
  <||>,
  TestID -> "scc-sweeps/crypto-des-converged-matches-compiler"];

VerificationTest[
  #["Sweeps"] & /@ LiveRange`SccSweepSolution[corpus["minimal-scc"]]["SCCs"],
  {3},
  TestID -> "scc-sweeps/minimal-scc-needs-3-changing-sweeps"];

VerificationTest[
  #["Sweeps"] & /@ LiveRange`SccSweepSolution[corpus["crypto-des"]]["SCCs"],
  {2},
  TestID -> "scc-sweeps/crypto-des-needs-2-changing-sweeps"];

(* A 2-sweep cap loses exactly the loop-carried tail: v41 fails to complete
   the cycle b4->b5->b8->b6->b4 back to its own use point, v68 similarly at
   b8. Deliberately pinned non-empty: if this returns <||>, the cap semantics
   changed and the sufficiency claim must be re-examined. *)
VerificationTest[
  LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["minimal-scc"], 2]["State"],
    corpus["minimal-scc"]["Final"]],
  <|Inactive[LiveRange`Blk][4] -> {Inactive[LiveRange`Val][41]},
    Inactive[LiveRange`Blk][8] -> {Inactive[LiveRange`Val][68]}|>,
  TestID -> "scc-sweeps/two-sweep-cap-drops-loop-carried-values"];

(* The 3-sweep cap is sufficient for BOTH corpus cases -- the old compiler
   cap survives these CFGs. The >3 witnesses live in std+cmd (479 of 2051
   SCCs); extracting one into the corpus is the pending step. *)
VerificationTest[
  LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["minimal-scc"], 3]["State"],
    corpus["minimal-scc"]["Final"]],
  <||>,
  TestID -> "scc-sweeps/three-sweep-cap-suffices-on-minimal-scc"];

(* ---------- real >3-sweep witnesses, extracted from std+cmd ----------

   LIVEDUMP-extracted from the instrumented compiler (2026-08-26 batch,
   vmbench/results/20260826T025403-postsweep/). Cross-validation: the sweep
   counts the Wolfram model reports independently reproduce the in-compiler
   converge-and-count instrumentation (forEachSpecial: 5, FprintFunc: 6). *)

VerificationTest[
  LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["forEachSpecial"]]["State"],
    corpus["forEachSpecial"]["Final"]],
  <||>,
  TestID -> "witness/forEachSpecial-converged-matches-compiler"];

VerificationTest[
  #["Sweeps"] & /@ LiveRange`SccSweepSolution[corpus["forEachSpecial"]]["SCCs"],
  {5},
  TestID -> "witness/forEachSpecial-needs-5-sweeps"];

(* runtime.forEachSpecial under the OLD 3-sweep cap: seven blocks lose the
   same four loop-carried values. Even four sweeps still miss them at b35.
   This is the >3 witness the 23.35% census statistic is made of. *)
VerificationTest[
  Length[LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["forEachSpecial"], 3]["State"],
    corpus["forEachSpecial"]["Final"]]],
  7,
  TestID -> "witness/forEachSpecial-cap3-wrong-at-seven-blocks"];

VerificationTest[
  Lookup[LiveRange`FindDifference[
     LiveRange`SccSweepSolution[corpus["forEachSpecial"], 3]["State"],
     corpus["forEachSpecial"]["Final"]],
   Inactive[LiveRange`Blk][24]],
  {Inactive[LiveRange`Val][3], Inactive[LiveRange`Val][13],
   Inactive[LiveRange`Val][66], Inactive[LiveRange`Val][68]},
  TestID -> "witness/forEachSpecial-cap3-drops-four-values-at-b24"];

VerificationTest[
  LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["FprintFunc"]]["State"],
    corpus["FprintFunc"]["Final"]],
  <||>,
  TestID -> "witness/FprintFunc-converged-matches-compiler"];

VerificationTest[
  Sort[#["Sweeps"] & /@ LiveRange`SccSweepSolution[corpus["FprintFunc"]]["SCCs"]],
  {1, 6},
  TestID -> "witness/FprintFunc-needs-6-sweeps"];

VerificationTest[
  Length[LiveRange`FindDifference[
    LiveRange`SccSweepSolution[corpus["FprintFunc"], 3]["State"],
    corpus["FprintFunc"]["Final"]]],
  37,
  TestID -> "witness/FprintFunc-cap3-wrong-at-37-of-78-blocks"];
