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
