(* ::Package:: *)

(* LiveRange` -- reference implementation of Go SSA live-variable analysis.

   Canonical definitions extracted from LiveRangeAnalysis.nb and
   ThreePassSuccess.nb, which had independently diverged copies of every
   function here. This package is the single source of truth; notebooks
   should Needs["LiveRange`"] rather than redefine.

   Normalizations applied during extraction (deliberate, see README):
     - Values are Inactive[Val][id] throughout. The notebooks mixed bare
       symbols (v68) with Inactive[Val][68]; bare symbols were created via
       Symbol[...] and so landed in whatever context happened to be current.
     - Liveness sets are sets of value IDENTITIES. Distances are preserved by
       the parser but stripped by the solver, because Union/Complement and
       SymmetricDifference are set operations and are meaningless on
       value->distance rules.
     - The use/phi/def maps are passed explicitly as a context instead of
       being read from global symbols. In the notebooks these were globals
       reassigned between examples, which let one example's maps leak into
       the next.
     - ThreePassSuccess.nb's passFunction referenced Source[] and Sink[],
       which are defined nowhere. They correspond to the use map and the def
       map respectively; that is how they are bound here.
*)

BeginPackage["LiveRange`"];

Blk::usage       = "Blk[id] is a basic block; blocks are represented as Inactive[Blk][id].";
Val::usage       = "Val[id] is an SSA value; values are represented as Inactive[Val][id].";

ParseCFG::usage           = "ParseCFG[text] parses Go compiler CFG dump text into a Graph.";
ParseBlockValues::usage   = "ParseBlockValues[text] parses 'bN: vX(d) vY(d)' lines into <|Inactive[Blk][N] -> {Inactive[Val][X] -> d, ...}|>.";
ImportCFG::usage          = "ImportCFG[dir] loads a corpus directory containing cfg/liveout/uses/phis/defs/entry .txt files.";

ValueNames::usage      = "ValueNames[list] reduces a list of value->distance rules to bare value identities.";
ValueDistances::usage  = "ValueDistances[list] extracts the distances from a list of value->distance rules.";
LookupOrEmpty::usage   = "LookupOrEmpty[assoc, key] returns assoc[key] or {} when absent. LookupOrEmpty[assoc] returns the curried form.";

PostOrder::usage    = "PostOrder[graph, source] returns the depth-first postorder vertex list.";
LeavesFirst::usage  = "LeavesFirst[domTree, entry] orders dominator-tree vertices deepest-first, as an Association vertex->depth.";

LivenessContext::usage      = "LivenessContext[uses, phis, defs] builds the context consumed by the solvers.";
PassFunction::usage         = "PassFunction[reverseGraph, ctx] returns the single-block transfer function used by Fold.";
IterativeSolution::usage    = "IterativeSolution[start, graph, entry, ctx] iterates to a fixed point; returns the FixedPointList.";
ThreePassSolution::usage    = "ThreePassSolution[start, graph, entry, ctx] runs forward/backward/forward postorder passes; returns {pass1, pass2, pass3}.";

FindDifference::usage = "FindDifference[a, b] returns per-block symmetric differences of value identities, omitting blocks that agree. An empty Association means the two solutions match.";

Begin["`Private`"];

(* ---------- representation ---------- *)

blockOf[id_Integer] := Inactive[Blk][id];
valueOf[id_Integer] := Inactive[Val][id];

digits[s_String] := ToExpression /@ StringCases[s, RegularExpression["\\d+"]];

blockToken[s_String] := blockOf[First[digits[s]]];
valueToken[s_String] := Module[{n = digits[s]},
  valueOf[n[[1]]] -> If[Length[n] >= 2, n[[2]], 0]];

(* ---------- parsing ---------- *)

ParseCFG[str_String] := Module[{lines, split, edges},
  lines = Select[StringSplit[str, "\n"], StringContainsQ[#, "->"] &];
  split = StringCases[#, RegularExpression["b\\d+"]] & /@ StringSplit[#, "->"] & /@ lines;
  edges = Flatten @ Map[
    Function[parts,
      If[Length[parts] < 2 || parts[[1]] === {} || parts[[2]] === {}, {},
        Outer[blockToken[#1] -> blockToken[#2] &, parts[[1]], parts[[2]]]]],
    split];
  Graph[DeleteDuplicates[edges], VertexLabels -> "Name"]];

ParseBlockValues[str_String] := Association @ Map[
  Function[parts,
    If[Length[parts] < 2 || parts[[1]] === {}, Nothing,
      blockToken[parts[[1, 1]]] -> (valueToken /@ parts[[2]])]],
  StringCases[#, RegularExpression["b\\d+(\\(\\d+\\))?|v\\d+\\(\\d+\\)"]] & /@
      StringSplit[#, ":"] & /@ StringSplit[str, "\n"]];

ImportCFG[dir_String] := Module[{rd, entryTxt},
  rd[f_] := If[FileExistsQ[FileNameJoin[{dir, f}]], Import[FileNameJoin[{dir, f}], "Text"], ""];
  entryTxt = rd["entry.txt"];
  <|"Graph"    -> ParseCFG[rd["cfg.txt"]],
    "Final"    -> ParseBlockValues[rd["cfg.txt"]],
    "Start"    -> ParseBlockValues[rd["liveout.txt"]],
    "Context"  -> LivenessContext[ParseBlockValues[rd["uses.txt"]],
                                  ParseBlockValues[rd["phis.txt"]],
                                  ParseBlockValues[rd["defs.txt"]]],
    "Entry"    -> blockOf[First[digits[entryTxt]]],
    "Name"     -> FileNameTake[dir]|>];

(* ---------- small helpers ---------- *)

ValueNames[rules_List]     := Replace[rules, (x_ -> _) :> x, {1}];
ValueDistances[rules_List] := Cases[rules, (_ -> d_) :> d];

LookupOrEmpty[m_Association, k_] := Lookup[m, Key[k], {}];
LookupOrEmpty[m_Association]     := LookupOrEmpty[m, #] &;

normalize[a_Association] := Map[ValueNames, a];

(* ---------- graph orders ---------- *)

PostOrder[g_Graph, source_] := Module[{sown},
  sown = Reap[DepthFirstScan[g, source, {"PostvisitVertex" -> Sow}]][[2]];
  If[sown === {}, {}, First[sown]]];

LeavesFirst[domTree_Graph, entry_] := Module[{verts, parent, depth},
  verts  = VertexList[domTree];
  parent = Association[Rule @@@ (Reverse /@ EdgeList[domTree])];
  depth[x_] := depth[x] = If[x === entry || ! KeyExistsQ[parent, x], 0, 1 + depth[parent[x]]];
  ReverseSortBy[AssociationMap[depth, verts], Identity]];

(* ---------- liveness ---------- *)

LivenessContext[uses_Association, phis_Association, defs_Association] :=
  <|"Uses" -> normalize[uses], "Phis" -> normalize[phis], "Defs" -> normalize[defs]|>;

PassFunction[revGraph_Graph, ctx_Association] := Function[{state, node},
  Module[{live, preds},
    (* 1. live-at-start of this block: uses plus what is live at its end,
          minus anything this block defines (a definition ends a live range) *)
    live = Complement[
      Union[LookupOrEmpty[ctx["Uses"], node], LookupOrEmpty[state, node]],
      LookupOrEmpty[ctx["Defs"], node]];
    (* 2. propagate to predecessors; phi references must be live at the end
          of the PARENT block, so they are added there *)
    preds = #[[2]] & /@ EdgeList[revGraph, node \[DirectedEdge] _];
    (* 3. update the parents' live-at-end sets *)
    Append[state, AssociationMap[
      Union[LookupOrEmpty[ctx["Phis"], #], LookupOrEmpty[state, #], live] &, preds]]]];

IterativeSolution[start_Association, graph_Graph, entry_, ctx_Association] :=
  FixedPointList[
    Fold[PassFunction[ReverseGraph[graph], ctx], #, PostOrder[graph, entry]] &,
    normalize[start]];

ThreePassSolution[start_Association, graph_Graph, entry_, ctx_Association] :=
  Module[{step, poFwd, poBwd, p1, p2, p3},
    step  = PassFunction[ReverseGraph[graph], ctx];
    poFwd = PostOrder[graph, entry];
    poBwd = PostOrder[graph, First[poFwd]];
    p1 = Fold[step, normalize[start], poFwd];
    p2 = Fold[step, p1, poBwd];
    p3 = Fold[step, p2, poFwd];
    {p1, p2, p3}];

(* ---------- comparison ---------- *)

FindDifference[a_Association, b_Association] := DeleteCases[
  AssociationMap[
    SymmetricDifference[
      ValueNames[LookupOrEmpty[a, #]],
      ValueNames[LookupOrEmpty[b, #]]] &,
    Union[Keys[a], Keys[b]]],
  {}];

End[];
EndPackage[];
