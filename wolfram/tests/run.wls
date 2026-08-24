#!/usr/bin/env wolframscript
(* Headless test runner.  Usage:  wolframscript -file tests/run.wls  *)

here = DirectoryName[$InputFileName];
SetEnvironment["LIVERANGE_ROOT" -> DirectoryName[StringDrop[here, -1]]];
report = TestReport[FileNameJoin[{here, "LiveRange.wlt"}]];

pass = report["TestsSucceededCount"];
fail = report["TestsFailedCount"];

Print["LiveRange` test report"];
Print["  passed: ", pass];
Print["  failed: ", fail];
Print[""];

Do[
  Print[If[res["Outcome"] === "Success", "  PASS  ", "  FAIL  "], res["TestID"]],
  {res, Values[report["TestResults"]]}];

If[fail > 0,
  Print["\n--- failure detail ---"];
  Do[
    If[res["Outcome"] =!= "Success",
      Print[""];
      Print["  ", res["TestID"], "  (", res["Outcome"], ")"];
      Print["    expected: ", InputForm[res["ExpectedOutput"]]];
      Print["    actual:   ", InputForm[res["ActualOutput"]]];
      If[res["ActualMessages"] =!= {},
        Print["    messages: ", InputForm[res["ActualMessages"]]]]],
    {res, Values[report["TestResults"]]}]];

Exit[If[fail > 0, 1, 0]]
