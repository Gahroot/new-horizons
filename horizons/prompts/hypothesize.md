Goal: $goal

Validation: $validation

Available tools (each hypothesis must name one as test_kind): $tools

Boundary map of current knowledge:
<untrusted>
$boundary_map
</untrusted>

Best result so far: $best

Hypotheses already tested in this run (do not repeat them):
<untrusted>
$tested
</untrusted>

Lessons recalled from memory:
<untrusted>
$lessons
</untrusted>

Paper IDs you may cite: $paper_ids

Propose $n distinct, falsifiable hypotheses that could move the metric toward the target. Take
different perspectives (e.g. an incremental fix, a different algorithm, a contrarian idea). For each:
- statement: one sentence, specific and testable
- rationale: why it might work, grounded in the boundary map
- test_kind: one of $tools
- falsification: the exact result that would prove it wrong
- expected_effect: predicted change in the metric
- citations: paper IDs from the list above (may be empty)
- alternatives: other explanations if the result looks positive
- readiness: "testable" if the available tools can test it, else "needs_resources"
- missing: if needs_resources, what is missing (data, hardware, tool)

Reply with JSON: {"hypotheses": [{...}, ...]}
