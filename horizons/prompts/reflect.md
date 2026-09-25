Goal: $goal

Hypothesis:
<untrusted>
$hypothesis
</untrusted>

Outcome decided by the evaluator: $outcome
Evidence: $evidence

Experiment log excerpt:
<untrusted>
$log
</untrusted>

Explain briefly why this outcome happened, then extract reusable lessons for future hypotheses.
Each lesson must name the specific mechanism that was tried and what the measurement showed about it
(with numbers when the evidence gives them). Generic advice such as "measure carefully" or "compare on
held-out data" is not a lesson; leave it out.
Categories: "system" (tooling/environment problems), "experiment" (what the measurement showed),
"literature" (what papers did or did not back up), "analysis" (interpretation, confounders).

Reply with JSON:
{"explanation": "...", "lessons": [{"category": "experiment", "text": "one actionable sentence"}]}
