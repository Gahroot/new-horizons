Goal: $goal

Validation: $validation

Candidate hypotheses:
<untrusted>
$candidates
</untrusted>

Act as a strict reviewer. Score each candidate 0-10 on: chance of meeting the target, how cleanly the
test falsifies it, novelty versus what was already tried, and cost. Penalise vague or untestable
statements.

Reply with JSON: {"ranking": [{"index": <candidate number>, "score": <0-10>, "reason": "..."}]}
