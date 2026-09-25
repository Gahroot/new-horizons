Goal: $goal

Validation metric: $validation

Baseline measurement: $baseline

Papers found (ID - title - abstract):
<untrusted>
$papers
</untrusted>

Write a boundary map of current knowledge for this goal:
- known_approaches: what has been tried, each with the paper IDs it comes from
- gaps: what is unknown, contested, or not yet combined
- promising_directions: concrete directions worth testing with the available tools ($tools), with citations

Reply with JSON:
{"summary": "2-4 sentences",
 "known_approaches": [{"approach": "...", "citations": ["<paper id>"]}],
 "gaps": ["..."],
 "promising_directions": [{"direction": "...", "citations": ["<paper id>"]}]}
