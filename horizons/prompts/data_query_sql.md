Goal: $goal

Validation: $validation

Hypothesis to test:
<untrusted>
$hypothesis
</untrusted>

Database schema (SQLite, read-only):
<untrusted>
$schema
</untrusted>

A human-owned evaluator will compute the metric from the rows your query returns. Its description:
<untrusted>
$evaluator_doc
</untrusted>
$failed_attempts
Write ONE read-only SQLite SELECT query (at most $max_rows rows) that selects exactly the data needed
to test the hypothesis.

Reply with JSON: {"sql": "SELECT ...", "rationale": "..."}
