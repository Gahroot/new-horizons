Goal: $goal

Validation: $validation

Hypothesis to test:
<untrusted>
$hypothesis
</untrusted>

The evaluator (owned by the human, you cannot change it) loads `candidate.py` and measures it. The
program below is the current best version ($parent_label). Keep its public interface exactly the same
(function names, arguments, return format) - only change the implementation to test the hypothesis.

```python
$program
```

Last measured metrics for that program: $parent_metrics
$failed_attempts
Constraints: Python standard library only, no network, no file writes outside the working directory,
must finish within $timeout_s seconds.

Reply with the complete new candidate.py in one ```python code block, and nothing else.
