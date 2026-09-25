Hypothesis that the new program is supposed to test:
<untrusted>
$hypothesis
</untrusted>

Changes from the parent program ($parent_label) to the new candidate, as a unified diff:
<untrusted>
$diff
</untrusted>

Act as a strict code reviewer. Decide whether these changes actually implement the mechanism the
hypothesis names, so that a measured result could fairly be credited to (or blamed on) the hypothesis.
Answer "no" when the diff implements a different idea, only part of a multi-part idea, a stub or
placeholder, or code paths that never run. Tuning or refactoring that does not add the named mechanism
is also "no". Ignore style, and do not judge whether the idea will work.

Reply with JSON: {"implements": true or false, "missing": "<what is absent or different; empty if true>"}
