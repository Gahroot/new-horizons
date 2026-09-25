Hypothesis:
<untrusted>
$hypothesis
</untrusted>

Papers (ID - title - abstract):
<untrusted>
$papers
</untrusted>

For each paper decide whether its abstract supports the hypothesis, challenges it, or is unrelated.
Quote the exact sentence from the abstract that justifies a supports/challenges judgement. If you
cannot quote a sentence, answer "unrelated".

Reply with JSON:
{"judgements": [{"paper_id": "...", "relation": "supports|challenges|unrelated",
                 "evidence_quote": "exact sentence from the abstract", "confidence": 0.0-1.0}]}
