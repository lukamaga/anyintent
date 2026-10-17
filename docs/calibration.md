# Calibrated outputs: check and scores

[Back to the README](../README.md)

Besides `classify`, two functions give numbers from 0 to 1 that were calibrated for R2Intent v1: `check` (a beta) and `scores`. Each was fitted on one part of the DEV data (7,081 rows of 223 intents) and checked on the rest, the held-out part (13,405 rows of 421 intents), against the limits of a rule registered before the calibration. Every result on this page is a held-out DEV result of R2Intent v1.

- The calibration error (ECE) is the average gap between a probability and how often the intent is right, over 15 equal-width bins; lower is better.
- The intervals are 95% intervals from 2,000 resamples of the held-out intents of each DEV set. ECE tends to run higher on resamples, so an interval can lie above its value.
- A value close to a limit of the rule, and the value that missed its limit, are printed with an extra digit (for example 0.03001), so that rounding never moves a value onto or across a limit.
- The Lithuanian and Polish DEV sets are machine translations of English ones.
- Unseen DEV sets are the 19 English DEV sets outside the seen-domain list (HINT3, the project's generated sets and the fresh V-37 and V-IND sets, generated for domains and industries held out of training) and their Lithuanian and Polish copies. The seen-domain list names the DEV sets in domains close to the training data of R2Intent v1; it was fixed by a rule before that data was built. The rule is approximate: HINT3's DEV set counts as unseen, although HINT3's training split is in the training data of R2Intent v1.

## Yes or no for one intent (check, beta)

```python
r = clf.check("I can't find my card anywhere",
              {"name": "lost card", "description": "The customer reports that a bank card was lost or stolen."})
r.yes, r.probability          # True or False, and the probability; yes means probability >= 0.5 (threshold=...)
r.score                       # the fused pair score (cross-encoder logit + w x cosine), the units of Prediction.score
```

`check(text, intent, threshold=0.5)` reads the message together with one intent and answers yes or no, with a probability. The intent is a name, a `{"name", "description"}` dict or a `(name, description)` tuple; a list is refused, so call `check` once per intent. The pair is scored with the same two passes as `classify` (the cosine of the two vectors and the cross-encoder logit of the pair), so the answer for an intent does not depend on the label set or on other intents, and `check` does not touch the current label set. A list of messages gives a list of results, in order.

What the probability means: the chance that the message expresses this intent, when a match and a non-match are taken as equally likely before the message is read. It was calibrated on DEV pairs in which the right intent of a message and other intents of the same label set carry equal weight, and 0.5 is the only threshold that was measured. If the intent is rare in your traffic, a yes is right less often. The share of yes answers that are right, from the held-out rates of right yes and right no answers:

| messages that express the intent | share of yes answers that are right [95% interval] |
|---|---|
| 1 in 2 | 0.888 [0.862, 0.909] |
| 1 in 5 | 0.665 [0.610, 0.715] |
| 1 in 20 | 0.295 [0.248, 0.346] |

A yes does not mean that the intent is the best one. On intents close to the right one (the other intents among the 5 that `classify` reads with the message), `check` wrongly says yes for 0.420 [0.391, 0.449] of them, and an intent outside those 5 can get a high probability although `classify` could never pick it. To choose one intent, use `classify`.

Results against the limits of its rule:

| clause | what was checked | limit | result [95% interval] |
|---|---|---|---|
| C1 | ECE, all held-out pairs | at most 0.030 (upper end of the interval at most 0.050) | 0.014 [0.008, 0.03001] |
| C2 | ECE, English / Lithuanian and Polish DEV sets | at most 0.050 each | 0.026 / 0.03005 |
| C3 | balanced accuracy at 0.5 | at least 0.85 (lower end at least 0.80) | 0.873 [0.856, 0.888] |
| C4 | ECE, unseen English DEV sets | at most 0.050 | 0.0566 [0.0375, 0.0790]: missed |
| C4 | ECE, unseen Lithuanian and Polish DEV sets | at most 0.050 | 0.015 |
| C5 | the library's engines auto and eager against the calibration scorer, on the CLINC150 DEV messages and their Lithuanian and Polish copies | probabilities within 0.01 on at least 99% of the pairs, ECE within 0.010 | 100% of the pairs, ECE difference below 0.001, with each engine |

Balanced accuracy is the mean of the right yes rate on the right intent, 0.853 [0.832, 0.873], and the right no rate on other intents of the same list, 0.893 [0.863, 0.915].

Beta: `check` missed one clause of its rule, C4. On the unseen English DEV sets, its probabilities were off by 0.0566 on average, above the limit of 0.050; every other clause passed. Under that rule `check` would not be in v1. It ships as a beta, with its parameters exactly as fitted, by a decision taken after this result. On the DEV sets of the seen-domain list (information, no clause), its probabilities were off by 0.066 [0.043, 0.092], so treat the probability of `check` as approximate for any intent. The HTTP and MCP texts mark it as a beta; their answers do not.

## A value per intent (scores)

```python
clf.set_labels([...])                  # scores works on the current label set
s = clf.scores("I can't find my card anywhere")
s.values                               # {name: value} for every intent of the label set, in set_labels order
s.read, s.label                        # the intents the model read, and classify's label
```

`scores(text)` runs the passes of `classify` and gives a value from 0 to 1 to every intent of the label set. `classify` reads 5 intents with each message (its k); on a label set of 6 intents or more the others are never read. A read intent's value is the chance that it is the right intent. An intent that was not read gets 0.0, which means "not read", not "impossible". `read` names the read intents, and `label`, the intent with the largest value, is the label `classify` gives. The largest value is the chance that this label is right.

The remainder: `s.remainder` is the chance that the right intent is one of your intents that the model did not read. The values and the remainder add up to 1, because both assume that one of your intents is right, so the remainder is not a "none of these intents" answer (there is no such mode). With 5 intents or fewer every intent is read and the remainder is 0. The intents not read share the remainder in proportions the model does not know. The HTTP and MCP answers do not show the remainder: they list only the read intents and their values.

How it is computed: on a label set of 6 intents or more, with f_(1) the highest and f_(5) the lowest fused score of the 5 read intents and L the size of the label set, each read intent j gets the logit (f_j - f_(5)) / T and the intents not read together get gamma + eta x (f_(1) - f_(5)) + kappa x log(L - 5); the values and the remainder are their softmax shares (T, gamma, eta and kappa are in the calibration file). On 5 intents or fewer, the values are the softmax shares of f_j / T over all of them.

Results against the limits of its rule; every clause passed:

| clause | what was checked | limit | result [95% interval] |
|---|---|---|---|
| S1 | ECE of the largest value, all held-out rows | at most 0.030 (upper end at most 0.050) | 0.019 [0.012, 0.035] |
| S2 | ECE of the largest value, English / Lithuanian and Polish DEV sets | at most 0.050 each | 0.016 / 0.026 |
| S3 | ECE of the largest value, unseen English / unseen Lithuanian and Polish DEV sets | at most 0.050 each | 0.032 / 0.043 |
| S4 | the library's engines auto and eager against the calibration scorer, on the CLINC150 DEV messages and their Lithuanian and Polish copies | every value and the remainder within 0.01 on at least 99% of the rows, ECE within 0.010 | 99.9% of the rows, ECE difference below 0.001, with each engine |
| S5 | ECE of every read value | at most 0.030 (upper end at most 0.050) | 0.005 [0.004, 0.010] |
| S6 | ECE of the remainder, label sets over 5 intents | at most 0.030 (upper end at most 0.050) | 0.006 [0.005, 0.018] |
| S6 | ECE of the remainder by label set size: 6 to 20, 21 to 40, 41 or more intents | at most 0.050 each | 0.008 [0.004, 0.034], 0.012 [0.006, 0.024], 0.013 [0.011, 0.035] |

The right intent was among the 5 read intents for 0.928 [0.912, 0.942] of the held-out rows (recall@5, information). Also for information: on held-out messages with an e-mail wrapper and on those without diacritics, the largest value understates the chance that the label is right (ECE 0.054 [0.037, 0.077] and 0.047 [0.0297, 0.068]), and for intents given by their names only it overstates it (ECE 0.041 [0.027, 0.070]).

## Notes for check and scores

- Where they hold. At the settings of R2Intent v1 (w = 60, k = 5, label texts `name: description`), for intents written with a description (both functions warn once when an intent has none), at the threshold 0.5 of `check`, and with the library's GPU engines `auto` and `eager` in fp16. CPU fp32 (PyTorch and ONNX fp32) was checked against the reference on tiny models only and can differ on near ties, as for `classify`. Label sets of 4 to 63 intents were measured; above 63 the remainder is extrapolated.
- Which models have them. They work only with a model release that ships the calibration file `r2intent_calibration.json` (next to `r2intent_meta.json`), and only for the functions that file names. R2Intent v1 ships it with `check` (beta) and `scores`. r2intent-zs-v1 ships none, so both functions refuse it; `classify` works the same either way.
- Refusals (`AnyIntentError`, at the call, with the reason): no calibration file or no block for the function; a `w=` override at load; the ONNX int8 variant; a calibration file of another model (its `model_sha256` is not the sha256 of the loaded `model.safetensors`; a local folder whose files changed after the load is refused too: load it again); a pair score or a read fused score that is not finite. `check` also refuses a `label_text_mode=` override at load and a list as the intent. `scores` also refuses a label set with another `label_text_mode=`, another w or k (a `k=` override at load), the few-shot mode, duplicate names and a call before `set_labels`.
- `Prediction.score`, `Prediction.margin` and `return_scores=True` stay in fused-score units, not probabilities; only `scores()` and `check()` give values from 0 to 1.
- Loading. Nothing changes at load. The calibration file is read at the first call of `check` or `scores` (a malformed file raises there, never at load). A model loaded from Hugging Face fetches only that file then, at the commit of the model files, with your stored login or `HF_TOKEN`. The fetch does not use `local_files_only=` or `token=` from the load: unless `HF_HUB_OFFLINE=1` is set, the first call of `check`, `scores` or `clf.calibration` on a model from the Hugging Face cache without this file asks the Hub for it once, and holds the instance lock while it waits (huggingface_hub sets no timeout on that request). To work offline, download it with the model files beforehand (`anyintent.hub.MODEL_FILES` lists the model files and `anyintent.hub.OPTIONAL_FILES` the optional ones), for example `hf download lukashm/r2intent-v1 r2intent_calibration.json --revision <commit>`, or set `HF_HUB_OFFLINE=1`. A model loaded from a local folder outside the Hugging Face cache never fetches it. `clf.calibration` shows the parsed file (or `None`).
