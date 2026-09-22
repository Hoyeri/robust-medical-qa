# Samples and input construction

## Attention sample

The attention analysis uses Bystander distractors from our unreleased, hardened version of MedDistractQA (internal name: MedDistractQA-Hard), with known source token positions. The original exploratory pool had 304 questions, of which 302 supported fixed-prefix replay. An outcome-independent SHA-256 ordering selected 96 for rationale-and-answer generation. The same 96 were evaluated in five conditions, producing 480 outcome records. [Attention blocking](01_attention_blocking.md) describes the controls and selection.

## Attribution sample

The attribution analyses use a separate set of 118 constructed MedQA-based questions. G denotes the original correct option. T denotes a competing option selected from the model's clean-question runner-up rankings across option rotations. The selection required a stable competitor in at least three of four rotations and a candidate finding associated with T in the item-local literature search. The frozen construction used a target-association share threshold of 0.95 and recorded audit labels.

M (stored as Ma) adds that selected finding to the patient. H1 adds the same finding to a third party. G remains the intended answer in the construction. "T-supporting" describes the finding-selection intent; the source set contains confirmed, nondirectional, and unaudited candidates. The probe labels are patient versus third-party attribution.

H1 was constructed only when the cue passed the symptom/history eligibility rule and a third-party template was available. The rule excludes terms associated with infection/exposure or familial, hereditary, and pregnancy contexts, where third-party information could be relevant to the patient. It is a lexical construction rule applied before probing.

| Construction step | Questions |
|---|---:|
| Parent set | 118 |
| Lab/exam/imaging cues without an H1 template | 56 |
| Symptom/history candidates | 62 |
| Candidates removed by the third-party exclusion rule | 9 |
| M/H1 pairs used in probing | 53 |

The nine excluded symptom candidates were rhinorrhea (1), cough/chronic cough (3), diarrhea (2), and concave nasal ridge (3). The last three match the lexical `nasal` exclusion. The 53 retained pairs contain 51 symptom and 2 history cues, spanning 35 HPO concepts. Their recorded audit labels are 4 confirmed T-direction, 7 nondirectional, and 42 unaudited.

Third-party wording follows the patient's age: coworker (39), classmate (5), another child at the clinic (5), or another baby at the clinic (4). The probe uses all 53 available pairs, without filtering by probe accuracy or intervention recovery. This gives 106 inputs; repeated seeds do not increase the independent question count.

## Actual pair

Item: `cmB_train-10163_B_v5`. [Machine-readable example](../examples/attribution_pair.jsonl).

Original question:

> A 4-year-old boy presents with a history of recurrent bacterial infections, including several episodes of pneumococcal sepsis. His family history is significant for 2 maternal uncles who died from similar symptoms. Laboratory tests reveal undetectable serum levels of all isotypes of immunoglobulins and reduced levels of B cells. Which of the following is the most likely diagnosis in this patient?

| Option | Diagnosis | Role |
|---|---|---|
| A | Bruton agammaglobulinemia | G |
| B | Common variable immunodeficiency | T |
| C | DiGeorge syndrome | Other |
| D | Hereditary angioedema | Other |

| Condition | Sentence inserted before the final question |
|---|---|
| M / Ma | He also has weight loss. |
| H1 | Another child at the clinic has weight loss. |

Both inputs retain the original question and choices. The shared finding phrase is `weight loss`; its token representations are pooled for the finding probe. The answer-position probe reads the final input token before answer prediction instead.

## Evaluation relationship

Attention blocking generates a new rationale and answer. Attribution and transmission use an empty-rationale answer prefix and read fixed-position representations/logits. Transmission M/H1 comparisons reuse the same 53 pairs as attribution probing. Their broader extraction contains 118 questions with available variants: M/U has 118 pairs and G+/U has 70.
