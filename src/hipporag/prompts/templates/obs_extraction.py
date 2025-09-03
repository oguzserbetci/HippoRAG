from ...utils.llm_utils import convert_format_to_template

one_shot_patient_paragraph = """\
PAS and PAS with diastase stains showed scattered tumor cells with diastase resistant granules . Immunohistochemical stains revealed strong positivity for S100 and TTF-1 ( Figure 1F ) with focal positivity for CD68 . Tumor cells were negative for GFAP , synaptophysin , keratins ( Cam5.2 , pancytokeratin ) , inhibin , carbonic anhydrase , PAX8 , and D2-40 .\
"""

ie_extraction_system = """You are an IE system for patient case reports and biomedical literature.

Extract a JSON array of OBSERVATIONS. Each statement is one short claim (your words) containing important features mentioned in the provided text.
For each statement, list linked FEATURES with fields:
- feature: short, normalized name, e.g. biomarker, operation, therapy, disease progression. 
- type: one of [immunohistochemistry, molecular, imaging, therapy, trial_endpoint, adverse_event, clinical] 
- category: one of [Features = baseline/biomarkers; Exposures = treatments/operations/medication; Outcomes = endpoints/toxicities/recurrence/progression] 
- status: one of [positive, negative, present, absent, given, not_given, stable, progressive, improved, worsened, increased, decreased] 
- relates_to: optional string or array naming the exposure(s) this observation refers to (e.g., "elacestrant") 

Rules: 
- Keep it simple. No numbers, no comparators, no spans.
- Clearly resolve abbreviations to their specific names to maintain clarity.
"""


input_frame = """
```
{passage}
```
"""

one_shot_output = """{
    "statements": [
        {
            "statement": "PAS and PAS with diastase stains showed diastase-resistant granules in tumor cells.",
            "features": [
                { "feature": "diastase-resistant granules", "type": "clinical", "category": "feature", "status": "present" }
            ]
        },
        {
            "statement": "Immunohistochemistry showed strong positivity for S100 and TTF-1.",
            "features": [
                { "feature": "S100", "type": "immunohistochemistry", "category": "feature", "status": "positive" },
                { "feature": "TTF-1", "type": "immunohistochemistry", "category": "feature", "status": "positive" }
            ]
        },
        {
            "statement": "Immunohistochemistry showed focal positivity for CD68.",
            "features": [
                { "feature": "CD68", "type": "immunohistochemistry", "category": "feature", "status": "positive" }
            ]
        },
        {
            "statement": "Tumor cells were negative for GFAP, synaptophysin, Cam5.2, pancytokeratin, inhibin, carbonic anhydrase, PAX8, and D2-40.",
            "features": [
                { "feature": "GFAP", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "synaptophysin", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "Cam5.2", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "pancytokeratin", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "inhibin", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "carbonic anhydrase", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "PAX8", "type": "immunohistochemistry", "category": "feature", "status": "negative" },
                { "feature": "D2-40", "type": "immunohistochemistry", "category": "feature", "status": "negative" }
            ]
        }
    ]
}"""


prompt_template = [
    {"role": "system", "content": ie_extraction_system},
    {"role": "user", "content": input_frame.format(passage=one_shot_patient_paragraph)},
    {"role": "assistant", "content": one_shot_output},
    {
        "role": "user",
        "content": convert_format_to_template(
            original_string=input_frame,
            placeholder_mapping=None,
            static_values=None,
        ),
    },
]
