# from `gold_with_3_distractors_context_cot_qa_codex.txt`

one_shot_rag_qa_docs = (
    "BCR-ABL1 Fusion has a targeted therapy Dasatinib for B-Lymphoblastic Leukemia/Lymphoma.",
        "BCR-ABL1 Fusion is well-studied and results in constitutive downstream JAK/STAT and PI3K signaling.",
        "Small molecule inhibitors of ABL1, including FDA-approved imatinib, dasatinib, and nilotinib, have had high levels of clinical activity in patients with the BCR-ABL1 fusion.",
        "PIK3CA also known as PI3K.",
        "PIK3CA, the catalytic subunit of PI3-kinase, is frequently mutated in a diverse range of cancers including breast, endometrial and cervical cancers.",
        "The PI3K pathway is an intracellular signal transduction pathway that regulates key cellular processes like growth, proliferation, survival, and metabolism. It starts with PI3K (phosphatidylinositol 3-kinase), which is activated by extracellular signals, leading to the activation of downstream proteins like AKT (also known as Protein Kinase B). The pathway is crucial for many cellular functions, and its dysregulation is frequently linked to diseases like cancer.",
        "The PI3K/AKT/mTOR pathway is an intracellular signaling pathway important in regulating the cell cycle. Therefore, it is directly related to cellular quiescence, proliferation, cancer, and longevity."
)



one_shot_ircot_demo = (
    f'{one_shot_rag_qa_docs}'
    '\n\nQuestion: '
    f"What cancers frequently have PIK3CA mutations?"
    '\nThought: '
    f"PIK3CA, also known as PI3K, is frequently mutated in a diverse range of cancers including breast, endometrial and cervical cancers. Answer: breast, endometrial, and cervical cancers."
    '\n\n'
)


rag_qa_system = (
    'As an experienced doctor with biomedical background in diverse fields, your task is to analyze text passages and corresponding questions meticulously. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "What cancers frequently have PIK3CA mutations?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "PIK3CA, also known as PI3K, is frequently mutated in a diverse range of cancers including breast, endometrial and cervical cancers. "
    "\nAnswer: breast, endometrial, and cervical cancers."
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]
