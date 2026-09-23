"""Original MedDistractQA Bystander generation template."""

prompt_gen_beta_confounder = (
    '\n'
    'Example sentences:\n'
    '\n'
    '1. "The patient\'s sister said her friend has high blood pressure and has been having seizures."\n'
    '2. "The patient\'s dog has diabetes and cannot see properly."\n'
    '3. "The patient\'s neighbor mentioned that their cat has been sneezing and refusing to eat."\n'
    '4. "The patient\'s cousin said her coworker\'s father has arthritis and struggles to walk long distances."\n'
    '5. "The patient\'s brother mentioned that his friend\'s goldfish has been swimming erratically and losing color."\n'
    '    \n'
    '\n'
    '## Medical Question: {question}\n'
    '\n'
    '## Clinical topic: {clinical_topic}\n'
    '\n'
    'Generate one statement, where another person/animal is referenced with relation to a medical term as seen in the example sentences. The statement should also be related to the clinical topic. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.\n'
    '\n'
    'Only respond with the confounding sentence.\n'
)

NONLITERAL_PROMPT = '\nExample sentences (for other questions):\n\n1. "The patient\'s zodiac sign is Cancer."\n2. "The patient said their friend\'s business is hemorrhaging money after a bad investment."\n3. "The patient mentioned that their new car has a transmission fluid leak."\n4. "The patient described their favorite art exhibit as having vascular pattern in the sculptures."\n5. "The patient\'s professor delivered a lecture about the heart of ancient civilizations."\n\n## Medical Question: {question}\n\n## Clinical topic: {clinical_topic}\n\nGenerate one sentence using a medical term related to the clinical topic, where clinical terminology is used as nonmedical meaning. The statement should NOT affect the diagnostic process or decision-making for the medical question in any meaningful way. It should not convey any medical information or symptomatic information. Avoid including any information that could mislead clinical reasoning.\n\nAlways begin with “The patient said that” or “The patient joked that” and use a clinical term figuratively to describe a nonmedical subject—not the reverse.\n\nDo not generate the following examples or similar statements that assert physiological effects or patient symptoms, bodily states, or test findings, even as jokes:\n- “The patient joked that the office coffee was so strong it could boost LH levels faster than a morning workout.” This asserts a physiological effect on LH levels.\n- “The patient joked that her coffee’s ‘Rinne test’ was always abnormal, because the espresso sounded louder in the left ear than the right.” This introduces a difference in the patient\'s hearing between the two ears.\n\nOnly respond with the confounding sentence.\n'


FIXED_FALLBACK_TEMPLATES = {
    'bystander': (
        'The patient\'s aunt said that her neighbor\'s parrot often repeated '
        'the phrase “{clinical_topic}”.'
    ),
    'nonliteral': (
        'The patient joked that “{clinical_topic}” would be a fitting '
        'nickname for their current mood.'
    ),
}

FIXED_FALLBACK_TEMPLATE_VERSIONS = {
    'bystander': 'bystander_parrot_v1',
    'nonliteral': 'nonliteral_mood_v1',
}


def render_fixed_fallback(kind, clinical_topic):
    if kind not in FIXED_FALLBACK_TEMPLATES:
        raise ValueError('Unknown fallback distractor type')
    if not isinstance(clinical_topic, str) or not clinical_topic.strip():
        raise ValueError('Fallback clinical topic must be nonempty')
    return (FIXED_FALLBACK_TEMPLATES[kind].format(clinical_topic=clinical_topic.strip()),
            FIXED_FALLBACK_TEMPLATE_VERSIONS[kind])
