"""Versioned extraction prompt. Bump PROMPT_VERSION to invalidate the cache."""

from __future__ import annotations

PROMPT_VERSION = "v1"

_INSTRUCTIONS = """You extract how a person tried to retrieve a photo from Google Photos (or their camera roll) when their memory was incomplete.

Return ONLY one JSON object. Use these exact token values.

primary_archetype (one): utility_document | episodic_travel | aesthetic_visual | micro_moment | disambiguation_comparative | pre_verbal_sensory
secondary_archetypes: array of other archetypes that also fit, or []
primary_failure_stage (one): semantic_mismatch | ocr_failure | ranking_flooding | expression_failure
contributing_factors: array of other failure stages, or []
outcome: abandoned | retrieved_with_friction | social_offload | unknown
user_segment: heavy_shooter | traveler | parent | student | document_keeper | older_user | general | unknown
retrieval_surface: search_bar | ask_photos | memories | albums | timeline | faces | lens | unknown
emotion: anxiety | frustration | nostalgia_loss | resignation | relief | unknown
device_platform: ios | android | web | unknown
media_type: photo | video | screenshot | document | scan | unknown
stakes_level: medical_financial_legal | sentimental | trivial | unknown

memory_anchors_retained: object with arrays temporal, sensory, emotional, social.
Put rough time and library-size cues in temporal (for example "last year", "4000 photos").
Put colors, objects, layout in sensory. Put feelings and life events in emotional.
Put companions in social. Use [] when the text does not say.

information_forgotten: array of metadata they do not have (exact date, place name, album, drug name).
search_formulation: attempt_1_natural, attempt_2_keywords, attempt_3_desperation. Use null if they never searched that way.
user_workaround: short phrase or null.
region: place mentioned in the text, or null. Do not invent a country.
representative_quote: one short verbatim span copied from the text, max 240 characters.
extraction_confidence: number from 0 to 1. Low if the text is vague praise rather than a real retrieval story.

Archetype guide:
- utility_document: receipt, medicine, prescription, serial number, whiteboard, parking slip
- episodic_travel: trip, dinner, college, deceased family or pet, birthday, wedding
- aesthetic_visual: color, lighting, composition, sunset, a look
- micro_moment: one expression, pose, blooper, a single pet trick
- disambiguation_comparative: "not the fiery one, the pastel one", near-identical burst
- pre_verbal_sensory: they cannot name the object

Failure guide:
- semantic_mismatch: words they remember are not how the system is indexed
- ocr_failure: text inside the photo was not findable
- ranking_flooding: too many similar hits, cannot pick the one
- expression_failure: they could not form a query at all

Example shape:
{"primary_archetype":"utility_document","secondary_archetypes":[],"memory_anchors_retained":{"temporal":["last year"],"sensory":["blue bottle"],"emotional":["when I had the flu"],"social":[]},"information_forgotten":["exact date","medication name"],"search_formulation":{"attempt_1_natural":"medicine I took when sick last year","attempt_2_keywords":"blue bottle","attempt_3_desperation":null},"primary_failure_stage":"semantic_mismatch","contributing_factors":["ocr_failure"],"user_workaround":"scrolled for a long time","outcome":"abandoned","user_segment":"general","retrieval_surface":"search_bar","emotion":"anxiety","region":null,"device_platform":"android","media_type":"photo","stakes_level":"medical_financial_legal","representative_quote":"couldn't find the medicine photo","extraction_confidence":0.8}
"""


def build_prompt(text: str) -> str:
    clipped = (text or "").strip()[:3500]
    return f"{_INSTRUCTIONS}\n\nTEXT:\n{clipped}"
