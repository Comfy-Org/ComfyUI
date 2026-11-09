from pydantic import BaseModel, Field


class VeedFabricRequest(BaseModel):
    image_url: str = Field(...)
    audio_url: str = Field(...)
    resolution: str = Field(...)


class VeedLipsyncRequest(BaseModel):
    video_url: str = Field(...)
    audio_url: str = Field(...)


class VeedBackgroundRemovalRequest(BaseModel):
    video_url: str = Field(...)
    output_codec: str = Field(...)
    refine_foreground_edges: bool = Field(...)
    subject_is_person: bool = Field(...)


class VeedGreenScreenRequest(BaseModel):
    video_url: str = Field(...)
    output_codec: str = Field(...)
    spill_suppression_strength: float = Field(...)


class VeedCleanAudioRequest(BaseModel):
    audio_url: str = Field(...)
    strength: float = Field(...)
    normalize_loudness: bool = Field(...)
    target_lufs: float | None = Field(None)
    output_format: str = Field(...)


class VeedCleanAudioVideoRequest(BaseModel):
    video_url: str = Field(...)
    strength: float = Field(...)
    normalize_loudness: bool = Field(...)
    target_lufs: float | None = Field(None)


class VeedDubbingRequest(BaseModel):
    video_url: str = Field(...)
    target_language: str = Field(...)
    source_language: str | None = Field(None)


class VeedFile(BaseModel):
    url: str = Field(...)


class VeedJobResult(BaseModel):
    video: VeedFile | None = Field(None)
    audio: VeedFile | None = Field(None)
    files: list[VeedFile] | None = Field(None)


class VeedJobError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class VeedJob(BaseModel):
    job_id: str = Field(...)
    status: str = Field(...)
    result: VeedJobResult | None = Field(None)
    error: VeedJobError | None = Field(None)


class VeedJobResponse(BaseModel):
    data: VeedJob = Field(...)


VEED_DUBBING_LANGUAGES = {
    "Afrikaans": "af",
    "Akan": "ak",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Assamese": "as",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Burmese": "my",
    "Cantonese": "yue",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chichewa": "ny",
    "Chinese (Mandarin)": "zh",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Filipino": "fil",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Ganda": "lg",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Kikuyu": "ki",
    "Kinyarwanda": "rw",
    "Kirundi": "rn",
    "Korean": "ko",
    "Kyrgyz": "ky",
    "Latvian": "lv",
    "Lingala": "ln",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Northern Sotho": "nso",
    "Norwegian": "no",
    "Occitan": "oc",
    "Odia": "or",
    "Pashto": "ps",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Sindhi": "sd",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Southern Sotho": "st",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swati": "ss",
    "Swedish": "sv",
    "Tajik": "tg",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Tibetan": "bo",
    "Tsonga": "ts",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Venda": "ve",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Wolof": "wo",
    "Yoruba": "yo",
    "Zulu": "zu",
}

VEED_DUBBING_ACCENTS = {
    "Arabic (Egypt)": "ar-EG",
    "Chinese (Taiwan)": "zh-TW",
    "English (Australia)": "en-AU",
    "English (Canada)": "en-CA",
    "English (UK)": "en-GB",
    "English (US)": "en-US",
    "French (Canada)": "fr-CA",
    "French (France)": "fr-FR",
    "Portuguese (Brazil)": "pt-BR",
    "Portuguese (Portugal)": "pt-PT",
    "Spanish (Argentina)": "es-AR",
    "Spanish (Chile)": "es-CL",
    "Spanish (Mexico)": "es-MX",
    "Spanish (Spain)": "es-ES",
}

VEED_DUBBING_STANDARD_ONLY = (
    "Asturian", "Bengali", "Chichewa", "Irish", "Lingala", "Luxembourgish", "Maltese", "Maori", "Occitan", "Odia",
    "Pashto", "Serbian", "Somali",
)
VEED_DUBBING_PRO_ONLY = (
    "Akan", "Albanian", "Amharic", "Basque", "Ganda", "Kikuyu", "Kinyarwanda", "Kirundi", "Northern Sotho",
    "Southern Sotho", "Sundanese", "Swati", "Tibetan", "Tsonga", "Tswana", "Uyghur", "Venda", "Wolof", "Zulu",
)
