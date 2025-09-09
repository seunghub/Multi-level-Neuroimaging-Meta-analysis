import pandas as pd

from src.constants import FMRI_DECODING_PATH


DEFAULT_COGNITIVE_ATLAS_CONCEPTS_FILEPATH = (
    FMRI_DECODING_PATH
    / "scripts"
    / "preprocessing"
    / "peak"
    / "assets"
    / "cogatlas_concepts.json"
)


class CognitiveAtlas:
    def __init__(self, concepts_path=DEFAULT_COGNITIVE_ATLAS_CONCEPTS_FILEPATH):
        # Inferred from http://cognitiveatlas.org/concepts/categories/all
        self.concepts_classes = {
            "ctp_C1": "Perception",
            "ctp_C2": "Attention",
            "ctp_C3": "Reasoning and decision making",
            "ctp_C4": "Executive cognitive control",
            "ctp_C5": "Learning and memory",
            "ctp_C6": "Language",
            "ctp_C7": "Action",
            "ctp_C8": "Emotion",
            "ctp_C9": "Social function",
            "ctp_C10": "Motivation",
            "": "Misc",
        }

        self.additional_concepts_mapping = {
            "anticipation": "Attention",
            "arousal": "Emotion",
            "arithmetic processing": "Reasoning and decision making",
            "concept": "Reasoning and decision making",
            "effort": "Action",
            "creative thinking": "Reasoning and decision making",
            "emotion regulation": "Emotion",
            "emotional enhancement": "Emotion",
            "guilt": "Emotion",
            "imagination": "Reasoning and decision making",
            "phonological processing": "Language",
            "semantic categorization": "Language",
            "story comprehension": "Language",
            "visual orientation": "Executive cognitive control",
            "strategy": "Reasoning and decision making",
            "thought": "Reasoning and decision making",
        }

        # Concept file taken from https://cognitiveatlas.org/api/v-alpha/concept?format=json
        self.concepts = pd.concat(
            [
                (
                    pd.read_json(concepts_path)
                    .loc[:, ["id_concept_class", "name", "definition_text"]]
                    .assign(
                        name=lambda df: df.name.str.lower(),
                        word=lambda df: df.name,
                    )
                    .fillna("")
                    .assign(
                        category=lambda df: df.id_concept_class.map(
                            self.concepts_classes
                        )
                    )
                    .drop(columns=["name"])
                    .drop_duplicates()
                    .loc[
                        lambda df: ~df.word.isin(
                            self.additional_concepts_mapping.keys()
                        )
                    ]
                ),
                pd.DataFrame(
                    {
                        "category": self.additional_concepts_mapping.values(),
                        "word": self.additional_concepts_mapping.keys(),
                    }
                ),
            ],
            axis=0,
        )

        self.category_colors = {
            category: color
            for category, color in zip(
                self.concepts_classes.values(),
                [
                    "red",
                    "blue",
                    "purple",
                    "green",
                    "black",
                    "cyan",
                    "red",
                    "blue",
                    "purple",
                    "cyan",
                    "yellow",
                ],
            )
        }

    def get_category_for_word(self, word):
        return self.concepts.loc[lambda df: df.word == word].category.values[0]

    def get_color_for_word_by_category(self, word):
        category = self.get_category_for_word(word)

        return self.category_colors[category]

    def get_definition_for_word(self, word):
        return self.concepts.loc[lambda df: df.word == word].definition_text.values[0]


if __name__ == "__main__":
    cognitive_atlas = CognitiveAtlas()
    print(cognitive_atlas.concepts.head())
    print(cognitive_atlas.get_category_for_word("action"))
    print(cognitive_atlas.get_category_for_word("language processing"))
    print(cognitive_atlas.get_category_for_word("phonological processing"))
