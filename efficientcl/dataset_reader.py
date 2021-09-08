import logging
import random
import torch
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer
from overrides import overrides

from efficientcl.common.contrastive_utils import sample_anchor_positive_pairs
from efficientcl.common.data_utils import sanitize
import random
from allennlp.common import util
import torch.distributed as dist
from nltk.tokenize import sent_tokenize
from efficientcl.common.augmentation import eda

logger = logging.getLogger(__name__)
import logging
logging.basicConfig(level=logging.ERROR)

@DatasetReader.register("efficientcl")
class EfficientCLDatasetReader(DatasetReader):
    """
    Read a text file containing one instance per line, and create a dataset suitable for a
    `EfficientCL` model.

    The output of `read` is a list of `Instance` s with the field:
        tokens : `ListField[TextField]`
    if `num_anchors > 0`, else:
        tokens : `TextField`

    Registered as a `DatasetReader` with name "efficientcl".

    # Parameters

    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        We use this to define the input representation for the text. See :class:`TokenIndexer`.
    num_anchors : `int`, optional
        The number of spans to sample from each instance to serve as anchors.
    num_positives : `int`, optional
        The number of spans to sample from each instance to serve as positive examples (per anchor).
        Has no effect if `num_anchors` is not provided.
    max_span_len : `int`, optional
        The maximum length of spans (after tokenization) which should be sampled. Has no effect if
        `num_anchors` is not provided.
    min_span_len : `int`, optional
        The minimum length of spans (after tokenization) which should be sampled. Has no effect if
        `num_anchors` is not provided.
    sampling_strategy : `str`, optional (default = None)
        One of "subsuming" or "adjacent". If "subsuming," positive spans are always subsumed by the
        anchor. If "adjacent", positive spans are always adjacent to the anchor. If not provided,
        positives may be subsumed, adjacent to, or overlapping with the anchor. Has no effect if
        `num_anchors` is not provided.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        num_anchors: int = None,
        num_positives: int = None,
        max_span_len: int = None,
        min_span_len: int = None,
        sampling_strategy: str = None,
        eda: bool = None,
        declutr: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.eda = eda
        self.declutr = declutr
        # If the user provided us with a number of anchors to sample, we automatically
        # check that the other expected values are provided and valid.
        self.instance = 0
        if num_anchors is not None:
            self._num_anchors = num_anchors
            self.sample_spans = True
            if max_span_len is None:
                raise ValueError("max_span_len must be provided if num_anchors is not None.")
            if min_span_len is None:
                raise ValueError("min_span_len must be provided if num_anchors is not None.")
            self._num_positives = num_positives
            self._max_span_len = max_span_len
            self._min_span_len = min_span_len
            self._sampling_strategy = (
                sampling_strategy.lower() if sampling_strategy is not None else sampling_strategy
            )
            if (
                self.sample_spans
                and self._sampling_strategy is not None
                and self._sampling_strategy not in ["subsuming", "adjacent"]
            ):
                raise ValueError(
                    (
                        'sampling_strategy must be one of ["subsuming", "adjacent"].'
                        f" Got {self._sampling_strategy}."
                    )
                )
        else:
            self.sample_spans = False

    @property
    def sample_spans(self) -> bool:
        return self._sample_spans

    @sample_spans.setter
    def sample_spans(self, sample_spans: bool) -> None:
        self._sample_spans = sample_spans

    @contextmanager
    def no_sample(self) -> Iterator[None]:
        """A context manager that temporarily disables sampling of spans. Useful at test time when
        we want to embed unseen text.
        """
        prev = self.sample_spans
        self.sample_spans = False
        yield
        self.sample_spans = prev

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            num_lines = sum(1 for line in open(file_path))

            # If we are sampling spans (i.e. we are training) we need to shuffle the data so that
            # we don't yield instances in the same order every epoch. Our current solution is to
            # read the entire file into memory. This is a little expensive (roughly 1G per 1 million
            # docs), so a better solution might be required down the line.
            data: Iterable[Any] = []
            if self.sample_spans:
                data = list(enumerate(data_file))
                random.shuffle(data)
                data = iter(data)
            else:
                data = enumerate(data_file)
            for idx , text in data:
                yield self.text_to_instance(text, num_lines)

    @overrides
    def text_to_instance(self, text: str, num_lines: int) -> Instance:  # type: ignore
        """
        # Parameters

        text : `str`, required.
            The text to process.

        # Returns

        An `Instance` containing the following fields:
            - anchors (`Union[TextField, ListField[TextField]]`) :
                If `self.sample_spans`, this will be a `ListField[TextField]` object, containing
                each anchor span sampled from `text`. Otherwise, this will be a `TextField` object
                containing the tokenized `text`.
            - positives (`ListField[TextField]`) :
                If `self.sample_spans`, this will be a `ListField[TextField]` object, containing
                each positive span sampled from `text`. Otherwise this field will not be included
                in the returned `Instance`.
        """
        # Some very minimal preprocessing to remove whitespace, newlines and tabs.
        # We peform it here as it will cover both training and predicting with the model.
        # We DON'T lowercase by default, but rather allow `self._tokenizer` to decide.
        text = sanitize(text, lowercase=False)

        # 12446 total iteration * 4 batch size for OpenWebtext
        # difficulty_step = int(self.instance /49784) + 1
        instance = self.instance

        self.instance += 1
        
        fields: Dict[str, Field] = {}
        if self.sample_spans :
            anchor_text, positive_text = sample_anchor_positive_pairs(
                text=text,
                num_anchors=self._num_anchors,
                num_positives=self._num_positives,
                max_span_len=self._max_span_len,
                min_span_len=self._min_span_len,
                sampling_strategy=self._sampling_strategy,
                declutr=self.declutr,
            )
            augmented_list =[]
            anchors: List[Field] = []
            for text in anchor_text:
                tokens = self._tokenizer.tokenize(text)
                anchors.append(TextField(tokens, self._token_indexers))
            print("number of token is", len(tokens))
            fields["anchors"] = ListField(anchors)
            fields["difficulty"] = LabelField(instance, skip_indexing=True)  
            fields["num_lines"] = LabelField(num_lines, skip_indexing=True)  
            if self.declutr: 
                positives: List[Field] = []
                for text in positive_text:
                    tokens = self._tokenizer.tokenize(text)
                    positives.append(TextField(tokens, self._token_indexers))
                fields["positives"] = ListField(positives)

            if self.eda:
                positives: List[Field] = [] 
                difficulty = int(instance/ int(num_lines/6))
                alpha = 0.1 * difficulty 
                for text in anchor_text:
                    sentence_list = sent_tokenize(text)
                    for sentence in sentence_list:
                        if len(sentence) <= 1:
                            continue
                        augmented_sentence = eda(sentence, alpha=alpha, num_aug=1)  
                        augmented_list.append(augmented_sentence) 
                    text = " ".join(map(str, augmented_list))
                    tokens = self._tokenizer.tokenize(text)
                    positives.append(TextField(tokens, self._token_indexers))
                fields["positives"] = ListField(positives)   
        else:
            tokens = self._tokenizer.tokenize(text)
            print("number of token is", len(tokens))
            fields["anchors"] = TextField(tokens, self._token_indexers)
            fields["difficulty"] = LabelField(instance, skip_indexing=True)
            fields["num_lines"] = LabelField(num_lines, skip_indexing=True) 
        return Instance(fields)
