from typing import Callable, Dict, List, Optional

import torch
from mmengine.config import ConfigDict
from torch.utils.data import DataLoader

from opencompass.registry import ICL_PROMPT_TEMPLATES, TEXT_POSTPROCESSORS
from opencompass.utils import build_model_from_cfg
from opencompass.utils.text_postprocessors import first_number_postprocess


class LMEvaluator:

    def __init__(self,
                 prompt_template: ConfigDict,
                 model_cfg: ConfigDict,
                 postprocessor: Optional[Callable] = None) -> None:
        self.prompt_tmpl = ICL_PROMPT_TEMPLATES.build(prompt_template)
        self.model = build_model_from_cfg(model_cfg=model_cfg)
        self.max_out_len = model_cfg.get('max_out_len', None)
        self.batch_size = model_cfg.get('batch_size', None)
        if postprocessor:
            self.postprocessor = TEXT_POSTPROCESSORS.build(postprocessor)
        else:
            self.postprocessor = first_number_postprocess

    @staticmethod
    def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
        """Return a dataloader of the input data list."""
        dataloader = DataLoader(datalist,
                                batch_size=batch_size,
                                collate_fn=lambda x: x)
        return dataloader

    def score(self, predictions, references: Optional[List] = None) -> Dict:
        scores = {}
        prompts = []
        print(predictions)
        exit()
        if references:
            for pred, ref in zip(predictions, references):
                prompts.append(
                    self.prompt_tmpl.generate_item(
                        dict(prediction=pred, reference=ref)))
        else:
            for pred in predictions:
                prompts.append(
                    self.prompt_tmpl.generate_item(dict(prediction=pred)))

        for i, entry in enumerate(self.get_dataloader(prompts,
                                                      self.batch_size)):
            print(entry)
            continue
            with torch.no_grad():
                results = self.model.generate_from_template(
                    entry, max_out_len=self.max_out_len)
                results = map(first_number_postprocess, results)
                scores.update({
                    k: v
                    for k, v in zip(
                        range(i * self.batch_size, (i + 1) *
                              self.batch_size), results)
                })
        return scores
