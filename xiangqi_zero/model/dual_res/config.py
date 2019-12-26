import json

from xiangqi_zero.model.utils.model_config import BasicConfig
from xiangqi_zero.env.common import *


class ModelConfigDualRes(BasicConfig):

    def __init__(self, **kwargs):
        super(ModelConfigDualRes, self).__init__(**kwargs)
        model_config_json = kwargs.get('model_config_json')
        if model_config_json is not None:
            with open(model_config_json, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            self.num_res_blocks = kwargs.get('num_res_blocks', 39)  # 19 or 39 in the paper
