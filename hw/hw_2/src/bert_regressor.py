from torch import nn
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from src.weighted_mse_loss import create_loss_fn


# Создаем конфигурацию для нашей модели
class BertRegressorConfig(PretrainedConfig):
    model_type = "BertRegressor"

    def __init__(
        self,
        hf_model_name="seara/rubert-tiny2-russian-sentiment",
        loss_type="WeightedMSELoss",
        classes_counts={},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hf_model_name = hf_model_name
        self.loss_type = loss_type
        self.classes_counts = classes_counts
        # Загружаем конфиг базовой BERT модели
        self.bert_config = AutoConfig.from_pretrained(hf_model_name)


# Наследуемся от PreTrainedModel для совместимости с HF
class BertRegressor(PreTrainedModel):
    config_class = BertRegressorConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(
            config.hf_model_name,
            config=config.bert_config,
        )
        # self.regressor = nn.Sequential(
        #     nn.Linear(config.bert_config.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, 1),
        # )
        self.regressor = nn.Linear(config.bert_config.hidden_size, 1)

        self.loss_fn = create_loss_fn(config.loss_type, config.classes_counts)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(cls_tokens).squeeze(-1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
