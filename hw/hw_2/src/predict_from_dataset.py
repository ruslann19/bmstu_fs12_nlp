from src.bert_regressor import BertRegressor
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset


def predict_from_dataset(
    model_path: str,
    dataset: Dataset,
) -> np.ndarray:
    model = BertRegressor.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Устанавливаем формат
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Создаем DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Переносим на устройство
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = outputs["logits"].cpu().numpy()
            all_predictions.append(predictions)

    # Объединяем предсказания
    predictions = np.concatenate(all_predictions, axis=0)

    return predictions
