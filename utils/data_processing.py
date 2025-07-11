# utils/data_processing.py
import re
import pandas as pd
from datasets import Dataset

def clean_text(text):
    """Remove múltiplos espaços e espaços no início/fim."""
    if pd.isna(text):
        return None
    return re.sub(r'\s+', ' ', str(text)).strip()

def validate_dataframe(df, required_columns):
    """Valida a presença de colunas essenciais e remove linhas com valores nulos."""
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame faltando colunas necessárias: {required_columns}")
    
    original_rows = len(df)
    df = df.dropna(subset=required_columns)
    if len(df) < original_rows:
        print(f"ATENÇÃO: {original_rows - len(df)} linhas removidas devido a valores nulos em colunas essenciais.")
    
    # Exemplo de validação de conteúdo: garantir que 'texto' não está vazio após limpeza
    df['texto'] = df['texto'].apply(clean_text)
    df = df.dropna(subset=['texto'])
    if len(df) < original_rows: # Nova verificação após limpeza
        print(f"ATENÇÃO: Mais {original_rows - len(df)} linhas removidas após limpeza de texto vazio.")

    # Aqui poderíamos adicionar validações mais complexas, como:
    # - Validação de formato de 'evento', 'secao', 'tema'
    # - Verificação de diversidade ou balanceamento de 'arcano_maior' (para qualidade dos dados)
    return df

def prepare_data_for_fine_tuning(dataframe, prompt_template):
    """Prepara os dados para o fine-tuning com base em um template de prompt."""
    data = []
    for _, row in dataframe.iterrows():
        prompt_treinamento = prompt_template.format(
            arcano_maior=row['arcano_maior'],
            evento=row['evento'],
            secao=row['secao'],
            tema=row['tema']
        )
        data.append({"input": prompt_treinamento, "output": row['texto']})
    return data

def tokenize_function(examples, tokenizer, max_length):
    """Função de tokenização para o dataset."""
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    labels = tokenizer(
        text_target=examples["output"],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
