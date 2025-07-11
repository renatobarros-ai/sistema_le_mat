# utils/data_processing.py
# Módulo para processamento e preparação de dados para fine-tuning
# Autor: Renato Barros
# Email: falecomrenatobarros@gmail.com
# Data: 2025
# Descrição: Funções utilitárias para limpeza, validação e preparação de dados

import re
import pandas as pd
from datasets import Dataset

def clean_text(text):
    """
    Limpa o texto removendo espaços múltiplos e espaços em branco desnecessários
    
    Args:
        text (str): Texto a ser limpo
        
    Returns:
        str or None: Texto limpo ou None se input for nulo
    """
    if pd.isna(text):
        return None
    return re.sub(r'\s+', ' ', str(text)).strip()

def validate_dataframe(df, required_columns):
    """
    Valida a estrutura do DataFrame e remove registros com valores nulos
    
    Args:
        df (pd.DataFrame): DataFrame a ser validado
        required_columns (list): Lista de colunas obrigatórias
        
    Returns:
        pd.DataFrame: DataFrame validado e limpo
        
    Raises:
        ValueError: Se alguma coluna obrigatória estiver ausente
    """
    # Verificação da presença das colunas obrigatórias
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame faltando colunas necessárias: {required_columns}")
    
    # Registro do número original de linhas
    original_rows = len(df)
    
    # Remoção de linhas com valores nulos nas colunas essenciais
    df = df.dropna(subset=required_columns)
    if len(df) < original_rows:
        print(f"ATENÇÃO: {original_rows - len(df)} linhas removidas devido a valores nulos em colunas essenciais.")
    
    # Limpeza e validação do conteúdo de texto
    df['texto'] = df['texto'].apply(clean_text)
    df = df.dropna(subset=['texto'])
    
    # Verificação adicional após limpeza de texto
    if len(df) < original_rows:
        print(f"ATENÇÃO: Mais {original_rows - len(df)} linhas removidas após limpeza de texto vazio.")

    # Área para validações adicionais específicas do domínio
    # Aqui podem ser adicionadas validações como:
    # - Verificação de formato de campos específicos
    # - Validação de balanceamento de classes
    # - Controle de qualidade dos dados
    
    return df

def prepare_data_for_fine_tuning(dataframe, prompt_template):
    """
    Prepara os dados para fine-tuning aplicando o template de prompt
    
    Args:
        dataframe (pd.DataFrame): DataFrame com os dados originais
        prompt_template (str): Template de prompt com placeholders
        
    Returns:
        list: Lista de dicionários com estrutura {"input": prompt, "output": texto}
    """
    data = []
    
    # Iteração sobre cada linha do DataFrame
    for _, row in dataframe.iterrows():
        # Formatação do prompt usando os dados da linha
        prompt_treinamento = prompt_template.format(
            carta=row['carta'],
            evento=row['evento'],
            secao=row['secao'],
            tema=row['tema']
        )
        
        # Criação do par input-output para o fine-tuning
        data.append({
            "input": prompt_treinamento, 
            "output": row['texto']
        })
    
    return data

def tokenize_function(examples, tokenizer, max_length):
    """
    Função de tokenização para preparar os dados para o modelo
    
    Args:
        examples (dict): Batch de exemplos com chaves "input" e "output"
        tokenizer: Tokenizador do modelo
        max_length (int): Comprimento máximo das sequências
        
    Returns:
        dict: Dicionário com tokens de entrada e labels tokenizados
    """
    # Tokenização dos prompts de entrada
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_length,
        truncation=True,
        padding=False  # Padding será feito pelo DataCollator
    )
    
    # Tokenização dos textos de saída (targets)
    labels = tokenizer(
        text_target=examples["output"],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    # Adição dos labels aos inputs do modelo
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs