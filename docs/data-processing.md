# Processamento de Dados

Este documento detalha como o sistema processa e prepara os dados para o fine-tuning do modelo PTT5.

## Pipeline de Processamento

### 1. Carregamento dos Dados

```python
# Carregamento do arquivo Excel
df = pd.read_excel(dataset_path)
logger.info(f"Dados carregados: {len(df)} registros")
```

**Formato suportado:**
- Arquivo Excel (.xlsx)
- Encoding UTF-8
- Estrutura tabular com colunas específicas

### 2. Validação e Limpeza

```python
# Validação das colunas obrigatórias
required_columns = ['texto', 'carta', 'evento', 'secao', 'tema']
df = validate_dataframe(df, required_columns)
```

**Validações aplicadas:**
- Verificação de colunas obrigatórias
- Remoção de registros com valores nulos
- Limpeza de espaços em branco
- Normalização de texto

### 3. Divisão Estratificada

```python
# Divisão mantendo proporção de cartas
train_df, val_df = train_test_split(
    df, 
    test_size=0.15, 
    random_state=42, 
    stratify=df['carta']
)
```

**Vantagens da estratificação:**
- Mantém distribuição equilibrada
- Garante representatividade
- Evita viés em conjuntos pequenos

## Função de Validação

### Implementação Completa

```python
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

    return df
```

### Limpeza de Texto

```python
def clean_text(text):
    """
    Limpa e normaliza texto
    
    Args:
        text: Texto original
    
    Returns:
        Texto limpo ou None se inválido
    """
    if pd.isna(text):
        return None
    
    # Conversão para string
    text = str(text)
    
    # Remoção de espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    
    # Remoção de espaços nas extremidades
    text = text.strip()
    
    # Verificação de texto vazio
    if not text:
        return None
    
    return text
```

## Preparação para Fine-tuning

### Aplicação de Templates

```python
def prepare_data_for_fine_tuning(dataframe, prompt_template):
    """
    Prepara dados aplicando template de prompt
    
    Args:
        dataframe: DataFrame com dados limpos
        prompt_template: Template com placeholders
    
    Returns:
        Lista de pares input-output
    """
    data = []
    
    for _, row in dataframe.iterrows():
        # Formatação do prompt
        prompt = prompt_template.format(
            carta=row['carta'],
            evento=row['evento'],
            secao=row['secao'],
            tema=row['tema']
        )
        
        # Criação do par treino
        data.append({
            "input": prompt,
            "output": row['texto']
        })
    
    return data
```

### Exemplo de Transformação

**Entrada:**
```
carta: "O Mago"
evento: "Ano Novo"
secao: "Geral"
tema: "Trabalho"
texto: "2025 é um ano para focar..."
```

**Saída:**
```
input: "Gere uma interpretação para a carta 'O Mago' em 'Ano Novo', na seção 'Geral', sobre o tema 'Trabalho', com o estilo da pessoa X."
output: "2025 é um ano para focar..."
```

## 🔤 Tokenização

### Processo de Tokenização

```python
def tokenize_function(examples, tokenizer, max_length):
    """
    Tokeniza exemplos para treinamento
    
    Args:
        examples: Batch de exemplos
        tokenizer: Tokenizador do modelo
        max_length: Comprimento máximo
    
    Returns:
        Dicionário com tokens e labels
    """
    # Tokenização dos inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    # Tokenização dos outputs
    labels = tokenizer(
        text_target=examples["output"],
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    # Adição dos labels
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs
```

### Configurações de Tokenização

```python
# Parâmetros da tokenização
tokenizer_config = {
    "max_length": 450,          # Comprimento máximo
    "truncation": True,         # Truncar se necessário
    "padding": False,           # Padding no DataCollator
    "add_special_tokens": True  # Tokens especiais
}
```

## Estatísticas dos Dados

### Métricas Calculadas

```python
def calculate_data_statistics(df):
    """
    Calcula estatísticas dos dados
    
    Args:
        df: DataFrame processado
    
    Returns:
        Dicionário com estatísticas
    """
    stats = {
        'total_samples': len(df),
        'unique_cartas': df['carta'].nunique(),
        'unique_eventos': df['evento'].nunique(),
        'unique_secoes': df['secao'].nunique(),
        'unique_temas': df['tema'].nunique(),
        'avg_text_length': df['texto'].str.len().mean(),
        'min_text_length': df['texto'].str.len().min(),
        'max_text_length': df['texto'].str.len().max(),
        'median_text_length': df['texto'].str.len().median()
    }
    
    return stats
```

### Distribuição dos Dados

```python
def analyze_data_distribution(df):
    """
    Analisa distribuição dos dados
    """
    print("=== DISTRIBUIÇÃO DOS DADOS ===")
    
    # Distribuição por carta
    print("\nDistribuição por carta:")
    carta_counts = df['carta'].value_counts()
    print(carta_counts)
    
    # Distribuição por evento
    print("\nDistribuição por evento:")
    evento_counts = df['evento'].value_counts()
    print(evento_counts)
    
    # Estatísticas de comprimento
    print("\nEstatísticas de comprimento:")
    length_stats = df['texto'].str.len().describe()
    print(length_stats)
```

## Validação de Qualidade

### Verificações Automáticas

```python
def validate_data_quality(df):
    """
    Valida qualidade dos dados
    
    Args:
        df: DataFrame a ser validado
    
    Returns:
        Relatório de qualidade
    """
    issues = []
    
    # Verificar textos muito curtos
    short_texts = df[df['texto'].str.len() < 50]
    if len(short_texts) > 0:
        issues.append(f"Textos curtos (<50 chars): {len(short_texts)}")
    
    # Verificar textos muito longos
    long_texts = df[df['texto'].str.len() > 1000]
    if len(long_texts) > 0:
        issues.append(f"Textos longos (>1000 chars): {len(long_texts)}")
    
    # Verificar distribuição de cartas
    min_samples_per_carta = df['carta'].value_counts().min()
    if min_samples_per_carta < 5:
        issues.append(f"Cartas com poucos exemplos: {min_samples_per_carta}")
    
    # Verificar duplicatas
    duplicates = df.duplicated(subset=['carta', 'evento', 'secao', 'tema']).sum()
    if duplicates > 0:
        issues.append(f"Registros duplicados: {duplicates}")
    
    return issues
```

## Visualização dos Dados

### Análise Estatística

Para visualizar a distribuição dos dados, você pode usar bibliotecas como matplotlib e seaborn. O sistema não inclui funções de visualização pré-implementadas, mas as estatísticas básicas são calculadas automaticamente durante o processamento.

## Otimizações de Processamento

### Processamento Eficiente

O sistema processa os dados de forma eficiente usando:

- **Pandas**: Para manipulação rápida de DataFrames
- **Processamento em lote**: Durante a tokenização (configurado no DataLoader)
- **Validação incremental**: Verificações aplicadas conforme necessário

Para otimizações adicionais, você pode implementar:
- Cache de dados processados usando pickle ou parquet
- Processamento paralelo para datasets muito grandes
- Validação assíncrona para melhor performance

## Próximos Passos

Após o processamento dos dados:

1. **Validar** qualidade dos dados processados
2. **Analisar** distribuições e estatísticas
3. **Configurar** parâmetros de treinamento
4. **Executar** tokenização
5. **Iniciar** fine-tuning

---

**Próximo**: [Arquitetura do Sistema](architecture.md)