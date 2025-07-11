# Configuração do Sistema

Este guia explica como configurar e personalizar o sistema de fine-tuning para suas necessidades específicas.

## 📋 Arquivo de Configuração Principal

O arquivo `config/training_config.yaml` contém todas as configurações do sistema:

```yaml
# Estrutura principal
model_config:      # Configurações do modelo
lora_config:       # Parâmetros LoRA
training_args:     # Argumentos de treinamento
early_stopping:    # Parada antecipada
dataset_path:      # Caminho dos dados
test_size:         # Tamanho do conjunto de validação
random_state:      # Seed para reprodutibilidade
```

## Configurações do Modelo

### Modelo Base

```yaml
model_config:
  name: "unicamp-dl/ptt5-base-portuguese-vocab"
  max_length: 450
```

**Modelo utilizado:**
- `ptt5-base`: Modelo base português otimizado (configurado no YAML)

### Quantização 4-bit

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "torch.float16"
```

**Parâmetros:**
- `load_in_4bit`: Habilita quantização (sempre true)
- `bnb_4bit_use_double_quant`: Dupla quantização (recomendado)
- `bnb_4bit_quant_type`: Tipo de quantização (`nf4` ou `fp4`)
- `bnb_4bit_compute_dtype`: Tipo de dados (`float16` ou `bfloat16`)

## Configurações LoRA

### Parâmetros Principais

```yaml
lora_config:
  r: 16                    # Rank das matrizes
  lora_alpha: 32           # Fator de escala
  lora_dropout: 0.1        # Dropout para regularização
  bias: "none"             # Configuração de bias
  task_type: "SEQ_2_SEQ_LM"
```

### Módulos Alvo

```yaml
target_modules: ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
```

**Explicação dos módulos:**
- `q`, `k`, `v`: Matrizes de atenção (Query, Key, Value)
- `o`: Projeção de saída da atenção
- `wi_0`, `wi_1`: Camadas feed-forward
- `wo`: Saída do feed-forward

### Ajustes por Cenário

| Cenário | r | alpha | dropout | Motivo |
|---------|---|-------|---------|---------|
| **Dados poucos** | 8 | 16 | 0.1 | Previne overfitting |
| **Dados médios** | 16 | 32 | 0.1 | Configuração padrão |
| **Dados muitos** | 32 | 64 | 0.05 | Maior capacidade |

## Configurações de Treinamento

### Batch Size e Acumulação

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

**Cálculo do batch efetivo:**
```
Batch efetivo = batch_size × accumulation_steps × num_gpus
```

**Ajustes por GPU:**

| GPU | VRAM | Batch Size | Accumulation |
|-----|------|------------|-------------|
| RTX 3060 | 12GB | 1 | 8 |
| RTX 3070 | 8GB | 1 | 4 |
| RTX 4080 | 16GB | 2 | 8 |
| RTX 4090 | 24GB | 4 | 8 |

### Learning Rate

```yaml
learning_rate: 0.00002
lr_scheduler_type: "cosine"
warmup_ratio: 0.15
```

**Valores recomendados:**
- **Pequeno dataset**: 0.00001 (mais conservador)
- **Médio dataset**: 0.00002 (padrão)
- **Grande dataset**: 0.00003 (mais agressivo)

### Épocas e Salvamento

```yaml
num_train_epochs: 20
save_steps: 60
eval_strategy: "epoch"
save_strategy: "epoch"
```

**Estratégias:**
- `"epoch"`: Avalia/salva a cada época
- `"steps"`: Avalia/salva a cada N steps
- `"no"`: Não avalia durante o treinamento

## Configurações de Avaliação

### Métricas

```yaml
metric_for_best_model: "eval_rougeL"
greater_is_better: true
load_best_model_at_end: true
```

**Métricas disponíveis:**
- `eval_rougeL`: ROUGE-L (recomendado)
- `eval_bleu`: BLEU
- `eval_loss`: Perda de validação

### Early Stopping

```yaml
early_stopping:
  patience: 4
  threshold: 0.001
```

**Configuração:**
- `patience`: Épocas sem melhoria antes de parar
- `threshold`: Melhoria mínima considerada

## Configurações de Dados

### Dataset

```yaml
dataset_path: "./database/db_462.xlsx"
test_size: 0.15
random_state: 42
```

**Parâmetros:**
- `dataset_path`: Caminho para o arquivo Excel
- `test_size`: Proporção para validação (0.1 = 10%)
- `random_state`: Seed para reprodutibilidade

### Formato dos Dados

O Excel deve conter estas colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `evento` | string | Contexto do evento |
| `carta` | string | Nome da carta |
| `tema` | string | Tema da interpretação |
| `secao` | string | Seção específica |
| `texto` | string | Texto de saída esperado |

## Configurações de Prompts

### Template Principal

```python
# Em prompts/pessoa_x_prompt.py
PESSOA_X_FINE_TUNING_PROMPT = (
    "Gere uma interpretação para a carta '{carta}' "
    "em '{evento}', na seção '{secao}', "
    "sobre o tema '{tema}', com o estilo da pessoa X."
)
```

### Personalizações

```python
# Exemplo de template personalizado
TEMPLATE_FORMAL = (
    "Forneça uma análise detalhada da carta '{carta}' "
    "no contexto de '{evento}', seção '{secao}', "
    "focando em '{tema}'."
)

TEMPLATE_CASUAL = (
    "Que tal uma interpretação da carta '{carta}' "
    "para '{evento}' em '{secao}', "
    "pensando em '{tema}'?"
)
```

## Configurações Avançadas

### Otimizações de Memória

```yaml
# Para GPUs com pouca memória
gradient_checkpointing: true
dataloader_pin_memory: true
max_grad_norm: 1.0
```

### Otimizações de Velocidade

```yaml
# Para treinamento mais rápido
fp16: true
dataloader_num_workers: 2
remove_unused_columns: false
```

### Configurações de Logging

```yaml
logging_steps: 8
report_to: "none"
```

## Configurações por Cenário

### Cenário 1: GPU Limitada (4-6GB)

```yaml
model_config:
  max_length: 256
training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  fp16: true
```

### Cenário 2: GPU Média (8-12GB)

```yaml
model_config:
  max_length: 450
training_args:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
```

### Cenário 3: GPU Potente (16GB+)

```yaml
model_config:
  max_length: 512
training_args:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  gradient_checkpointing: false
  fp16: true
```

## Configurações Experimentais

### Testes A/B

```yaml
# Arquivo: config/experimental_config.yaml
experimental:
  different_optimizer: "adamw_hf"
  custom_scheduler: "polynomial"
  advanced_quantization:
    bnb_4bit_quant_type: "fp4"
```

### Configurações Debug

```yaml
# Para debugging
training_args:
  max_steps: 100
  eval_steps: 10
  save_steps: 10
  logging_steps: 1
```

## Ferramentas de Configuração

### Validação de Configuração

```python
# Script para validar configurações
def validate_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validações
    assert config['model_config']['max_length'] > 0
    assert config['training_args']['learning_rate'] > 0
    assert config['test_size'] < 1.0
    
    print("✅ Configuração válida!")
```

### Calculadora de Batch

```python
def calculate_effective_batch(device_batch, accumulation, num_gpus=1):
    return device_batch * accumulation * num_gpus

# Exemplo
effective_batch = calculate_effective_batch(1, 8, 1)
print(f"Batch efetivo: {effective_batch}")
```

## 📝 Backup e Versionamento

### Backup de Configurações

```bash
# Criar backup antes de alterações
cp config/training_config.yaml config/training_config_backup.yaml
```

### Configurações Versionadas

```
config/
├── training_config.yaml        # Atual
├── training_config_v1.yaml     # Versão 1
├── training_config_v2.yaml     # Versão 2
└── experimental_config.yaml    # Experimentos
```

## Próximos Passos

Após configurar o sistema:

1. **Validar** as configurações
2. **Testar** com dataset pequeno
3. **Monitorar** o treinamento
4. **Ajustar** conforme necessário
5. **Documentar** as alterações

---

**Próximo**: [Guia de Uso](usage.md)