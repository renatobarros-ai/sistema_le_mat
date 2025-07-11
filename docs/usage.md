# Guia de Uso

Este guia explica como usar o sistema de fine-tuning PTT5, desde a execução básica até a interpretação de resultados.

## Execução Básica

### Comando Principal

```bash
# Executar o treinamento
python train_model.py
```

### Monitoramento em Tempo Real

```bash
# Executar com output detalhado
python train_model.py | tee training_output.log

# Monitorar em outra janela
tail -f results/training_log_*.log
```

## Fases do Treinamento

### 1. Inicialização

```
=== INICIANDO FINE-TUNING PTT5 - SISTEMA DE CARTAS ===
1. Carregando configurações de treinamento...
Configurações carregadas para o modelo: unicamp-dl/ptt5-base-portuguese-vocab
```

**O que acontece:**
- Carregamento das configurações YAML
- Verificação de arquivos e dependências
- Configuração do sistema de logging

### 2. Processamento de Dados

```
2. Carregando e processando dataset...
Dados carregados: 462 registros
Dados após validação/limpeza: 462 registros válidos
Cartas únicas: 22
Comprimento médio: 156.7 caracteres
```

**Métricas importantes:**
- **Registros válidos**: Quantidade após limpeza
- **Cartas únicas**: Diversidade do dataset
- **Comprimento médio**: Complexidade dos textos

### 3. Divisão dos Dados

```
Dataset dividido: 392 treino, 70 validação
Datasets criados: 392 treino, 70 validação
```

**Divisão estratificada:**
- Mantém proporção de cartas em treino/validação
- Evita vazamento de dados
- Garante representatividade

### 4. Carregamento do Modelo

```
4. Carregando modelo base e configurando quantização...
Modelo carregado: unicamp-dl/ptt5-base-portuguese-vocab com quantização 4-bit
```

**Processo:**
- Download do modelo (primeira vez)
- Aplicação da quantização 4-bit
- Configuração dos adaptadores LoRA

### 5. Configuração LoRA

```
5. Configurando LoRA...
Parâmetros treináveis configurados
trainable params: 2,359,296 || all params: 225,801,216 || trainable%: 1.0446
```

**Interpretação:**
- Apenas ~1% dos parâmetros são treináveis
- Redução massiva de recursos necessários
- Mantém qualidade do modelo base

### 6. Tokenização

```
6. Tokenizando dados...
Tokenização concluída: 392 treino, 70 validação
```

**Transformação:**
- Texto → tokens numéricos
- Aplicação de padding/truncation
- Criação de máscaras de atenção

### 7. Treinamento

```
10. INICIANDO TREINAMENTO...
VRAM disponível: 12.0GB
Batch efetivo: 8
```

**Informações do hardware:**
- VRAM disponível para o treinamento
- Batch efetivo calculado
- Otimizações aplicadas

## Interpretando o Progresso

### Métricas Durante o Treinamento

```
Epoch 1/20
  Train Loss: 2.1234
  Eval Loss: 1.9876
  Eval ROUGE-L: 0.1234
  Eval BLEU: 0.0987
```

**Interpretação:**
- **Train Loss**: Deve diminuir consistentemente
- **Eval Loss**: Deve diminuir sem overfitting
- **ROUGE-L**: Deve aumentar (0.0-1.0)
- **BLEU**: Deve aumentar (0.0-1.0)

### Progressão Saudável

```
Epoch 1: Train Loss: 2.12, Eval Loss: 1.98, ROUGE-L: 0.12
Epoch 5: Train Loss: 1.45, Eval Loss: 1.52, ROUGE-L: 0.25
Epoch 10: Train Loss: 1.02, Eval Loss: 1.15, ROUGE-L: 0.38
Epoch 15: Train Loss: 0.85, Eval Loss: 1.08, ROUGE-L: 0.42
```

**Sinais positivos:**
- Loss de treino diminui suavemente
- Loss de validação segue o treino
- ROUGE-L aumenta consistentemente
- Não há grande divergência entre treino/validação

### Sinais de Alerta

```
# Overfitting
Epoch 10: Train Loss: 0.50, Eval Loss: 1.20, ROUGE-L: 0.15

# Underfitting
Epoch 20: Train Loss: 2.00, Eval Loss: 2.10, ROUGE-L: 0.05

# Instabilidade
Epoch 15: Train Loss: 1.20, Eval Loss: 3.50, ROUGE-L: 0.01
```

## Resultados Finais

### Relatório de Conclusão

```
=== RESULTADOS FINAIS ===
📊 ROUGE-L Final: 0.4250
📊 BLEU Final: 0.2180
📊 Loss Final: 1.0850
📁 Métricas detalhadas salvas em: ./results/training_metrics_20250711_143022.json
```

### Interpretação das Métricas

| Métrica | Faixa | Qualidade |
|---------|-------|-----------|
| **ROUGE-L** | 0.0-0.2 | Baixa |
| | 0.2-0.4 | Média |
| | 0.4-0.6 | Boa |
| | 0.6+ | Excelente |
| **BLEU** | 0.0-0.1 | Baixa |
| | 0.1-0.3 | Média |
| | 0.3-0.5 | Boa |
| | 0.5+ | Excelente |

### Salvamento do Modelo

```
13. Salvando modelo treinado...
⏳ Salvando adaptadores LoRA... (pode demorar 1-2 minutos)
⏳ Salvando tokenizer...
✅ Modelo salvo com sucesso em: ./model_save/lora_model_462_optimized
```

## Arquivos Gerados

### Estrutura de Resultados

```
results/
├── training_log_20250711_143022.log        # Log completo
├── training_metrics_20250711_143022.json   # Métricas detalhadas
├── checkpoint-120/                         # Checkpoint intermediário
└── checkpoint-240/                         # Checkpoint final
```

### Modelo Salvo

```
model_save/lora_model_462_optimized/
├── adapter_config.json                     # Configuração LoRA
├── adapter_model.bin                       # Pesos dos adaptadores
├── tokenizer.json                          # Tokenizer
├── tokenizer_config.json                   # Configuração do tokenizer
└── special_tokens_map.json                 # Tokens especiais
```

## Análise Detalhada

### Arquivo de Métricas JSON

```json
{
  "timestamp": "20250711_143022",
  "hardware_info": {
    "gpu": "NVIDIA GeForce RTX 4080",
    "vram": "16.0GB",
    "cuda_version": "11.8"
  },
  "dataset_info": {
    "total_samples": 462,
    "train_samples": 392,
    "val_samples": 70,
    "unique_cartas": 22,
    "avg_text_length": 156.7
  },
  "model_config": {
    "model_name": "unicamp-dl/ptt5-base-portuguese-vocab",
    "lora_r": 16,
    "lora_alpha": 32,
    "max_length": 450,
    "epochs_completed": 20
  },
  "training_metrics": {
    "train_runtime": 1234.56,
    "train_samples_per_second": 2.5,
    "total_flos": 1.23e15,
    "train_loss": 0.8543
  },
  "final_eval_metrics": {
    "eval_loss": 1.0850,
    "eval_rougeL": 0.4250,
    "eval_bleu": 0.2180
  }
}
```

## Usando o Modelo Treinado

### Usando o Modelo Treinado

Após o treinamento, o modelo LoRA será salvo em `./model_save/lora_model_462_optimized/`. Para usar o modelo:

1. **Carregamento**: O modelo contém os adaptadores LoRA treinados
2. **Tokenizer**: Salvo junto com o modelo para compatibilidade
3. **Inferência**: Use bibliotecas padrão do Hugging Face para geração

**Estrutura dos arquivos salvos:**
```
model_save/lora_model_462_optimized/
├── adapter_config.json          # Configuração LoRA
├── adapter_model.bin           # Pesos dos adaptadores
├── tokenizer.json              # Tokenizer
└── tokenizer_config.json       # Configuração do tokenizer
```

Para implementar a inferência, consulte a documentação oficial do PEFT e Transformers sobre como carregar e usar modelos LoRA.

## Ajustes Durante o Treinamento

### Early Stopping

```
Early stopping triggered at epoch 12
Best model restored from checkpoint-180
```

**Quando acontece:**
- Métricas param de melhorar
- Evita overfitting
- Economiza tempo de treinamento

### Interrupção Manual

```bash
# Pressionar Ctrl+C para interromper
^C
⚠️ Treinamento interrompido pelo usuário (Ctrl+C)
💾 Salvando checkpoint de emergência...
✅ Checkpoint de emergência salvo em: ./model_save/emergency_checkpoint_20250711_143022
```

## Monitoramento de Recursos

### Uso de GPU

Para monitorar o uso de GPU durante o treinamento:

```bash
# Monitorar GPU em tempo real
watch -n 1 nvidia-smi

# Verificar uso de memória
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

O sistema exibe automaticamente informações de VRAM no início do treinamento.

### Tempo de Treinamento

```
Tempo estimado por época: 2-3 minutos
Tempo total estimado: 40-60 minutos
```

**Fatores que afetam o tempo:**
- Tamanho do dataset
- Poder da GPU
- Configurações de batch
- Comprimento das sequências

## Próximos Passos

Após o treinamento bem-sucedido:

1. **Avaliar** resultados nas métricas
2. **Testar** o modelo com exemplos
3. **Ajustar** configurações se necessário
4. **Documentar** os experimentos
5. **Implementar** em produção

---

**Próximo**: [Troubleshooting](troubleshooting.md)