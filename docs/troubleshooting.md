# Troubleshooting

Este guia ajuda a resolver problemas comuns durante a instalação e uso do sistema de fine-tuning PTT5.

## Problemas de Instalação

### Erro de Dependências Python

**Problema:**
```bash
ERROR: Could not find a version that satisfies the requirement torch>=1.9.0
```

**Soluções:**
```bash
# 1. Atualizar pip
pip install --upgrade pip

# 2. Instalar PyTorch específico para sua CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Usar conda (alternativa)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Erro de CUDA

**Problema:**
```bash
RuntimeError: CUDA out of memory
```

**Soluções:**
```yaml
# Reduzir batch size em training_config.yaml
training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
```

```python
# Limpar cache da GPU
import torch
torch.cuda.empty_cache()
```

### Erro de Transformers

**Problema:**
```bash
ImportError: cannot import name 'BitsAndBytesConfig' from 'transformers'
```

**Soluções:**
```bash
# Atualizar transformers
pip install --upgrade transformers>=4.30.0

# Verificar versão
python -c "import transformers; print(transformers.__version__)"
```

## Problemas de Dados

### Arquivo não Encontrado

**Problema:**
```bash
FileNotFoundError: [Errno 2] No such file or directory: './database/db_462.xlsx'
```

**Soluções:**
```bash
# Verificar estrutura de pastas
ls -la database/

# Criar pasta se não existir
mkdir -p database

# Verificar caminho no config
cat config/training_config.yaml | grep dataset_path
```

### Erro de Colunas

**Problema:**
```bash
ValueError: DataFrame faltando colunas necessárias: ['texto', 'carta', 'evento', 'secao', 'tema']
```

**Soluções:**
```python
# Verificar colunas do Excel
import pandas as pd
df = pd.read_excel("database/db_462.xlsx")
print("Colunas encontradas:", df.columns.tolist())

# Renomear colunas se necessário
df = df.rename(columns={
    'arcano_maior': 'carta',
    'interpretacao': 'texto'
})
```

### Dados Inválidos

**Problema:**
```bash
ATENÇÃO: 45 linhas removidas devido a valores nulos
```

**Soluções:**
```python
# Identificar linhas com problemas
df = pd.read_excel("database/db_462.xlsx")
print("Linhas com valores nulos:")
print(df.isnull().sum())

# Verificar registros específicos
null_rows = df[df.isnull().any(axis=1)]
print(null_rows)
```

## Problemas de Treinamento

### Erro de Memória GPU

**Problema:**
```bash
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Soluções por GPU:**

| GPU | VRAM | Configuração Recomendada |
|-----|------|-------------------------|
| RTX 3060 | 12GB | batch_size=1, accumulation=4 |
| RTX 3070 | 8GB | batch_size=1, accumulation=2 |
| RTX 4080 | 16GB | batch_size=2, accumulation=4 |

```yaml
# Configuração para GPU com pouca memória
training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  gradient_checkpointing: true
  max_length: 256  # Reduzir sequência
```

### Erro de Quantização

**Problema:**
```bash
RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
```

**Soluções:**
```bash
# Atualizar bitsandbytes
pip install --upgrade bitsandbytes

# Verificar compatibilidade CUDA
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### Treinamento Muito Lento

**Problema:**
Treinamento demora mais de 5 minutos por época

**Soluções:**
```yaml
# Otimizações de velocidade
training_args:
  dataloader_num_workers: 2
  dataloader_pin_memory: true
  fp16: true
  
model_config:
  max_length: 256  # Reduzir se possível
```

## Problemas de Métricas

### Métricas Sempre Zero

**Problema:**
```bash
📊 ROUGE-L Final: 0.0000
📊 BLEU Final: 0.0000
```

**Diagnóstico:**
```python
# Verificar decodificação
def debug_metrics():
    # Verificar se predições estão sendo decodificadas
    print("Amostras decodificadas:", len(decoded_preds))
    if len(decoded_preds) > 0:
        print("Primeira predição:", decoded_preds[0][:100])
        print("Primeira referência:", decoded_labels[0][:100])
```

**Soluções:**
```yaml
# Ajustar parâmetros de geração
training_args:
  eval_accumulation_steps: 1  # Reduzir acumulação
  
model_config:
  max_length: 450  # Verificar comprimento
```

### Overfitting

**Problema:**
```bash
Epoch 10: Train Loss: 0.50, Eval Loss: 2.20
```

**Soluções:**
```yaml
# Regularização
lora_config:
  lora_dropout: 0.2  # Aumentar dropout

training_args:
  weight_decay: 0.01  # Adicionar weight decay
  
early_stopping:
  patience: 3  # Reduzir paciência
```

## Problemas de Configuração

### Erro de YAML

**Problema:**
```bash
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Soluções:**
```bash
# Verificar sintaxe YAML
python -c "import yaml; yaml.safe_load(open('config/training_config.yaml'))"

# Validar indentação (usar espaços, não tabs)
# Verificar dois pontos após chaves
```

### Conflito de Tipos

**Problema:**
```bash
TypeError: 'str' object cannot be interpreted as an integer
```

**Soluções:**
```yaml
# Garantir tipos corretos
training_args:
  learning_rate: 0.00002  # Float, não string
  num_train_epochs: 20    # Int, não string
  per_device_train_batch_size: 1  # Int
```

## Problemas de Execução

### Processo Interrompido

**Problema:**
Treinamento para inesperadamente

**Soluções:**
```bash
# Verificar logs
tail -f results/training_log_*.log

# Executar com nohup
nohup python train_model.py > training_output.log 2>&1 &

# Monitorar processo
ps aux | grep train_model
```

### Erro de Permissão

**Problema:**
```bash
PermissionError: [Errno 13] Permission denied: './results/'
```

**Soluções:**
```bash
# Ajustar permissões
chmod 755 results/
mkdir -p results model_save

# Verificar proprietário
ls -la results/
```

## Ferramentas de Diagnóstico

### Comandos de Teste

Use estes comandos para testar o sistema:

```bash
# Teste 1: Verificar Python e bibliotecas
python -c "import torch; print('PyTorch OK')"
python -c "import transformers; print('Transformers OK')"
python -c "import yaml; print('YAML OK')"

# Teste 2: Verificar módulos do projeto
python -c "from utils.data_processing import validate_dataframe; print('Utils OK')"
python -c "from prompts.pessoa_x_prompt import PESSOA_X_FINE_TUNING_PROMPT; print('Prompts OK')"

# Teste 3: Verificar configuração
python -c "import yaml; yaml.safe_load(open('config/training_config.yaml')); print('Config OK')"

# Teste 4: Verificar dataset (se existir)
python -c "import pandas as pd; df = pd.read_excel('database/db_462.xlsx'); print(f'Dataset: {len(df)} registros')"

# Teste 5: Verificar CUDA
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

### Monitoramento Simples

```bash
# Monitorar recursos do sistema
htop  # CPU e RAM
nvidia-smi  # GPU (se disponível)

# Monitorar logs do treinamento
tail -f results/training_log_*.log
```

## Problemas Críticos

### Modelo Não Salva

**Problema:**
```bash
RuntimeError: Model saving failed
```

**Soluções:**
```bash
# Verificar espaço em disco
df -h

# Criar pasta manualmente
mkdir -p model_save

# Verificar permissões
ls -la model_save/
```

### Perda de Dados

**Problema:**
Arquivos de treinamento corrompidos

**Soluções:**
```bash
# Backup automático
cp config/training_config.yaml config/training_config_backup.yaml

# Verificar integridade
python -c "import pandas as pd; df = pd.read_excel('database/db_462.xlsx'); print('OK')"
```

## 📞 Suporte Adicional

### Informações para Suporte

Ao reportar problemas, inclua:

1. **Versão do Python**: `python --version`
2. **Versão do PyTorch**: `python -c "import torch; print(torch.__version__)"`
3. **Versão do CUDA**: `nvidia-smi`
4. **Sistema operacional**: `uname -a`
5. **Arquivo de configuração**: `cat config/training_config.yaml`
6. **Logs de erro**: Últimas 50 linhas do log

### Coleta de Informações

Para reportar problemas, colete estas informações:

```bash
# Informações básicas do sistema
echo "Data: $(date)"
echo "Python: $(python --version)"
echo "Sistema: $(uname -a)"

# Informações de GPU (se disponível)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# Espaço em disco
df -h .

# Estrutura do projeto
ls -la config/
ls -la database/

# Versões das bibliotecas
pip list | grep -E "(torch|transformers|pandas|peft)"
```

## Recursos Adicionais

### Documentação Útil

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### Comunidade

- [Hugging Face Forum](https://discuss.huggingface.co/)
- [PyTorch Forum](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)

---

**Voltar ao**: [README principal](../README.md)