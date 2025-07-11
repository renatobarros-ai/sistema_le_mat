# 🔧 Guia de Instalação

Este guia detalha como configurar o ambiente e instalar todas as dependências necessárias para o sistema de fine-tuning PTT5.

## 📋 Pré-requisitos

### Hardware Mínimo
- **GPU**: NVIDIA com 4GB+ VRAM (recomendado 8GB+)
- **RAM**: 8GB mínimo (16GB+ recomendado)
- **Armazenamento**: 10GB livres
- **CPU**: Qualquer processador moderno

### Software Base
- **Python**: 3.8 ou superior
- **CUDA**: 11.8 ou superior
- **Git**: Para clonar o repositório

## 🚀 Instalação Passo a Passo

### 1. Verificar Ambiente

Primeiro, verifique se o Python e CUDA estão instalados:

```bash
# Verificar Python
python --version
# Deve retornar Python 3.8+

# Verificar CUDA (se disponível)
nvidia-smi
# Deve mostrar informações da GPU
```

### 2. Clonar o Repositório

```bash
# Clone o repositório
git clone https://github.com/renatobarros-ai/sistema_le_mat.git
cd sistema_le_mat
```

### 3. Criar Ambiente Virtual

**Recomendado**: Use um ambiente virtual para isolar as dependências:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 4. Instalar Dependências

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Ou instalar manualmente as principais:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate accelerate peft
pip install pandas openpyxl scikit-learn PyYAML
pip install bitsandbytes rouge-score sacrebleu
```

### 5. Verificar Instalação

Execute este script para verificar se tudo está funcionando:

```python
# test_installation.py
import torch
import transformers
import peft
import bitsandbytes
import pandas as pd

print("✅ Verificação de instalação:")
print(f"Python: {torch.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

## 🔧 Configuração do Ambiente

### Variáveis de Ambiente

O sistema configura automaticamente as seguintes variáveis:

```bash
# Configuradas automaticamente no código
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
```

### Estrutura de Pastas

Após a instalação, certifique-se de que as seguintes pastas existam:

```bash
# Criar pastas necessárias
mkdir -p results
mkdir -p model_save
mkdir -p database
```

## 🗂️ Preparação dos Dados

### Dataset

1. **Formato**: O sistema espera um arquivo Excel (`.xlsx`) com as colunas:
   - `carta`: Nome da carta
   - `evento`: Contexto do evento
   - `secao`: Seção específica
   - `tema`: Tema da interpretação
   - `texto`: Texto de saída esperado

2. **Localização**: Coloque o arquivo em `database/db_462.xlsx`

3. **Exemplo de estrutura**:
   ```
   carta        | evento    | secao    | tema      | texto
   O Mago       | Ano Novo  | Geral    | Trabalho  | Interpretação...
   A Imperatriz | Natal     | Amor     | Família   | Interpretação...
   ```

### Configuração

O arquivo `config/training_config.yaml` já está configurado com valores otimizados, mas você pode ajustar:

- **Batch size**: Ajuste conforme sua GPU
- **Learning rate**: Modifique se necessário
- **Epochs**: Controle a duração do treinamento

## 🚨 Problemas Comuns

### Erro de CUDA
```bash
# Se encontrar erro de CUDA, instale a versão correta do PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erro de Memória
```bash
# Reduza o batch size em training_config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
```

### Erro de Dependências
```bash
# Atualize o pip e tente novamente
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔄 Próximos Passos

Após a instalação bem-sucedida:

1. **Verifique** se o dataset está no formato correto
2. **Configure** os parâmetros em `training_config.yaml`
3. **Execute** o treinamento com `python train_model.py`
4. **Monitore** o progresso nos logs

## 📊 Verificação Final

Execute esta checklist antes de iniciar o treinamento:

- [ ] Python 3.8+ instalado
- [ ] CUDA funcionando (se usando GPU)
- [ ] Todas as dependências instaladas
- [ ] Dataset no formato correto
- [ ] Pastas criadas
- [ ] Configuração ajustada

Se todos os itens estiverem marcados, você está pronto para iniciar o treinamento!

## 🆘 Suporte

Se encontrar problemas durante a instalação:

1. Consulte o [guia de troubleshooting](troubleshooting.md)
2. Verifique se sua versão do Python/CUDA é compatível
3. Abra uma issue no repositório com detalhes do erro

---
**Próximo**: [Configuração do Sistema](configuration.md)